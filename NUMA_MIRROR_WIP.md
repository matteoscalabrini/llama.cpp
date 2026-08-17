# NUMA weight mirroring for the CPU expert phase — LXC 111 (2× EPYC 7532)

Branch `numa-mirror-20260816`, worktree `/root/llama.cpp-numa`, based on mainline `6a32c29a7` (b10297).
Status as of 2026-08-17: **implemented, correctness-gated bit-exact, +60% on the CPU-only
instrument.** Full-model hybrid A/B results at the bottom.

## What it does

Keeps one full replica of each large host weight buffer per NUMA node. CPU GEMM threads read
the replica belonging to the node they are running on; everything else (activations, KV, wdata,
GPU uploads) is untouched.

- `tensor->data` always stays the **primary** buffer, so the CUDA op-offload upload path,
  `--fit` accounting and every non-GEMM consumer see exactly what they saw before.
- Only two call sites change what pointer they read weights from: `ggml_compute_forward_mul_mat`
  (incl. both llamafile-sgemm calls) and `ggml_compute_forward_mul_mat_id` — **MUL_MAT_ID is the
  one that matters here, our CPU phase is MoE expert matmuls.**
- Off by default. When off, `ggml_numa_mirror_remap()` is one relaxed atomic load returning its
  argument, so the vanilla path is unchanged.

## Why it is correct

Replicas are byte-identical `memcpy`s, read by the same kernels doing the same arithmetic in the
same order. The result is not "close", it is **bit-identical** — which is what the gate measured
(320/320 top-20 logprobs identical). A thread migrating between nodes mid-op is harmless: it just
reads the other identical copy. Anything less than bit-exact means the offset math is wrong.

## Design (differs from roblee04/numa_llamacpp in three ways)

| | roblee04 reference | here |
|---|---|---|
| discovery | allocator hook, `GGML_NUMA_REPLICATE` compile flag | lazy scan of the graph at `graph_compute`, runtime env only |
| coverage | `mul_mat` (dense models) | `mul_mat` **+ `mul_mat_id`** + both llamafile sgemm sites |
| repack | requires `GGML_CPU_REPACK=OFF` | compatible with repack ON — see below |

Discovery by graph scan (`ggml_numa_mirror_scan_graph`, called single-threaded from
`ggml_backend_cpu_graph_compute`) means no loader/allocator surgery: any host buffer feeding a
CPU GEMM and larger than `GGML_NUMA_MIRROR_MIN_MB` (default 1024) gets a replica on first compute.

**Repack is a non-issue on this box, verified by reading `repack.cpp`:** repacking only covers
Q4_0/Q4_K/Q2_K/Q5_K/Q6_K/Q8_0. GLM's experts are IQ3_XXS/IQ4_XS (Q3_K_XL) or Q3_K — none are
repack types, so no expert tensor is ever repacked. Had they been, repacked tensors are consumed
by the extra-buffer op path which returns before reaching the hooked GEMMs, so a replica would
have wasted RAM but not broken correctness.

## Usage

```
GGML_NUMA_MIRROR=1 [GGML_NUMA_MIRROR_MIN_MB=1024] numactl --membind=0 llama-server ...
```

**Do not run mirror mode under `numactl --interleave=all`** — the allocator binds explicitly and
interleave defeats the primary's placement. `--membind=0` is the right prefix: it puts the primary
on node 0 (which is also where all 4 GPUs live, so op-offload H2D stays socket-local) and the
replica is built on node 1.

RAM: 2× the CPU-resident weights. 302 GB GLM ⇒ ~604 GB of 1007 GB. A Q4_K_XL-class model
(~410 GB) would NOT leave comfortable headroom. `ggml_numa_mirror_alloc_replica` refuses and
falls back to the primary if a node has less than size + 2 GB free, so the failure mode is
"slower", not OOM.

## Measurements

### The premise, tested directly (truncated 10-layer GLM Q3_K_M, CPU-only, `-mmp 0 -r 3`)

| arm | threads | CPUs | weights in | tg128 | pp512 |
|---|---|---|---|---|---|
| A1 / A2 | 64 | both sockets | interleaved | 19.38 / 19.33 | 183.4 |
| D | 64 | both sockets | node0 (half remote) | 15.06 | 172.6 |
| B | 32 | node0 | node0 — local | 26.39 | 121.7 |
| C | 32 | node1 | node1 — local | 27.72 | 129.6 |
| **E** | 32 | node0 | node1 — **all remote** | **12.46** | 110.4 |

**B vs E is the clean measurement: same socket, same threads, only the weights move — local is
2.12× remote.** The expert phase is locality-bound. Note this does *not* contradict the
"decode is not bandwidth-bound" finding (17.4 GB/s = 14% of bus, averaged over the whole token):
remote-access latency and xGMI throughput bite long before the DRAM controllers saturate, and the
whole-token average hides the CPU-phase burst.

Also note the prefill/decode split: t32 single-socket wins decode (+36%) but loses 33% of
prefill, because prefill scales with cores. Any single-socket config is a decode/prefill trade;
the mirror is not.

### Mirror A/B (same binary, same session, one variable)

| arm | tg128 | pp512 |
|---|---|---|
| mirror OFF, t64 interleave (prod shape) | 19.29 | 181.5 |
| t32 node0 local (best config-only option) | 26.10 | 123.2 |
| **mirror ON, t64** | **30.89** | **184.3** |

**+60.1% decode over the prod shape, prefill flat (+1.5%, noise).** It also beats the best
config-only topology by 18% on decode while keeping 50% more prefill throughput.

Engagement proof (peak resident by node, `numastat -p`): mirror ON = 28.1 GB node0 + 17.1 GB
node1 = 45.3 GB total (replica mid-population); mirror OFF = 14.2 + 14.2 = 28.4 GB (textbook
interleave). Correctness gate: **BIT-EXACT PASS**, 320/320 top-20 logprobs identical.

### Full model (GLM-5.2 UD-Q3_K_XL, 302 GB, hybrid, prod flags)

Prod flags verbatim, same binary, decode measured before prefill, first two decode runs
discarded as warmup, decode figure = mean of runs 3-5.

**Unpinned regime (`GGML_CUDA_NO_PINNED=1`, what prod runs today) — 3 arms, one session:**

| arm | decode tg | prefill pp | resident (node0/node1) |
|---|---|---|---|
| `--interleave=all`, no mirror — **today's prod shape** | 7.20 | 85.4 | 146.9 / 146.9 |
| `--membind=0`, no mirror | 6.12 | 95.5 | 293.7 / 0.0 |
| **`--membind=0` + mirror** | **9.06** | **108.4** | 293.7 / 292.8 |

**+25.8% decode and +26.9% prefill against today's prod shape.** MTP draft acceptance was
0.92308 in every arm — byte-identical accepted/generated counts, i.e. the model is behaving
identically, not merely similarly.

The middle arm is the attribution control and it matters:

- `--membind=0` *alone* **costs 15% of decode** (7.20 → 6.12). That is arm D from the premise
  table at full scale: all weights on one node means half the threads read across xGMI.
- So the decode win belongs to the mirror, not the placement flag: **6.12 → 9.06 = +48% at
  identical placement.**
- Prefill splits the other way: `--membind=0` alone is worth +12% (85.4 → 95.5) because all four
  GPUs hang off socket 0, so op-offload H2D staging becomes socket-local. The mirror then adds a
  further +13.5% (95.5 → 108.4) — most plausibly because moving the CPU threads' expert reads onto
  node 1 stops them contending with the GPU's DMA stream on node 0's memory controller.

Cost: 586.6 GB resident instead of 293.8 GB, and ~20 s of extra load time for the replica memcpy
(load-to-ready 403 s vs 383 s).

**Pinned regime (no `GGML_CUDA_NO_PINNED`, where prod is heading) — 3 more arms, same session:**

| arm | decode tg | prefill pp | resident (node0/node1) |
|---|---|---|---|
| `--interleave=all`, no mirror | 7.44 | 148.3 | 153.0 / 153.0 |
| `--interleave=all` + mirror — **silently no-ops, see below** | 7.28 | 147.6 | 153.1 / 153.0 |
| **`--membind=0` + mirror** | **9.02** (runs 3-5: 9.35 / 9.61 / 8.10) | 146.8 | 306.0 / 292.9 |

Pinned decode gain **+21%** on the mean, +26% on the median; prefill flat (−1%), because pinning
has already solved prefill and there is nothing left for socket-locality to win there.

**So the mirror is worth ~+21-26% decode in both regimes.** The regimes differ only in what
happens to prefill: unpinned it gains +27%, pinned it is neutral.

Best measured configuration overall — **pinned + `--membind=0` + mirror: 9.02-9.35 tg / 146.8 pp**,
against today's production shape (unpinned + interleave) at 7.20 tg / 85.4 pp: **+25-30% decode
and +72% prefill.** That combines this work with the other thread's pinning finding; neither
alone gets there.

### `--interleave=all` makes the mirror silently do nothing — use `--membind=0`

The pinned+interleave+mirror arm built **no replica at all** (306.1 GB resident, a plain 153/153
split) and duly performed like the baseline. Two independent causes, both worth knowing:

1. `ggml_numa_mirror_alloc_replica` refuses a node with less than `size + 2 GB` in
   `/sys/devices/system/node/nodeN/meminfo` `MemFree`. Under interleave node 1 already holds half
   the model plus page cache, so the check fails even though the kernel would happily reclaim
   cache. **MemFree ignores reclaimable memory — the check is too conservative.**
2. Pinned (`cudaMallocHost`) pages cannot be migrated, so `mbind(MPOL_MF_MOVE)` on the primary
   cannot repair an interleaved layout after the fact. It also is not free: that arm's
   load-to-ready was 742 s vs 528 s, ~3.5 minutes spent walking 302 GB of VMAs to accomplish
   nothing.

Both disappear with `--membind=0`, which places the primary correctly at allocation time instead
of trying to move it afterwards. **`--membind=0` is required, not a preference.**

The failure is silent because `GGML_LOG_INFO` from a dynamically loaded backend does not reach
llama-server's log at default verbosity — none of the "NUMA mirror:" lines appear anywhere.
**Verify engagement with `numastat -p $(pgrep -x llama-server)` instead: mirrored means total
resident ≈ 2× the model with a full copy on each node.** Fixing this properly (log to stderr
directly, and count reclaimable memory in the free check) is open item 0 below; the binary that
produced every number above was deliberately left unrebuilt so the results stay reproducible.

## Recommended production change (measured, not promoted — prod config left untouched)

GLM-5.2 entry in `/etc/llama-swap/config-ik.yaml`: replace `numactl --interleave=all` with
`numactl --membind=0`, add `GGML_NUMA_MIRROR=1`, and drop `GGML_CUDA_NO_PINNED=1` (the other
thread's finding). Expected 7.20 → ~9.0-9.4 tg and 85 → ~147 pp. Costs ~590 GB RAM instead of
~300 GB, and ~9 minutes to load instead of ~6.5.

Not promoted here because the pinning half belongs to another session's thread and prod entries
are a shared surface; the two changes should land together, with a date-suffixed backup.

## Interaction with the pinned-memory promotion (READ THIS BEFORE PROMOTING EITHER)

The other campaign thread found `GGML_CUDA_NO_PINNED=1` costs 45% of prefill and wants it removed
(+81% pp). That change and this one **interact**:

- Pinned (`cudaMallocHost`) pages cannot be migrated — `mbind(MPOL_MF_MOVE)` on the primary will
  fail, so under pinning the primary stays wherever it was allocated.
- Under `numactl --interleave=all` + pinning, the primary is interleaved and only node-1 threads
  gain locality (node-0 threads keep reading a half-remote primary): a half-mirror.
- **The fix needs no code:** launch with `numactl --membind=0` instead of `--interleave=all`.
  The pinned primary then lands entirely on node 0 — which is GPU-local for all four cards, so
  op-offload H2D gets *better*, not worse — and node 1 reads its replica.

So the combined prod recipe is `--membind=0` + pinning + `GGML_NUMA_MIRROR=1`, not
`--interleave=all` + anything. This has not yet been measured; it is the top open item.

## Open items

0. **Make engagement observable and the free check honest.** Log to stderr directly rather than
   through `GGML_LOG_INFO` (nothing currently reaches llama-server's log), and count reclaimable
   memory instead of bare `MemFree` in `ggml_numa_mirror_alloc_replica`. Also consider skipping
   the `MPOL_MF_MOVE` attempt when the primary is unmigratable — it cost 3.5 minutes of load time
   for nothing in the interleave arm.
1. ~~Measure `--membind=0` + pinned + mirror~~ — done, see the pinned table above.
2. `-t 32 -tb 64` with `-Cr/--cpu-range`: prefill wants 64 threads, decode wants fewer. Untested
   and orthogonal to the mirror. The premise table shows the split clearly (t32 local wins decode
   by 36% and loses prefill by 33%), so a per-phase thread count should compose with the mirror.
3. Buffer table is fixed at 16 entries and replicas are never freed before process exit —
   fine for one-model-per-process, wrong for a long-lived multi-model host.
4. If a 5th GPU lands on socket 1, the op-offload upload source should become that socket's
   replica; today all uploads read the primary (node 0), which is correct while all GPUs are
   on socket 0.
5. Upstream shape: ggerganov's stated direction (#19102) is a backend device per NUMA node.
   A mirrored *buffer type* would fit that; this env-gated remap is deliberately the smallest
   thing that could prove the number.

## Files

- `ggml/src/ggml-cpu/ggml-cpu.c` — the mirror itself (registration, replicas, remap, graph scan)
- `ggml/src/ggml-cpu/ggml-cpu-impl.h` — declarations
- `ggml/src/ggml-cpu/ggml-cpu.cpp` — one call in `ggml_backend_cpu_graph_compute`
- Harnesses in `/root/`: `numa_precheck2.sh` (locality premise), `numa_gate.sh` +
  `numa_verify.sh` (engagement + bit-exactness + truncated perf), `numa_full_ab.sh` (full model)
