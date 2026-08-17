# Result log — NUMA weight mirroring, LXC 111 (2× EPYC 7532, 4× RTX 3090)

**Date:** 2026-08-17 · **Target:** GLM-5.2 CPU expert phase · **Status:** shipped, live in production

---

## 1. Bottom line

Implemented per-NUMA-node replication of CPU-resident weights in llama.cpp, verified bit-exact,
and promoted it to the production GLM-5.2 entry.

**Production, before → after:** decode **7.44 → 9.59 t/s (+29%)**, prefill **148.3 → 158.0 t/s (+6.5%)**,
at the cost of 599 GB resident instead of 306 GB. Output verified identical and coherent.

Measured against where GLM-5.2 started the week (7.20 t/s / 85.4 t/s), production is now
**+33% decode and +85% prefill** — the prefill share belongs to the parallel pinned-memory fix,
the decode share to this work.

---

## 2. The precondition was mis-assessed, and that is the main technical finding

The task was gated on "verify the expert phase is bandwidth-bound first; if kernel-bound, park it."
A prior measurement had recorded DRAM fill counters at **17.4 GB/s = 14% of bus** during live decode
and concluded *not bandwidth-bound ⇒ NUMA mirroring is dead for decode*.

That inference does not hold. **Low aggregate bandwidth utilisation does not imply insensitivity to
placement.** Remote-access latency and the xGMI interconnect bind long before the memory controllers
saturate, and an average taken across a whole token hides the CPU-phase burst.

Tested directly instead of inferred — socket and thread count held constant, only the weights moved
(truncated 10-layer GLM Q3_K_M, CPU-only, `llama-bench -mmp 0 -r 3`):

| arm | threads | CPUs | weights resident on | tg128 | pp512 |
|---|---|---|---|---|---|
| A1 / A2 (repeat) | 64 | both sockets | interleaved | 19.38 / 19.33 | 183.4 |
| D | 64 | both sockets | node 0 (half remote) | 15.06 | 172.6 |
| B | 32 | node 0 | node 0 — local | 26.39 | 121.7 |
| C | 32 | node 1 | node 1 — local | 27.72 | 129.6 |
| **E** | 32 | node 0 | node 1 — **all remote** | **12.46** | 110.4 |

**B vs E is the controlled pair: local memory is 2.12× remote.** The phase is locality-bound. The
premise held; the earlier verdict was reversed on evidence.

Secondary result visible in the same table: prefill and decode want opposite topologies. A single
socket wins decode by 36% and loses prefill by 33%, because prefill scales with cores. The mirror
avoids that trade entirely — it is the only option that gives all 64 cores local reads.

---

## 3. What was built

`ggml-cpu.c` (+~300 lines), `ggml-cpu-impl.h` (declarations), `ggml-cpu.cpp` (one call site).

- One full replica of each large host weight buffer per NUMA node; each CPU GEMM thread reads the
  copy belonging to the node it is executing on (`getcpu()` at op entry).
- `tensor->data` continues to point at the **primary** buffer, so CUDA op-offload uploads, `--fit`
  accounting and every non-GEMM consumer are untouched.
- Hooked at exactly three places: `ggml_compute_forward_mul_mat`, **`ggml_compute_forward_mul_mat_id`**
  (the MoE expert path — the one that matters for this workload), and both llamafile-sgemm call sites.
- Buffers are discovered by a lazy scan of the graph at `graph_compute`, so no loader or allocator
  surgery, and any host buffer feeding a CPU GEMM above `GGML_NUMA_MIRROR_MIN_MB` (default 1024) is
  covered regardless of how it was loaded.
- **Off by default.** When off, the remap is a single relaxed atomic load that returns its argument.
- Pointers outside every registered buffer pass through unchanged (activations, KV, scratch).

Divergences from the reference design (`roblee04/numa_llamacpp`, which targets dense models,
CPU-only, and requires `GGML_CPU_REPACK=OFF`): `MUL_MAT_ID` coverage, runtime env gate instead of a
compile flag, graph-scan discovery instead of an allocator hook, and repack compatibility.

**Repack conflict resolved as a non-issue, by reading `repack.cpp`:** repacking only covers
Q4_0/Q4_K/Q2_K/Q5_K/Q6_K/Q8_0. GLM's experts are IQ3_XXS/IQ4_XS or Q3_K, so no expert tensor is ever
repacked and the build keeps `GGML_CPU_REPACK=ON`. Had they been repacked, those tensors are consumed
by the extra-buffer op path which returns before the hooked GEMMs — wasted RAM, never wrong results.

---

## 4. Correctness

Replicas are byte-identical `memcpy`s consumed by the same kernels in the same order, so the
expected result is bit-exactness, not approximate agreement — a weaker outcome would indicate wrong
offset arithmetic.

- **320/320 top-20 logprobs bit-identical**, mirror on vs off, plus identical generated text.
- **MTP draft acceptance 0.92308 in all six full-model arms** — identical accepted/generated counts.
- Thread migration between nodes mid-op is harmless by construction: the thread reads the other
  identical copy.
- Engagement is proven by residency, not by trust: mirrored = ~2× model resident with a full copy on
  each node; unmirrored = a 50/50 interleave split.

---

## 5. Full-model measurements

GLM-5.2 UD-Q3_K_XL (302 GB), production flags, same binary across arms, decode measured **before**
prefill, first two decode runs discarded as warm-up, decode = mean of runs 3–5, prefill at 9.5k prompt.

**Unpinned (`GGML_CUDA_NO_PINNED=1`):**

| arm | decode | prefill |
|---|---|---|
| `--interleave=all`, no mirror | 7.20 | 85.4 |
| `--membind=0`, no mirror *(control)* | 6.12 | 95.5 |
| `--membind=0` + mirror | 9.06 | 108.4 |

**Pinned (production regime):**

| arm | decode | prefill |
|---|---|---|
| `--interleave=all`, no mirror | 7.44 | 148.3 |
| `--interleave=all` + mirror *(no-op — see §7)* | 7.28 | 147.6 |
| **`--membind=0` + mirror** | **9.02** (median 9.35) | 146.8 |

**Attribution.** The control arm is the load-bearing one. `--membind=0` *alone* **costs 15% of decode**
(7.20 → 6.12): all weights on one node means half the threads read across xGMI. So the decode gain is
the replicas, not the placement flag — **6.12 → 9.06 = +48% at identical placement.**

Prefill attributes the other way: `--membind=0` alone is worth +12% (85.4 → 95.5) because all four
GPUs hang off socket 0, making op-offload H2D staging socket-local; the mirror adds a further +13.5%,
most plausibly by moving CPU expert reads off node 0 so they stop contending with the GPU's DMA stream.
In the pinned regime prefill is already solved and the mirror is neutral there (−1%).

**Net: the mirror is worth +21–26% decode in both regimes.**

---

## 6. Production change

`/etc/llama-swap/config-ik.yaml`, GLM-5.2 entry only:

```diff
-  numactl --interleave=all /root/llama.cpp/build/bin/llama-server
+  env:
+    - GGML_NUMA_MIRROR=1
+  numactl --membind=0 /root/llama.cpp-numa/build/bin/llama-server
```

Backup: `config-ik.yaml.pre-numamirror-20260817`. The edit was applied by a script that aborts unless
it matches exactly one occurrence, then diffed against the backup — prior sessions have had `sed`
silently hit four entries. A comment block in the entry carries the numbers, the `--membind=0`
requirement, the engagement check and the revert path.

**Binary provenance.** `/root/llama.cpp` carried two *uncommitted* local patches (`rr-offload`,
`no-expert-ids`; both default-off `getenv` gates) plus a diagnostic print. A git worktree takes
committed HEAD, so the new worktree lacked them. They were ported and rebuilt before promotion:
the promoted binary is a **strict superset** of what production was serving — same upstream commit
`6a32c29a7`, plus the mirror, plus those patches.

**Live verification through the production path** (llama-swap :8080, not a hand-rolled server):

| check | result |
|---|---|
| running cmd | `numactl --membind=0 /root/llama.cpp-numa/build/bin/llama-server` ✓ |
| mirror engaged (`numastat`) | 599.0 GB total — node0 306.2 / node1 292.8 ✓ |
| decode, n=5 | 9.58 / 9.49 / 9.69 (runs 3–5) |
| prefill @9.5k | 158.0 |
| load to ready | 560 s (was ~528 s) |
| correctness probe | *"Tokyo; 17\*23 = 391."* — `finish_reason: stop`, 10.32 t/s |

---

## 7. Operational traps discovered (these cost real time)

1. **`--interleave=all` makes the mirror silently build nothing.** The pinned+interleave+mirror arm
   showed 306 GB resident in a plain 153/153 split and performed exactly like baseline. Two
   independent causes: (a) the replica allocator refuses a node with less than `size + 2 GB` in
   `MemFree`, and `MemFree` ignores reclaimable page cache; (b) pinned `cudaMallocHost` pages cannot
   be migrated, so `mbind(MPOL_MF_MOVE)` cannot repair an interleaved layout after the fact — it
   merely burned 3.5 minutes of load time walking 302 GB of VMAs. **`--membind=0` is required, not
   preferred**: it places the primary correctly at allocation time.
2. **There is no log line proving engagement.** `GGML_LOG_INFO` emitted from a dynamically loaded
   backend never reaches llama-server's log at default verbosity. Verify with
   `numastat -p $(pgrep -x llama-server)`.
3. **GLM returns empty `content` with `finish_reason: length`** when the reasoning phase consumes the
   token budget; the text is in `reasoning_content`. On a freshly patched inference path this reads
   exactly like corrupted output. Give coherence probes ~700 tokens of headroom.
4. A git worktree does not inherit the parent tree's uncommitted patches — `git status` the tree you
   are replacing before swapping binaries.

---

## 8. Open items

| # | item |
|---|---|
| 1 | **Branch not pushed.** 3 commits on `numa-mirror-20260816` in `/root/llama.cpp-numa`; pushing to the fork needs credentials not present in that worktree. |
| 2 | Make engagement observable (log to stderr directly), count reclaimable memory in the free check, and skip the `MPOL_MF_MOVE` attempt when the primary is unmigratable. The binary that produced every number above was deliberately left unrebuilt so results stay reproducible. |
| 3 | **RAM is now the binding constraint:** 599 GB of 1007 GB while GLM is resident. Swapping is unaffected (llama-swap unloads first), but anything co-resident has ~400 GB, not ~700. A Q4_K_XL-class model (~410 GB) would not fit mirrored. |
| 4 | Per-phase thread counts (`-t 32 -tb 64` with `-Cr/--cpu-range`) are untested and orthogonal — §2 shows decode and prefill want different core counts. |
| 5 | If a 5th GPU lands on socket 1, op-offload upload source should become that socket's replica; today all uploads read the node-0 primary, correct only while all GPUs are on socket 0. |
| 6 | Upstream direction (ggml discussion #19102) is a backend device per NUMA node. A mirrored *buffer type* fits that shape; this env-gated remap was deliberately the smallest thing that could prove the number. |

**Remaining path to the 15 t/s target:** this delivered the CPU-expert-phase share (7.44 → ~9.6).
The measured decomposition is CPU experts ~220 ms + GPU ~110 ms executed serially per forward; the
outstanding levers are CPU/GPU phase overlap (~1.5×) and the per-op OMP barrier/imbalance cost inside
the CPU phase (~1.5–2× on the 220 ms). Both are upstream-shaped engineering, not configuration.

---

## 9. Artifacts

| path | contents |
|---|---|
| `/root/llama.cpp-numa` | worktree, branch `numa-mirror-20260816`, 3 commits, clean tree |
| `/root/llama.cpp-numa/NUMA_MIRROR_WIP.md` | design + full postmortem |
| `/etc/llama-swap/config-ik.yaml.pre-numamirror-20260817` | revert point |
| `/root/numa_precheck2.sh`, `numa_precheck2.log` | locality premise (§2) |
| `/root/numa_gate.sh`, `numa_verify.sh`, `numa_verify.txt` | bit-exactness + engagement + truncated A/B |
| `/root/numa_full_ab.sh`, `numa_attrib.sh`, `numa_pinned_ab.sh`, `numa_full_ab_results.txt` | all six full-model arms |
| `/root/verify_prod.sh`, `prod_verify.txt` | live production verification |

**Revert:** restore the backup config and `systemctl restart llama-swap` as its own command (chaining
it behind a `pkill` has previously left the old config being served, because `pkill -f` self-matches
through `pct exec`).
