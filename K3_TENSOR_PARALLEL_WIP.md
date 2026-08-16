# Kimi-K3 tensor parallelism — WIP postmortem (2026-08-15)
Goal: -sm tensor for K3 on 4x3090 (attack the serialized GPU phase; proj. 8-12 t/s).
Result: SERVER BOOTS AND RUNS (4.4 t/s) but output is numerically corrupt:
greedy => near-coherent first tokens, then deterministic "scattering" repetition.
Patch: /root/k3_tp_wip_v2_20260815.patch (352 lines, apply to worktree @ PR-head e88142750
+ the pdep patch). Launch script: /root/bench/kimi_launch_tp.sh (.bak copy).

## What was built (all LOCAL PATCH-tagged)
1. llama-arch.cpp: KIMI_K3 removed from sm-tensor exclusion.
2. llama-model.cpp split policy (llama_meta_device_get_split_state):
   - name normalization for mangled statics (Meta(...)#blk.N.x (reshaped)#0)
   - MLA: q_b AXIS_1, k_b/v_b AXIS_2, latent kv-cache MIRRORED
   - recurrent (KDA) layers: conv1d_{q,k,v} AXIS_2, beta/f_b/g AXIS_1, dt/a AXIS_0,
     f_a+ssm_norm MIRRORED, caches AXIS_0; shexp AXIS_1/0 vs ffn_down_shexp
   - K3 granularity override (=1; generic paths SIGFPE on unset ssm_d_state/n_gqa)
   - K3 r-cache segments {ne/3, 3} (three q|k|v conv sections)
3. ggml-backend-meta.cpp:
   - narrow leaf-policy call for compute-usage static ssm_a
   - batched-matmul case: both srcs split on same batch axis>=2 -> return src state
   - flash-attn MQA/MLA case: Q AXIS_2-split + K/V MIRRORED -> AXIS_1
4. kimi-k3.cpp: ggml_cont on inp_embd (host-buffer view fix);
   state write-back view_1d -> shape-preserving view_2d (did NOT fix corruption).

## Walls fixed in order
arch gate -> policy v1 (mirror recr = 39GB OOM, K3 is majority-linear!) ->
policy v2 (split recr; KDA lowers to GGML_OP_GATED_DELTA_NET which meta supports) ->
SIGFPE granularity -> r-cache VIEW mismatch (segments) -> mangled-name policy misses ->
MULMAT batch-axis case -> FA-MQA case -> inp_embd host-buffer assert. ALSO: a resident
production model silently ate VRAM for 4 runs (ttl:0 trap) — unload before TP tests!

## Current bug (open)
Deterministic corruption, position-early. Leads, in order:
1. ssm_a pre-reshaped static (compute-usage buffer): is its DATA actually sliced
   per device when the leaf patch declares it split? Verify the set_tensor path
   sliced it (dump per-device bytes) — prime suspect.
2. r-cache 1-D conv views through segments: verify per-device offset translation
   (causal_conv1d views at qkv*conv_state_size offsets).
3. My 3 meta-backend cases may be state-correct but MATERIALIZATION-wrong
   (per-device subgraph building may need explicit narrow/offset views for the
   FA-MQA and batched-mm cases — check how existing cases materialize).
4. Method for next session: logit-level bisect — same binary, -sm layer vs -sm
   tensor, --dump-logits on 1 token; then per-layer activation compare (cb() hooks).

## Facts for the resume session
- 96 heads/4 devs = 24; all split dims divide evenly; granularity 1 is safe here.
- Weight slices: 21.4 GB/device incl mirrored embd(4.7)+output-slice; fits.
- delta-net handler (meta) asserts srcs AXIS_1 + state axis 0/1/2 — satisfied.
- Layer-split production uses the SAME worktree binary — graph patches (cont,
  view_2d) are ACTIVE in layer mode: verified coherent post-change (see memory).
- Every debug print left in the build is tagged: K3-split-dbg / META-BUF-OK /
  MULMAT-MISMATCH / GENERIC-MISMATCH / VIEW-MISMATCH / SIMPLE-TENSOR-NONMETA /
  LEAF-DBG. Strip before any perf comparison.

## 2026-08-16 session — ROOT CAUSE FOUND VIA GLM_DSA BISECTION INSTRUMENT

**Strategy executed:** ported the MLA policy branch to GLM_DSA (zero KDA layers = control
for the "shared plumbing vs KDA paths" question). GLM corrupted with the same signature
(deterministic repetition, top-20 overlap 3/20) -> bug was in SHARED plumbing. Then
eval-callback layer walk (truncated 5-layer GGUF, ~1 min loads) found first divergence.

**ROOT CAUSE (fixed): ggml-backend-meta.cpp view_offs scaling heuristic.**
`split_internal_offset = view_offs <= view_src->nb[split_dim_view_src]` compares against
nb[0]=4 bytes when view_src is an axis-0-split activation -> ANY intra-row offset gets
scaled by ne_local/ne_global. Concrete victim: q_pe = view(q, offs=nope_dims*ts) + ROPE.
q_pe materialized at offset 192B instead of 768B per device -> RoPE over mid-nope garbage
-> Qcur poisoned -> corruption. K3's MLA layers build the IDENTICAL pattern.
FIX: offset smaller than the view's own nb[split_dim] is intra-tile (repeats per shard),
never scale. After fix: GLM tensor-vs-layer top-20 logprobs 20/20 overlap, max dlogprob
0.00000 (bit-exact); greedy text identical 11 words then fp-order drift (expected).
q_nope (offset 0) never hit the bug — that's why absorption always matched.

**Instruments built (all reusable for K3):**
- /root/glm_tp_probe2.sh TAG {layer|tensor} [extra args] — probe server port 10099,
  guard vs llama-swap residents, MODEL env override. /root/k3_tp_probe.sh for K3.
- /root/models/glm52-trunc5.gguf — real 5-layer GGUF carved via gguf-py
  (/tmp/glm_truncate.py recipe; --override-kv block_count FAILS: loader wants all tensors).
  ~1 min loads. Recipe works for a truncated K3 too.
- llama-eval-callback built + debug.cpp skip for non-contig meta views + GGML_META_SOFT_READS=1
  (NaN-fill unreadable PARTIAL/non-contig states instead of abort). /root/evalcb_diff.py
  aligns streams incl. skip sentinels. CAVEAT: nodes downstream of a delayed AllReduce read
  mid-subgraph garbage in the walk (ffn_inp "divergence" was this artifact) — trust the walk
  only up to the first PARTIAL node; verify end-to-end via server logits.
- GLM_TP_MIRROR=attn|shexp env bisect switches in the policy branch (mirror==split identity
  proved shexp split correct; attn variant asserts in FA case — needs all-mirrored FA case).
- GGML_OP_LIGHTNING_INDEXER meta case added (handle_generic scalar, indexer fully mirrored).

**Patch: /root/k3_glm_tp_wip_v3_20260816.patch (regenerated with all of today).**
K3 tensor-mode retest after the fix: CONFIRMED FIXED same day: coherent text, top-20 overlap 17/20 vs layer (IQ1_S reduction-order drift, not corruption). TP speed 4.07-4.21 tg vs 4.47 layer in the SAME probe config (-7%): per-layer allreduce latency eats the 4-way GPU-phase win at batch-1; speed work = next campaign (delayed-reduce scope, comm path, x8 card). Layer-mode prod coherence on the rebuilt binary verified via the ref probe..
If K3 still corrupts: shared plumbing now PROVEN clean (GLM exact) -> remaining bug is
KDA-only (ssm_a slicing, r-cache conv segment views) -> walk a truncated K3 with the
same instrument.
