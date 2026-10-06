# WRX80 GLM-5.3 inference optimization execution ledger

Last updated: 2026-10-06
Branch: `wrx80/glm53-cuda-resident`
Companion roadmap: `docs/experiments/wrx80-glm53-optimization-roadmap.md`
Raw experiment log: `docs/experiments/wrx80-glm53-resident-2026-10-06.md`

This is the short, resumable execution ledger. Keep it current after every meaningful
measurement, implementation checkpoint, or platform decision. The longer roadmap explains
architecture and history; this file answers **what is left, what can run in parallel, and
what must wait**.

## Current measured state

The current long WRX80 WSL run is healthy and remains the protected baseline.

Latest completed large agent turn in Colibri profile sequence 6:

- wall: **4457.849 s**
- prompt: **19,679 tokens**
- completion: **49 tokens**
- expert disk: **1.569 s**
- expert wait: **0 s**
- expert matmul: **1881.451 s**
- attention: **2473.181 s**
- lm-head: **27.273 s**
- forwards: **155**

Interpretation: warmed steady-state latency is now overwhelmingly **attention + expert
compute**, not storage. Storage work remains important for cold start, prewarm, cache
representation, and platform efficiency, but it is no longer the top steady-state
optimization target for this workload.

Current service:
- WSL Colibri active with one useful inference in flight, no queued work and no failures;
- latest completed profile is `seq=7`: **30,075 prompt + 89 completion tokens**, **4625.896 s wall**, **1458.807 s expert matmul**, **3086.756 s attention**, **1.239 s expert disk**, **170 forwards**;
- ~81.8 GiB WSL memory available, swap 0;
- RTX 4090 roughly 20.8 GiB resident at the latest observation, with live utilization sampled at 20%; CPU was ~794% aggregate.

Do not restart WSL or start a competing full native model until this baseline run reaches
a safe stop/finish.

## Dependency legend

- **I — independent:** useful work can proceed now without waiting for another unit.
- **P — partially independent:** design/tests can proceed, but final qualification depends
  on named predecessors.
- **S — serialized:** must wait for predecessors or exclusive ownership of WRX80.

## Remaining units of work

### U00 — Measurement decomposition and attribution
**Class:** I
**Status:** ACTIVE
**Priority:** P0

Goal: make the remaining attention/expert wall time directly attributable.

Completed checkpoints:
- `151c0897` adds cumulative **indexer** and **router** subphase timers to GLM-5.3 and emits them in the rolling `[PROF]` line without changing the existing serve `PROF` wire contract. CPU engine build and CUDA backend correctness test pass; deployment waits for the next safe service restart.
- `5985dd95` adds opt-in `GLM53_PROFILE_VRAM=1` sampling after attention/FFN boundaries, outside the phase timers, and reports cumulative peak-used/min-free VRAM. CPU and CUDA-linked GLM-5.3 builds pass; deployment likewise waits for a safe restart.
- `6c02e1b2` splits MoE execution into cumulative **shared expert**, **persistent-resident GPU**, **streamed GPU**, and **CPU routed fallback** timers. The profiling flag is cached once so disabled instrumentation does not repeatedly call `getenv` in hot loops. CPU and CUDA-linked builds pass.

Work:
- add H2D/D2H byte + time counters by phase;
- separate synchronization/accumulation overhead where it remains material after the new MoE timers;
- record the **largest transient allocation** per request in addition to sampled VRAM high-water;
- preserve exact commit/environment with each benchmark.

Exit:
- every major >5% wall-time bucket has a measured owner.

### U10 — Transient-aware near-full VRAM residency planner
**Class:** I
**Status:** ACTIVE
**Priority:** P0

Goal: safely drive total 4090 occupancy toward the highest stable 23–24 GiB range.

Work:
- measure dense/MLA/expert scratch high-water after prewarm;
- model allocator/fragmentation reserve rather than using a fixed guess;
- admit hot experts against predicted steady-state high-water;
- long-generation stress at the resulting target.

Exit:
- no OOM/fallback/corruption over repeated long prefill+decode runs at the chosen target.

### U20 — Continuous device-resident activation pipeline
**Class:** I
**Status:** ACTIVE / PARTIAL
**Priority:** P0

Goal: eliminate the host bubbles between current high-utilization CUDA bursts.

Completed checkpoints:
- `e825ed26` adds a tested `coli_cuda_attention_absorbed_sparse_project_batch_dev_out` backend/Windows-ABI primitive. Sparse MLA + resident `o_proj` can now leave `[S,O]` on device instead of forcing its final D2H. CUDA parity and native-Windows loader ABI tests pass. GLM-5.3 caller integration is intentionally deferred until the following hyperconnection boundary can also stay on device.
- `425ba6ce` adds `coli_cuda_pipe_hc_post`, a device-pointer hyperconnection post primitive that keeps branch/residual/post/comb/output on the GPU and sums source streams in the same order as the CPU `coli_hc_post` loop. CUDA numerical parity, Linux header/loader parity, and native-Windows loader ABI (`55 mandatory + 9 optional`) all pass.
- `ddac0bb4` wires an opt-in `COLI_CUDA_GLM53_CHAIN=2` verification bridge through **sparse MLA -> resident `o_proj` dev-out -> device mHC post**. It uploads the current residual/post/comb, keeps the expensive branch on device, downloads only the final residual bank for comparison, and keeps the existing CPU branch/post authoritative. CPU and CUDA-linked GLM builds plus CUDA backend parity pass; runtime drift measurement waits for the next safe service restart.
- `2ffb20f6` adds batch-capable `coli_cuda_pipe_hc_pre`: device residual -> fidelity-first inverse-RMS / `hc_fn` mix projection / sigmoid pre-post coefficients / Sinkhorn -> device collapsed/post/comb. Scalar reduction order matches the CPU reference while independent tokens/rows/columns run in parallel. Two-token CUDA numerical parity passes at the existing `1e-4` gate; CPU and CUDA-linked GLM builds pass; Linux loader/header parity and native-Windows ABI (`56 mandatory + 9 optional`) pass.
- `54887458` wires a verification-only real-geometry **mHC pre -> GPU RMSNorm** caller path under `COLI_CUDA_GLM53_CHAIN=2`. It recomputes attention-site pre/post/comb and normalized input from the real residual bank on-device, downloads only verification outputs, and reports norm/post/comb drift while the CPU path remains authoritative. CPU and CUDA-linked GLM builds plus the CUDA backend suite pass; runtime drift measurement waits for the next safe restart.
- `2c660d2b` keeps the attention-site device residual resident across the site-loop boundary and feeds that exact `H×D` pointer directly into FFN-site device mHC-pre under verification mode. Per-layer mHC/norm weights are persistently resident, so the second site skips both the residual re-upload and repeated ~1.5 MiB `hc_fn` uploads. CPU and CUDA-linked GLM builds plus the CUDA backend suite pass.
- `ba391c9a` makes the **device-normalized `D` row** the explicit host FFN boundary in verification mode. CPU mHC-pre/RMSNorm remains the oracle, but downstream FFN consumes the downloaded device-normalized row instead of depending on the full host residual bank. CPU and CUDA-linked GLM builds plus the CUDA backend suite pass.
- `723e1b73` lets the S=1 CUDA router consume that already-resident normalized device row directly, skipping its hidden-row H2D upload. Router mode 2 still keeps CPU selection/weights authoritative and now reports `dev_in=1` when the transfer-free input is used. CPU/CUDA-linked builds and CUDA backend correctness pass.
- `ae41b638` adds a verification-only S=1 **resident routed-MoE from device row** path using the existing resident expert issue/take backend. It snapshots the current resident-tier host result as oracle, recomputes the same selected resident experts directly from `x_dev`, downloads one D-row contribution, and reports max absolute/relative drift. CPU/CUDA-linked builds and CUDA backend correctness pass.
- `9bee6c1f` adds `coli_cuda_pipe_swiglu_clamped`, matching GLM-5.3's asymmetric SwiGLU contract exactly: positive-only gate clamp, symmetric up clamp, then SiLU(gate)×up. CUDA numerical parity, Linux loader/header parity, and native-Windows ABI (`57 mandatory + 9 optional`) all pass. This is the prerequisite for moving the always-on shared expert onto the resident device row without changing model math.
- `2351d9d1` completes the S=1 **shared expert from resident device row** verification path: resident `rg/ru/rd` matrices run `pipe_gemm -> pipe_gemm -> pipe_swiglu_clamped -> pipe_gemm` from `x_dev`, then one D-row is downloaded and compared with the existing host shared-expert result. CPU and CUDA-linked GLM builds pass; host output remains authoritative until real-run parity.
- `2812d439` refactors the shared-expert helper to **retain its D-row contribution on-device** and moves the verification download to the caller. Numerics are unchanged, CPU/CUDA-linked GLM builds pass, and the device row is now composable with routed-expert contributions before mHC post.
- `9959c261` mirrors that refactor for the S=1 resident routed-expert helper: it now returns its device accumulator, while the caller performs the verification download. CPU/CUDA-linked GLM builds pass; CPU output remains authoritative.
- `5b4824ab` composes the retained shared and resident-routed device D-row contributions with `coli_cuda_pipe_add` and verifies the combined row against the host `out` state before nonresident experts are processed. CPU/CUDA-linked GLM builds pass; GPU output remains qualification-only.
- `6484e9e4` returns a complete device FFN branch only when **all selected routed experts are resident**, then threads that D-row into FFN-site device mHC post using a non-aliasing output scratch slot. CPU/CUDA-linked GLM builds pass; any nonresident selection returns `NULL` and stays on the unchanged host path.
- `f4b59e62` adds rolling S=1 hot-tier coverage telemetry: sparse decode-site sets entering device-row qualification versus sets where every selected routed expert is resident. CPU/CUDA-linked GLM builds pass; the next safe deployment will report `resident_coverage ... pct=...` and decide whether to grow/reorder residency or prioritize streamed-device accumulation.
- `7ec30f01` preserves a successfully verified device residual across the **layer boundary**, allowing the next layer's attention mHC-pre to consume it directly. Any host-only site or GPU-post failure clears residency immediately, preventing stale scratch reuse. CPU/CUDA-linked GLM builds pass.
- `4e3cd6e2` adds rolling **cross-layer residency continuity** telemetry: device mHC-pre site entries versus entries that consumed an already-resident residual. The next verification run will report `chain_residency resident_in=... pre_sites=... pct=...`; CPU and CUDA-linked builds pass.
- `ecbb35bd` moves device mHC-post from the CUDA default stream onto Colibri's per-device **nonblocking stream**, matching sparse MLA dev-out ordering. CUDA backend parity plus CPU/CUDA-linked GLM builds pass. This is a prerequisite for removing the end-of-attention stream-wide synchronize without racing the following post step.
- `29909ce1` moves all device mHC-pre kernels plus both RMSNorm variants onto that same nonblocking device stream. The staged-image CUDA parity test passes on the RTX 4090; concurrent sparse-indexer work was intentionally left unstaged. The residual -> mHC-pre -> RMSNorm -> sparse MLA -> mHC-post chain now has explicit same-stream ordering at its attention boundaries.
- `68ffa6f2` moves the resident FFN chain onto the home device stream: expert-event waits/reduction, shared-expert clamped SwiGLU, shared+routed row addition, and pipe GEMM. The staged-image CUDA parity suite passes on the RTX 4090. This removes the default-stream split between resident FFN compute and FFN mHC-post while preserving deterministic event order.
- `a0d3601c` makes `coli_cuda_pipe_download` explicitly enqueue D2H verification copies on the per-device stream and synchronize only that stream. This removes ambiguity with `cudaStreamNonBlocking` while preserving the synchronous caller contract; the staged-image CUDA parity suite passes on the RTX 4090.
- `a6b627a4` moves the sparse-MLA dev-out synchronization point from **after** MLA+`o_proj` to immediately after its host-input uploads. The host buffers are therefore safe to release, while the expensive attention/projection kernels remain queued asynchronously on the device stream and can flow directly into same-stream mHC-post. CUDA backend parity and CPU/CUDA-linked GLM builds pass.
- `98ac0ed8` removes that upload-phase stream-wide synchronize for the ephemeral query/selection inputs by copying them into backend-owned pinned staging, recording a completion event after their H2D copies, and waiting only when the staging block is reused. A destructive lifetime test overwrites the caller buffers immediately after dev-out returns and still reproduces the reference projection. RTX CUDA parity plus CPU/CUDA-linked GLM builds pass on the combined U20/U40 tree; persistent latent upload remains queued directly from the layer-state buffer.
- `faf40f35` amortizes decode-time sparse latent scratch growth in 8 MiB quanta instead of exact-size `cudaFree`/`cudaMalloc` growth. With `kv_lora_rank=512`, one quantum covers about 4096 additional tokens while adding under 8 MiB of slack, avoiding a potential allocator synchronization at each newly longer decode step. RTX CUDA parity plus CPU/CUDA-linked GLM builds pass.
- `518013e8` makes S=1 sparse MLA upload only the selected latent rows through the existing pinned staging arena. On the real Flash geometry (`index_topk=2048`, `index_kpool=4`, tail enabled, `kv_lora_rank=512`), a 46,676-token context falls from about 91.2 MiB of latent H2D per sparse layer to about 4.0 MiB, roughly 22.8x less transfer; across 42 sparse layers that is about 3.74 GiB -> 168 MiB per forward. Multi-token/prefill keeps the prior full-history path. A dedicated S=1 compact-vs-full parity test plus the RTX CUDA suite and CPU/CUDA-linked GLM builds pass.
- `a4ecafc1` narrows the streamed clamped-expert readiness fence from `cudaDeviceSynchronize()` to `cudaStreamSynchronize(0)`. Lazy int4 upload/conversion is the only dependency and is queued on legacy stream 0; the grouped expert stream is nonblocking, so device-wide synchronization unnecessarily waited for unrelated work. RTX CUDA parity plus CPU/CUDA-linked GLM builds pass.

Work:
- deploy verification mode at the next safe restart and measure real router/chain drift, VRAM high-water, all-resident decode coverage, and cross-layer `resident_in=1` continuity;
- after real-run parity, make the shared expert and device-row resident routed-expert contributions authoritative and avoid their host activation staging;
- preserve residual stream across layer boundaries;
- replace broad synchronizations with dependency-local synchronization.

Exit:
- GPU utilization becomes sustained rather than bursty and wall time falls.

### U30 — GPU-native router and resident MoE decode
**Class:** P
**Depends on:** U20 for best final path; implementation/profiling can start now
**Status:** ACTIVE / DECODE ROUTER IMPLEMENTED, NOT YET DEPLOYED
**Priority:** P0

Goal: remove CPU routing/top-k and keep S=1 decode expert dispatch device-side.

Completed checkpoints:
- `da318128` adds an opt-in `COLI_CUDA_GLM53_ROUTER=1` S=1 router. Router/bias weights are lazily resident on the 4090, only the current hidden row is uploaded, and top-k indices/weights return through the existing device router. Any CUDA allocation/upload/router failure falls back before route tracing to the unchanged CPU path. The backend test explicitly verifies GLM's correction-bias selection versus raw-sigmoid normalized weights; CPU and CUDA-linked GLM builds pass.
- `5b335f09` adds `COLI_CUDA_GLM53_ROUTER=2` qualification mode: GPU routing runs, but the scalar CPU router remains authoritative and selection/weight drift is reported. This is the deployment gate before mode 1 becomes a trusted runtime optimization.
- `0979097e` prewarms all 42 sparse-layer router matrices+biases before the resident-expert tier is sized, eliminating first-decode lazy uploads and making expert residency account for router VRAM first. The checkpoint's raw router payload is **189.1 MiB** for the actual 4096-hidden/288-expert model; startup logs the real `cudaMemGetInfo` charge because allocator padding can be larger. CPU and CUDA-linked builds pass.
- `54c32afb` moves S=1 router logits/select plus its tiny host readback onto the per-device stream and synchronizes only that stream for the host-visible top-k result. CUDA backend parity and CPU/CUDA-linked GLM builds pass; this removes another default-stream boundary from the device-normalized FFN path.

Work:
- deploy mode 2 at the next safe restart and qualify route parity over real decode;
- if parity is clean, switch to mode 1 and measure router wall-time delta;
- direct dispatch to resident experts;
- re-sweep the historical small-group CPU cutoff after H2D weight/activation transfers
  are removed;
- preserve deterministic CPU reference/fallback.

Exit:
- S=1 routing/MoE wall time beats current hybrid policy.

### U40 — GPU sparse indexer + persistent attention state
**Class:** P
**Depends on:** U20 device-resident activation conventions
**Status:** ACTIVE / AUTHORITATIVE DECODE WIRED, NATIVE QUALIFICATION PENDING
**Priority:** P1

Goal: remove the remaining CPU-side sparse-index construction and attention-state traffic.

Completed checkpoints:
- `8aac5c2d` adds an exact decode-first CUDA sparse-index primitive matching `coli_sparse_index_select_range` for leading padding, complete pools, deterministic top-k selection, and incomplete-tail emission. RTX 4090 numerical parity, CPU GLM build, CUDA-linked GLM build, and Linux header/loader parity pass. The public CUDA ABI is wired as mandatory (`58 mandatory + 9 optional`); the Windows-invoked loader fixture still self-skips its Windows-only classes in this environment, so full native DLL fixture execution remains a later qualification item. The primitive is not yet called by GLM inference.
- `906ed70c` wires opt-in `COLI_CUDA_GLM53_INDEXER=2` qualification for S=1 decode. CPU sparse selection remains authoritative; the CUDA indexer recomputes the same 2051-slot selection and reports exact integer parity with cumulative pass/mismatch/failure counters. CPU/CUDA-linked GLM builds and the RTX CUDA backend suite pass. Mode 1 authoritative routing is intentionally deferred until a real-model verification run shows clean parity.
- `b88f2545` replaces the decode indexer's repeated selected-pool rescans with a device-side `taken` bitmap, preserving exact CPU selection semantics while making each top-k iteration O(pools) rather than O(pools * prior-ranks). RTX CUDA parity plus CPU/CUDA-linked GLM builds pass.
- `12a655a6` parallelizes each decode top-pool search across a 256-thread block with deterministic score/index tie-breaking, then broadcasts the chosen pool before the next rank. RTX CUDA parity plus CPU/CUDA-linked GLM builds pass.
- `1eaaad15` adds lifecycle-safe per-layer pooled-index cache state to each CUDA device context, including explicit teardown, without changing runtime behavior.
- `b14a74f8` makes decode sparse-index qualification persist compressed 4-token pool vectors on device and append only newly completed pools. Cache identity includes the layer APE plus host key/gate storage pointers, so host-state reallocation or geometry changes safely reset the cache; coarse 8 MiB/64 KiB cache growth rebuilds rather than copying stale allocations. A stateful CUDA test warms at sequence 8 then extends to sequence 10 with an incomplete appended pool and remains exactly equal to the CPU reference. RTX CUDA parity plus CPU/CUDA-linked GLM builds pass. At the current ~46.7k-token geometry this removes roughly 45.6 MiB of repeated key+gate H2D per sparse layer per qualified decode call after warm-up.
- `8a458bef` persists the decode validity history alongside the pooled index cache, appending only newly visible validity bytes and letting both pool construction and top-k tail handling read the resident mask. This removes the remaining O(context) validity-mask H2D from qualified decode while preserving the stateful incomplete-pool parity test; RTX CUDA parity plus CPU/CUDA-linked GLM builds pass.
- `792108b9` keeps each layer's immutable k-pool compression APE resident beside the pooled-index cache instead of uploading it whenever a new pool completes. RTX CUDA parity plus CPU/CUDA-linked GLM builds pass; this removes another per-pool H2D command at negligible VRAM cost.
- `26bf8671` completes `COLI_CUDA_GLM53_INDEXER=1` authoritative decode plumbing. Mode 1 runs the CUDA indexer first and skips CPU sparse selection on success, but falls back immediately to the unchanged CPU selector on any backend failure; mode 2 remains CPU-authoritative exact-parity qualification. CPU and CUDA-linked GLM builds pass. Do not enable mode 1 until a real-model mode-2 run is clean.
- `548b5269` splits cumulative indexer profiling into `index_proj` and `index_select` while preserving the aggregate `indexer` bucket. The next real-model run can therefore distinguish CPU projection/LayerNorm cost from sparse-selection cost before choosing the next GPU migration target. CPU and CUDA-linked GLM builds pass.
- `6cf9aa76` removes another O(context) host decode sweep: each full-attention layer now owns a session-lifetime all-valid mask instead of malloc+memset of `seen` bytes on every MLA call. Selector semantics, CPU fallback, and the CUDA validity-cache contract are unchanged. RTX 4090 backend numerical parity, CPU GLM build, CUDA-linked GLM build, and `git diff --check` pass.

Work:
- persistent latent/KV/index state in VRAM where capacity permits;
- GPU sparse scoring/selection;
- no per-layer q/latent/selected upload;
- GPU indexer -> sparse MLA -> o_proj -> residual chain.

Exit:
- no bulk host round trip in decode attention.

### U50 — CPU critical-path optimization
**Class:** I initially; P for overlap stage
**Depends on:** U00 for final targeting; U20 for CPU/GPU overlap
**Status:** ACTIVE RESEARCH
**Priority:** P1

Goal: reduce unavoidable host work and overlap it with GPU execution.

Work:
- 16 physical vs 24/32 SMT thread sweeps for prewarm/prefill/decode;
- SIMD/router/indexer reductions;
- top-k algorithm improvement;
- reusable aligned buffers / less allocator churn;
- prefill chunk sweep;
- overlap next-layer route/index metadata and host-expert prefetch with current CUDA work.

Exit:
- CPU is either removed from the critical path or productively overlapped.

### U60 — Decode specialization / stable scratch / CUDA graphs
**Class:** S/P
**Depends on:** U20, U30, and preferably U40 stable
**Status:** PLANNED
**Priority:** P2

Goal: optimize the common S=1/small-batch agent decode loop after pipeline boundaries settle.

Work:
- persistent scratch and pointer stability;
- remove per-token allocation;
- CUDA graph capture where control flow permits;
- fuse tiny residual/norm/router kernels;
- separate decode thresholds from prefill thresholds.

Exit:
- long S=1 generation is stable and materially faster than the generic path.

### U70 — Native Windows production-model qualification

Full-model staging note: the Mac Pro -> WRX80 transfer of `GLM-5.3-colibri-int4-g64` completed successfully on 2026-10-06 (`RSYNC_RC=0`, source total **420.43 GB**) into `E:\\z-models\\GLM-5.3-colibri-int4-g64`; no Internet redownload is required for the later native/full-model qualification.

**Class:** S on host, implementation support otherwise complete
**Depends on:** current WSL baseline run reaching safe stop
**Status:** CURRENT NATIVE BUILD/ABI/NUMERICS PASS; FULL-MODEL A/B NEXT
**Priority:** P0 platform experiment

Current-branch qualification checkpoint:
- `8c7e262d` was fast-forwarded into the clean native checkout at `E:\z-src\colibri-native`; `coli_cuda.dll` rebuilt for `sm_89` and `glm53.exe CUDA_DLL=1 ARCH=native` rebuilt successfully.
- Native Windows `LoaderStubFixtureTest`: 12/12 pass; ABI remains 58 mandatory + 9 optional.
- Native Windows `backend_cuda_test.exe`: RTX 4090 q8/q4/q2/f32/e8 correctness passes, including the expected deliberate OOM diagnostic.
- `objdump -p coli_cuda.dll` confirms exports for `coli_cuda_sparse_index_select_decode`, `coli_cuda_attention_absorbed_sparse_project_batch_dev_out`, and `coli_cuda_pipe_hc_post`.
- No new WSL model inference was launched during this qualification.

Why first:
- no second 195 GB model copy is needed;
- native build, CUDA DLL ABI, loader suite, real-model metadata/plan/doctor, and guarded
  launcher are already qualified;
- tests storage, Windows scheduling/CPU behavior, and native file mapping at once.

Serialized test order:
1. save final WSL baseline profile/environment;
2. stop WSL inference cleanly;
3. run native Windows baseline with the same fixed prompt/generation;
4. run identical native arm with `-MapExperts`;
5. compare cold load, prewarm, prefill, warmed decode, attention/expert buckets, CPU duty,
   GPU duty, and wall time;
6. only then raise native residency to the aggressive target if the constrained run is sane.

Decision:
- if native Windows materially wins warmed wall time with comparable correctness, adopt it
  as the primary WRX80 runtime and treat WSL storage work as secondary;
- if it mainly improves cold/prewarm I/O but not warmed wall time, keep platform choice open
  and proceed to U80/U90.

### U80 — WSL Plan 9 -> VirtioFS transport A/B
**Class:** S
**Depends on:** current run stopped; preferably U70 measured first
**Status:** STAGED, NOT ACTIVE
**Priority:** P1 platform experiment

Work:
- record current Plan 9/DrvFS baseline;
- enable staged VirtioFS candidate;
- restart WSL only in an explicit experiment window;
- prove the new transport with `findmnt`;
- run the same 8-thread per-thread/shared/map iobench arms and fixed Colibri workload.

Exit:
- quantified cold/warm difference versus Plan 9.

### U90 — Dedicated ext4 VHDX on E: control arm
**Class:** S
**Depends on:** U70; U80 useful but not strictly required
**Status:** PLANNED
**Priority:** P1/P2 depending on U70

Goal: test Linux-native filesystem semantics on the same physical SN850X.

Work:
- create a dedicated VHDX on E:;
- attach with supported `wsl --mount --vhd`;
- format ext4;
- copy the model once;
- run the exact same storage + Colibri workload.

Decision:
- this is the preferred Linux control if native Windows does not clearly win;
- skip/deprioritize the expensive copy if native Windows wins decisively and WSL is no
  longer operationally important.

### U95 — Cache-reclaim and host power-policy A/B
**Class:** S
**Depends on:** a chosen WSL transport for the reclaim arm; no active inference during
host-policy changes
**Status:** STAGED
**Priority:** P2

Arms:
- `autoMemoryReclaim=gradual`;
- `autoMemoryReclaim=disabled` with WSL capped at **224 GB**, not 244 GB;
- Windows Balanced/ASPM Moderate baseline;
- ASPM Off on AC.

Guardrail:
- no-reclaim must preserve roughly 32 GB physical Windows headroom so GPU/driver
  allocations are not starved.

### U100 — Second-GPU / expert-parallel design
**Class:** I for software design; S for physical qualification
**Status:** PLANNED
**Priority:** P2

Work now:
- design two-device GLM expert placement/dispatch around the backend's existing P2P and
  resident-expert primitives;
- favor expert parallelism over fine-grained tensor parallelism on PCIe-only 4090s.

Physical qualification waits for the second 4090 topology to be decided.

### U110 — Production qualification
**Class:** S
**Depends on:** U10 + chosen portions of U20/U30/U40/U50/U60 + platform decision
**Status:** BLOCKED
**Priority:** final gate

Require:
- deterministic correctness;
- repeated long generation;
- stable 23+ GiB-class VRAM target if beneficial;
- no swap/OOM/resource leaks;
- representative agent turn;
- exact launch profile checked into automation.

### U120 — Restart the expensive Tensor64/SC64 agent analysis
**Class:** S
**Depends on:** U110
**Status:** BLOCKED

Run with:
- 12-hour wall budget;
- profiling active;
- preserved artifacts;
- no automatic one-hour kill.

## Parallel lanes

The following lanes may proceed at the same time without waiting for platform switching:

**GPU lane:** U00 -> U10 + U20 -> U30/U40 -> U60

**CPU lane:** U00 -> U50, then overlap work joins U20/U30

**Platform-preparation lane:** source/tests/docs for U70/U80/U90/U95 can proceed while
the current WSL inference runs, but full-model/restart/storage benchmarks are serialized.

**Multi-GPU design lane:** U100 software design is independent of the single-GPU runtime.

## Serialized WRX80 host sequence after the current run

This order avoids unnecessary 195 GB copies and answers the highest-value question first:

1. Preserve current WSL baseline and stop it cleanly.
2. **U70 native Windows baseline.**
3. U70 native Windows + mapped experts.
4. Decision gate:
   - native wins materially -> keep Windows and continue GPU/CPU work there;
   - native is neutral/loses warmed wall time -> continue Linux transport experiments.
5. U80 VirtioFS.
6. U90 ext4 VHDX on E: if still decision-relevant.
7. U95 no-reclaim / ASPM experiments after the storage transport is chosen.
8. U110 production qualification.
9. U120 expensive agent restart.

## Priority interpretation

For **steady-state agent latency**, current evidence says:

1. U20/U30/U40 attention + device-resident compute path;
2. U10 near-full useful VRAM residency;
3. U00/U50 measured CPU critical path and overlap;
4. U60 decode specialization;
5. platform/storage only where it changes compute/CPU behavior or cold/prewarm cost.

For **cold-start / model-load / prewarm latency**:

1. U70 native Windows;
2. U80 VirtioFS;
3. U90 ext4 VHDX;
4. U95 page-cache retention / ASPM.

Do not confuse the roughly 9.5x tiny native-vs-/mnt/e storage microprobe with the current
warmed inference bottleneck: the latest large turn spent only 1.569 s in expert-disk work.

### Native Windows qualification run Q1 — 2026-10-06

Configuration:
- Flash model on `E:\z-models\GLM-5.3-Flash-colibri-int4-g64`;
- `OMP_NUM_THREADS=16`;
- `COLI_CUDA_RESIDENT_EXPERT_GB=18`, reserve `3`;
- `GLM53_EXPERT_GB=175`, expert prewarm enabled;
- `COLI_CUDA_GLM53_CHAIN=2`, `ROUTER=2`, `INDEXER=2`;
- `GLM53_PROFILE_VRAM=1`, `GLM53_PROF_EVERY=1`, `PROF=1`;
- fixed 3,400-character deterministic prompt, 551 prompt tokens, 2 greedy decode tokens.

Observed:
- host expert prewarm: 10,954 slots / 155.1 GB in 73.5 s;
- resident matrices: 4.33 GiB VRAM; routers: 252 MiB; hot experts: 1,142 experts / 15.06 GiB VRAM;
- profile peak CUDA used 21,985.5 MiB, minimum free 2,578 MiB;
- native GPU sample remains bursty: mostly 0–20% utilization over a 10 s trace, peak 31%, roughly 63–115 W;
- S=1 CUDA sparse indexer exact parity: 8/8 reported passes, 0 mismatches/failures;
- router qualification reported exact selected-index parity; observed weight drift remained sub-micro;
- shared/resident expert device-row verification remained at float-noise scale;
- 84 decode resident expert sets were observed and **0% were fully resident** at this conservative tier;
- decode result: 2 tokens in 2.1 s = 0.934 tok/s;
- prefill `CHAIN=2` verification showed large drift after the first attention site. The chain path is decode-oriented, so `2f6440c4` now gates resident-chain mode to `n == 1`, leaving multi-token prefill on the proven path and reserving chain verification for S=1 decode.

Raw artifacts on WRX80:
- `E:\z-results\glm53-native-2026-10-06\q1-parity-short.log`
- `E:\z-results\glm53-native-2026-10-06\q1-gpu-sample.csv`

Next:
- rebuild native at `2f6440c4`;
- rerun the same workload with mode-2 router/indexer/chain;
- require clean S=1 chain parity before using chain mode 1;
- then run an authoritative decode arm with indexer/router mode 1 and compare wall time/utilization;
- separately sweep expert reserve only after preserving at least ~1.5–2 GiB transient headroom.

### Native Windows qualification Q2/Q2b — stack fix + decode-only chain

Q2, after `2f6440c4` restricted chain mode to S=1, made prefill cleanly use the proven path but crashed after forward 4. Windows Application Error event 1000 identified exception `0xc00000fd` in `coli_cuda.dll` at RVA `0x21a57`. Disassembly mapped this to MSVC `__chkstk`: `coli_cuda_expert_group_host_clamped` had a ~5.27 MiB stack frame because three local `ColiCudaTensor[64]` arrays each embedded the tensor's 512-entry ragged-KV metadata.

- `e01b090` moves those temporary descriptor arrays to heap-backed `std::vector` storage. The native frame shrank to ~2 KiB (`sub rsp,0x7f8`). Native and WSL RTX CUDA correctness suites pass; CPU/CUDA-linked GLM builds and native host relink pass.
- `f075f0f` adds startup-only resident-history coverage telemetry. The current 1,142-expert / 15.06 GiB tier covers 24.2% of historical selection mass, but is highly uneven: layer 5 has only 3.0% / 4 resident experts while layer 19 has 37.9% / 39. Offline equal-per-layer allocation with the same 1,142 experts retains ~23.79% total historical mass while raising worst-layer historical coverage to ~16.05%.
- Q2b (`q2b-stackfix-parity.log`) crossed the former crash boundary and completed. S=1 chain verification is at float-noise scale (reported absolute drift ~1e-8 to 1e-6); sparse indexer reported 8/8 exact passes with zero mismatch/failure; router selected indices matched; shared/resident MoE drift stayed at float-noise scale.
- Q2b decode: 2 tokens in 2.4 s = 0.850 tok/s under heavy qualification overhead. Peak CUDA used 22,809.5 MiB, minimum free 1,754 MiB. Resident complete-set coverage remained 0/84; chain resident-input continuity remained 12.2%.

Next experimental gate is B1 from the native Windows plan: disable chain verification, keep router/indexer mode 2, generate 8–16 deterministic tokens, capture utilization and parity, then promote router/indexer one at a time if clean.

## Wake prompt

> Resume the WRX80 GLM-5.3 optimization from
> `docs/experiments/wrx80-glm53-execution-ledger.md`. Check the live service/profile and
> repository state first, preserve any active inference, then take the highest-priority
> unblocked unit and make a small measured change with a small commit; keep this ledger
> updated after every meaningful checkpoint.

If inference itself is stalled or idle:

> Inference is stalled/idle: use the execution ledger, verify the stall is real, and
> advance the highest-priority independent GPU/CPU/source unit without restarting or
> discarding useful work. Prefer measurable latency reductions, frequent tiny commits,
> and update the ledger before moving to the next unit.


### Native Windows B1/B2 decode ladder — 2026-10-06

B1 (CHAIN=0, ROUTER=2, INDEXER=2) completed on the fixed 551-token prompt with 8 greedy decode tokens:
- sparse indexer reported 8/8 exact parity checks with zero mismatch/failure;
- router qualification reported exact selected-index parity in the visible checks;
- decode: **8 tokens in 8.2 s = 0.975 tok/s**;
- peak CUDA used 21,759.5 MiB, minimum free 2,804 MiB.

B2 (CHAIN=0, ROUTER=1, INDEXER=2) also completed cleanly:
- sparse indexer reported 8/8 exact parity checks with zero mismatch/failure;
- decode: **8 tokens in 8.1 s = 0.982 tok/s**;
- peak CUDA used 22,623.5 MiB, minimum free 1,940 MiB.
This is a small but directionally positive result for authoritative GPU routing.

The first marshaled B3 attempt did **not** reach model code: PowerShell Start-Process -ArgumentList flattened the prompt and glm53.exe rejected the bare word `this`. `c70ae1f` fixes prompt quoting; an argv smoke test confirmed a multiword prompt remains one argument. No B3 performance/correctness conclusion should be drawn from the failed 29 ms harness attempt.

Next serialized gate: B3 with CHAIN=0, ROUTER=1, INDEXER=1, same prompt and 8-token tail. Require authoritative indexer success with zero fallback before advancing.


### Native Windows B3 authoritative indexer + N30 profiling — 2026-10-06

B3 (`CHAIN=0`, `ROUTER=1`, `INDEXER=1`) completed successfully on the same fixed 551-token prompt and 8-token decode tail:
- authoritative sparse indexer reported **8/8 success, fallback=0**;
- decode: **8 tokens in 7.9 s = 1.018 tok/s**;
- peak CUDA used 22,623.5 MiB, minimum free 1,940 MiB;
- cumulative indexer time at forward 13 was 3.059 s, split into 1.966 s projection + 1.093 s selection.

This clears the N20 authoritative router/indexer short-run gate. Compared with B1/B2 (0.975 / 0.982 tok/s), authoritative index selection removes the CPU reference selector and improves the short decode result again.

`bf8d61a` adds opt-in `COLI_CUDA_SPARSE_PROFILE=1` backend timing without changing the public ABI. It reports cumulative:
- selected-index D2H + stream wait (`index_wait`);
- sparse staging-event reuse wait (`stage_wait`);
- compact latent host gather (`pack`);
- q/selection/compact-latent H2D enqueue (`h2d_enqueue`).

Validation for `bf8d61a`:
- `git diff --check` clean;
- native RTX CUDA numerical suite passed (`q8/q4/q2/f32/e8 correctness ok`);
- CPU/native GLM build current;
- native `coli_cuda.dll` rebuild + CUDA-linked `glm53.exe` relink passed;
- Windows loader fixture: 12/12 tests passed.

Next: run one short authoritative B3-equivalent arm with `COLI_CUDA_SPARSE_PROFILE=1`, use those measurements to choose the next attention optimization, then proceed to N40 residency policy work if the sparse boundary is not dominant.


### N40 offline residency-policy simulation — 2026-10-06

`93e68b2` adds `c/scripts/wrx80_glm53_residency_sim.py`, which reads the text `.coli_usage` history and compares equal-budget admission policies without touching runtime.

At the current 1,142-expert resident budget across 42 routed layers:
- current global-heat admission: **24.09%** total historical selection mass, worst layer **2.40%**, median layer **25.90%**, 3..39 resident experts/layer;
- per-layer round-robin by local heat rank: **23.72%** total mass, worst layer **15.90%**, median layer **23.88%**, 27..28 resident experts/layer.

So a fair per-layer allocation sacrifices only ~0.37 percentage points of aggregate historical hit mass while improving the weakest layer by ~6.6x. This is a strong candidate for increasing complete expert-set residency, which the current global policy has measured at 0% in short decode runs.

Next N40 runtime step should be opt-in first: implement a fair/interleaved admission mode, preserve the global policy as baseline, and A/B complete-set coverage + wall time before changing defaults.


### Experimental-matrix correction — native Windows attention path

Review of `mla_layer()` found that B1/B2/B3 used `CHAIN=0` without setting `COLI_CUDA_GLM53_ATTN=1`. Therefore those runs correctly qualify and measure the GPU router, GPU sparse indexer, resident expert tier, and other enabled CUDA expert work, but their sparse MLA attention remained on the CPU path. They must not be treated as the all-GPU-attention benchmark.

This does not invalidate the N20 result: router/indexer promotion from B1 -> B2 -> B3 was isolated as intended, and B3 still established 8/8 authoritative GPU indexer successes with zero fallback. It does change the next performance gate.

Current source also shows `CHAIN=1` is not yet production-effective: `run_layers()` only builds the device mHC pre/RMSNorm site-entry buffers when `chain_mode == 2`, so mode 1 never gets `chain_pre_ok`/`chain_ready` and falls back before the device-resident MLA chain. Q2b remains useful evidence that the S=1 mode-2 chain numerics are clean.

Corrected next sequence:
1. Native ordinary GPU-attention baseline: `CHAIN=0`, `COLI_CUDA_GLM53_ATTN=1`, `ROUTER=1`, `INDEXER=1`, fair residency OFF.
2. Promote the already-qualified S=1 site-entry path so `CHAIN=1` can actually become authoritative, keeping multi-token prefill on the proven path.
3. Re-qualify with mode 2, then compare real mode 1 against the ordinary GPU-attention arm.

`fce27ad` also adds opt-in `COLI_CUDA_RESIDENT_LAYER_FAIR=1`; default behavior remains the original global heat admission policy until an A/B run proves the fair policy improves complete-set residency and wall time.


### Native Windows G2 fair-residency A/B — 2026-10-06

G2 changed only `COLI_CUDA_RESIDENT_LAYER_FAIR=1` relative to frozen G1 (`CHAIN=0`, `ATTN=1`, `ROUTER=1`, `INDEXER=1`, same 551-token prompt and 8-token tail).

Observed:
- same resident budget: 1,142 experts / 15.06 GiB VRAM;
- historical selection-mass coverage: 23.7% total vs ~24.1% for global heat order;
- worst-layer historical coverage improved from ~2.4% to **15.9%**;
- resident experts/layer became **27..28** instead of roughly 3..40;
- authoritative indexer reported 8/8 successes with fallback=0;
- cumulative forward-13 attention: 107.968 s; FFN: 101.742 s;
- peak CUDA used 22,653.5 MiB, minimum free 1,910 MiB;
- decode: **8 tokens in 6.0 s = 1.332 tok/s**.

Frozen G1 was 1.299 tok/s, so this first fair-policy sample is about **2.5% faster** while dramatically improving worst-layer residency coverage. Treat the speed delta as provisional until the queued fair-repeat/global-repeat pair completes.


### Fair-residency repeat checkpoint — 2026-10-06

Queued fair-policy repeat `g2r-fair-attn1-router1-indexer1-chain0-g8` completed cleanly:
- decode: **8 tokens in 6.1 s = 1.307 tok/s**;
- authoritative indexer remained 8/8 successful with fallback=0;
- cumulative forward-13 attention 110.605 s; FFN 101.122 s;
- peak CUDA used 22,641.5 MiB, minimum free 1,922 MiB.

This repeat is only ~0.6% above frozen G1 at 1.299 tok/s, so the first G2 result at 1.332 tok/s was partly favorable run variance. The structural residency improvement remains real: fair admission keeps 27..28 experts/layer and ~15.9% worst-layer historical coverage versus the global policy's ~2.4% floor. Keep fair admission as promising but not yet a decisive wall-clock winner until the queued global control repeat and sparse-profile arms complete.


### Full GLM-5.3 comparative benchmark lane — 2026-10-06

A native-Windows comparison lane is now defined for the full `GLM-5.3-colibri-int4-g64` checkpoint. The full model is a supported `glm_moe_dsa` family in this engine, not a Flash-only compatibility hack. Colibri analysis reports ~419.3 GB model bytes, ~407.7 GB routed-expert bytes, 75 expert layers, 256 routed experts/layer, hidden size 6144, and 78 transformer layers.

The comparison matrix is `c/scripts/wrx80_glm53_full_comparison.json`. It holds the hardware/software budget constant (`CHAIN=0`, `ATTN=1`, `ROUTER=1`, `INDEXER=1`, 18 GB resident-expert target, 3 GB reserve, 175 GB host expert budget) and runs:
- fresh Flash fair-residency 8-token control;
- full-model 1-token compatibility smoke with global residency;
- full-model 8-token global-residency short benchmark;
- full-model 8-token fair-residency short benchmark;
- disabled ~32k-character medium-context full-model benchmark, enabled only after the short lane passes.

The first run is gated on `E:\\z-results\\glm53-native-2026-10-06\\chain2-requal-approved.txt` containing `PASS`, so full-model work cannot overtake CHAIN=2 correctness requalification.


### Utilization phase cursor

The canonical next-work cursor is now `docs/experiments/wrx80-glm53-utilization-execution-cursor.md`. ACTIVE unit: U00, repair/complete isolated validation of the conservative CHAIN=1 step-1 promotion before CHAIN=2 requalification.


### U00 validation progress — 2026-10-06 14:27 local

The staged conservative CHAIN=1 source passed the isolated native CPU build. The earlier validator failure occurred in the wrapper/build environment after `make clean`, before any source compile failure was observed. Next gate is explicit native CUDA DLL + CUDA-linked GLM validation in the isolated worktree.


### U00.2 checkpoint — CUDA DLL build PASS

The staged CHAIN=1 step-1 source now passes both the isolated CPU build and isolated CUDA DLL build. Next gate is CUDA-linked `glm53.exe`, followed by RTX backend numerical tests and loader ABI fixture before source commit.


### U00.3 checkpoint — full isolated validation PASS

The conservative CHAIN=1 step-1 source passed the complete isolated validation ladder: native CPU build, CUDA DLL build, CUDA-linked `glm53.exe`, RTX q8/q4/q2/f32/e8 numerical test, and the 12-case Windows loader ABI fixture. The earlier CUDA-test wrapper error was PowerShell stderr handling, not a test failure. Next unit is source-only commit/push of the validated `c/glm53.c` change after confirming the main checkout matches the validated worktree patch.


### U00 complete - CHAIN=1 step-1 source committed

The conservative CHAIN=1 site-entry/device-FFN/device-post promotion is now committed as `fb61f73` after the full isolated validation ladder passed. Main and isolated source copies matched exactly after newline normalization. The next serialized gate is U01.1 CHAIN=2 short requalification; do not release the full-model comparison sentinel before reviewing that run.


### U01.1 CHAIN=2 short requalification - REVIEWED PASS

Exact-source qualification used detached worktree `E:\\z-src\\colibri-u01-chain2` at source commit `fb61f73`, so the main checkout's unfinished `c/backend_cuda.cu` profiling change was not present in the binary.

Run: `u01q-chain2-fb61f73-global-g8`, CHAIN=2 / ATTN=1 / ROUTER=1 / INDEXER=1 / global residency, deterministic 551-token prompt and 8-token decode tail.

Reviewed evidence:
- 16 pre+norm verification reports: max norm abs 7.15256e-7, post abs 1.78814e-7, comb abs 2.98023e-7;
- 16 whole-chain residual reports: max abs 9.53674e-7;
- 16 shared-expert dev-row reports: max abs 1.78814e-7;
- 16 resident-MoE dev-row reports: max abs 2.98023e-8;
- 16 shared+resident reports: max abs 2.98023e-8;
- authoritative indexer: 8/8 reports, final pass=8, fallback=0;
- no CUDA error, engine_error, mismatch, or nonzero fallback markers;
- decode: 8 tokens in 10.3 s = 0.776 tok/s; performance is not the purpose of CHAIN=2 qualification;
- peak CUDA used 22,809.5 MiB, minimum free 1,754 MiB.

The temporary one-run marshal marked `ok=false` only because it required the nonexistent strings `GLM53 shared expert verify` and `GLM53 resident expert verify`. Source inspection confirmed the real labels are `GLM53 shared expert dev-row verify` and `GLM53 resident MoE dev-row verify`, and both appeared with the clean values above. Treat this as a temporary manifest-spec error, not a model failure.

Evidence summary: `E:\\z-results\\glm53-native-2026-10-06\\u01-chain2-review.json`.

U01 is cleared. ACTIVE next unit: U10.1, prepare the first real CHAIN=1 short A/B against ordinary GPU attention. The full GLM-5.3 comparison lane may run first because it is already waiting on the U01 approval sentinel; do not rebuild/replace binaries while that lane is active.
