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
**Status:** ACTIVE / DECODE PRIMITIVE IMPLEMENTED, NOT YET WIRED
**Priority:** P1

Goal: remove the remaining CPU-side sparse-index construction and attention-state traffic.

Completed checkpoints:
- `8aac5c2d` adds an exact decode-first CUDA sparse-index primitive matching `coli_sparse_index_select_range` for leading padding, complete pools, deterministic top-k selection, and incomplete-tail emission. RTX 4090 numerical parity, CPU GLM build, CUDA-linked GLM build, and Linux header/loader parity pass. The public CUDA ABI is wired as mandatory (`58 mandatory + 9 optional`); the Windows-invoked loader fixture still self-skips its Windows-only classes in this environment, so full native DLL fixture execution remains a later qualification item. The primitive is not yet called by GLM inference.

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
**Status:** READY FOR FULL-MODEL A/B
**Priority:** P0 platform experiment

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
