# WRX80 GLM-5.3 utilization execution cursor

Date: 2026-10-06
Branch: `wrx80/glm53-cuda-resident`
Platform: native Windows on WRX80 / RTX 4090
Canonical repo: `E:\z-src\colibri-native`

This is the canonical restart cursor for the next phase of GLM-5.3 optimization. It supersedes the stale "Current cursor" section in `wrx80-glm53-native-windows-work-plan-2026-10-06.md`, while that document remains the historical plan and experiment record.

## Goal

Increase steady-state decode throughput by raising useful GPU duty cycle and useful CPU overlap. Prefer removal of host/device serialization and S=1 CPU fallback over micro-kernel tuning.

The governing metric is warmed wall-clock decode throughput on the fixed short native benchmark, followed by medium and long context. GPU utilization, CPU utilization, power and phase timings are diagnostic metrics, not goals by themselves.

## Mandatory operating discipline

1. Before every source unit, inspect HEAD, origin, tracked dirt, active `glm53` processes, GPU use and any active marshal/validator.
2. Preserve useful inference. Never rebuild or replace the native DLL while a benchmark is active.
3. Treat tracked dirty `c/glm53.c` and `c/backend_cuda.cu` as owned in-progress work. Inspect before modifying. Do not blindly restore either file.
4. Keep changes extremely small. One measurable hypothesis per source commit where practical.
5. Validation order: `git diff --check` -> relevant source/unit test -> CPU GLM build -> CUDA DLL/native GLM build -> RTX numerical suite/loader fixture when affected -> commit -> docs checkpoint -> push.
6. It is acceptable to commit documented unfinished work if doing so leaves a recoverable checkpoint.
7. Do not add/delete the existing native build artifacts merely to make status clean.
8. Do not start medium/long context until the short-run winner is stable and the active correctness gate has passed.
9. Full GLM-5.3 comparison is a parallel benchmark lane, not permission to bypass Flash correctness gates.

## Frozen short baselines

All use the deterministic ~551-token prompt and 8-token tail unless noted.

- G1 ordinary native GPU attention, global residency, CHAIN=0 / ATTN=1 / ROUTER=1 / INDEXER=1: 1.299 tok/s.
- G2 fair residency first sample: 1.332 tok/s.
- G2R fair repeat: 1.307 tok/s.
- G1R global repeat: 1.269 tok/s.
- G1P global later sample: 1.352 tok/s. This run did not exercise the uncommitted sparse-boundary instrumentation, because the DLL had not been rebuilt with that source.

Fair admission is therefore structurally superior and appears directionally faster, but wall-clock variance remains material. Its repeatable structural result is the stronger evidence: ~27..28 resident experts/layer and ~16% weakest-layer historical coverage instead of ~3..38 experts/layer and ~2.4% weakest-layer coverage.

## Active source ownership

### `c/glm53.c`

Contains the conservative CHAIN=1 promotion in progress.

Intent:
- S=1 mode 1 may use qualified device mHC pre/RMSNorm;
- resident shared/routed FFN composition may remain on device;
- device mHC post becomes authoritative;
- after successful post, the H*D residual bank is downloaded back to host so all existing host projections and fallback assumptions remain coherent.

This is step 1 only. Do not combine it with removal of residual readback.

### `c/backend_cuda.cu`

Contains unfinished sparse-project profiling extensions.

Intent:
- measure S=1 sparse-project input H2D enqueue time;
- measure projected-output D2H + stream synchronization;
- correlate with existing sparse index wait;
- use the data to decide whether the next attention optimization is selection residency, latent residency, output residency, or a combination.

Do not claim N30 measurements from runs built before this source is rebuilt.

## Serialized optimization ladder

### U00 - Repair validation harness and qualify staged CHAIN=1 step 1

Priority: P0, ACTIVE.

Current issue:
The isolated CHAIN=1 validator reached `make clean` but produced no CPU/CUDA build/test logs or result file. Treat this as a validator/harness failure until proven otherwise, not a model failure.

Next smallest unit:
- inspect why the validator exited after clean;
- run the validation commands manually or fix the validator in the isolated worktree;
- preserve logs under `E:\z-results\glm53-native-2026-10-06\chain1-validation`;
- if all gates pass, commit only the intended `c/glm53.c` CHAIN=1 step-1 change;
- update ledger and push.

Done when:
- CPU build passes;
- CUDA DLL + CUDA-linked native GLM build pass;
- RTX backend numerical suite passes;
- loader fixture passes;
- source commit and docs commit are pushed.

Blocks: U10, full-model comparison approval sentinel.

### U01 - CHAIN=2 short requalification

Priority: P0, serialized after U00.

Run the predeclared short CHAIN=2 qualification using ordinary GPU attention and authoritative router/indexer. Preserve mHC pre/norm/post/combined errors and whole residual-bank drift.

Done when:
- all expected mode-2 verify sites are numerically within previously qualified S=1 tolerances;
- indexer/router remain clean;
- no CUDA/engine error;
- result is checkpointed;
- `chain2-requal-approved.txt` contains PASS only after evidence is reviewed.

Blocks: U10 production CHAIN=1 A/B and full-model comparison marshal.

### U10 - Real CHAIN=1 short A/B

Priority: P0.

Compare conservative production CHAIN=1 against the frozen ordinary GPU-attention baseline. Start with fair residency because it is the leading admission policy, but retain a global arm if attribution is ambiguous.

Measure:
- tok/s and seconds/token;
- cumulative attention/FFN/router/indexer;
- VRAM headroom;
- GPU util/power timeline;
- CPU process time;
- count of host residual downloads.

Done when:
- CHAIN=1 produces correct output with no fallback caused by chain plumbing;
- repeated short run establishes whether the conservative chain helps, hurts or is neutral.

### U11 - Remove cross-site/layer residual readback

Priority: P0 if U10 is correct and non-regressive.

Hypothesis:
The conservative host mirror is now one of the largest avoidable S=1 synchronization points. Keep residual/post/comb/normed state device-resident across attention -> FFN -> next layer. Download only at an actual CPU fallback or externally required boundary.

Implement incrementally:
1. site-0 to site-1 continuity within one layer;
2. site-1 to next-layer site-0 continuity;
3. explicit host-invalid/device-valid state tracking;
4. forced fallback test proving host reconstruction/synchronization is correct.

Done when:
- no routine H*D residual D2H occurs during normal S=1 decode;
- qualification and fallback tests pass;
- wall-clock and utilization improve or the unit is rejected with measurements.

### U20 - S=1 routed-expert GPU crossover

Priority: P0/P1 after U10, independent of U11 source if isolated carefully.

Observed issue:
`small GLM expert groups stay on CPU (8 rows, CUDA minimum 32)`.

This is a decode-shaped mismatch: top-8 S=1 naturally creates tiny expert groups.

Benchmark threshold ladder with identical short workload:
- minimum group 32 baseline;
- 8;
- 4;
- 1.

Measure launch cost, kernel time, H2D/D2H, CPU MoE time and total tok/s.

If generic grouped CUDA loses below 32, implement a specialized S=1/top-8 path rather than forcing the generic batch kernel.

Candidate specialized path:
- one token / top-k experts;
- resident expert pointers directly where available;
- compact streamed expert staging for misses;
- fused gate/up + activation + down + weighted accumulation where practical;
- avoid host accumulation.

Done when:
- the crossover is measured, not guessed;
- the winning threshold/path is committed;
- CPU MoE fallback during the short decode is materially reduced.

### U30 - Sparse-attention host-boundary removal

Priority: P1.

First finish/rebuild the existing sparse-project instrumentation. Collect real counters for:
- selected-index D2H + wait;
- staging event wait;
- compact latent host pack;
- q/selection/latent H2D enqueue;
- sparse attention + o_proj completion;
- projected-output D2H + synchronization.

Then remove the dominant boundary in smallest-first order.

Likely sequence:
1. preserve selected indices on device while optionally retaining the tiny host copy needed by current packing;
2. device-side gather or paged/selective latent residency, preserving compact-transfer semantics;
3. keep sparse attention + o_proj output on device and feed it directly into CHAIN=1;
4. remove final synchronous projected-output D2H on the normal device path.

Guardrail:
Never regress to full f32 latent upload/residency. At ~46.7k context it was already multi-GiB across layers and is not viable at long context.

Done when:
- the former hard stream sync is removed from normal S=1 decode or shown not to matter;
- transfer volume and wall-time effects are measured;
- long-context memory model remains safe.

### U40 - Complete-set-aware expert residency

Priority: P1, can advance independently after short policy baselines are preserved.

Current fair admission is a good floor but optimizes per-layer fairness, not the probability that all top-k experts needed by a decode step are resident.

Build an offline simulator first:
- reconstruct top-k expert sets from available routing/usage traces where possible;
- score candidate admissions by marginal fully-resident-set probability per MiB;
- compare global heat, fair round-robin, per-layer quota, co-occurrence-aware greedy and hybrid policies.

Runtime implementation stays opt-in until A/B proves it.

Done when:
- policy improves complete-set residency at equal VRAM budget;
- short wall-clock A/B is non-regressive;
- VRAM reserve remains >= configured guardrail.

### U50 - CPU/GPU overlap

Priority: P1 after major correctness-sensitive synchronization points are understood.

Use 100-250 ms GPU telemetry plus CPU thread/process sampling and phase markers.

Look for CPU work that can happen while GPU is busy:
- next-layer router preparation;
- expert metadata lookup;
- prefetch of likely streamed experts;
- usage/stat accounting;
- compact latent preparation if it remains host-side;
- asynchronous disk/page-cache work.

Avoid speculative parallelism that races data dependencies. Prefer explicit double-buffering and events.

Done when:
- measured idle gaps shrink;
- CPU work overlaps GPU rather than extending the critical path;
- no regression in determinism/correctness.

### U60 - Launch-granularity and graph/fusion pass

Priority: P2, blocked on U11/U20/U30 measurements.

Only optimize small-kernel launch overhead after host/device serialization is reduced.

Candidates:
- fuse mHC micro-kernels where numerically safe;
- fuse router/select residual micro-ops;
- CUDA graph capture for stable S=1 decode shapes;
- persistent decode kernel only if profiling shows launch latency is now first-order.

Done when:
- Nsight-equivalent or internal timing shows launch gaps are a material remaining fraction;
- each fusion is benchmarked independently.

### U70 - Startup/prewarm utilization

Priority: P2 for warmed throughput, P1 for interactive usability.

Flash currently prewarms roughly 155 GB of host experts in ~75 s.

Explore:
- parallel prewarm with bounded IO/memory pressure;
- demand-driven warmup;
- overlap host prewarm with GPU matrix/router/expert admission;
- reuse/persist warm-page knowledge where safe.

Do not trade warmed decode performance for startup cosmetics.

## Context graduation

### U80 - Medium context

Enable only after the short winner is stable.

Use the existing ~32k-character / ~5k-token benchmark definition first. Preserve all utilization and phase telemetry. Compare winner against a recent control, not an old run.

### U90 - Long context

Enable only after U80 passes. Start with the existing ~128k-character / ~20k-token gate before moving toward the much larger target contexts.

Watch:
- latent/cache growth;
- VRAM reserve;
- index-selection scaling;
- CPU gather scaling;
- expert-cache churn;
- sustained GPU utilization.

## Full GLM-5.3 comparison lane

The full checkpoint is approximately 419.3 GB model data with ~407.7 GB routed-expert weights, 78 transformer layers, 75 expert layers, hidden size 6144 and 256 routed experts/layer.

Matrix:
`c/scripts/wrx80_glm53_full_comparison.json`

Sequence:
1. fresh Flash fair-residency short control;
2. full model 1-token compatibility smoke;
3. full model 8-token global residency;
4. full model 8-token fair residency;
5. medium-context full model only after short compatibility/performance passes.

This lane is gated behind U01 by `chain2-requal-approved.txt`.

Use the full model to answer:
- how much throughput changes from Flash to full at the same hardware budget;
- whether residency policy behaves differently with 256 experts/layer and 75 sparse layers;
- whether CPU fallback, storage traffic or router/expert work becomes the dominant bottleneck;
- whether optimizations developed on Flash transfer or need full-model-specific thresholds.

## Current execution cursor

ACTIVE: U00.

Exact next actions:
1. Verify no `glm53` benchmark is active and no useful marshal owns the GPU.
2. Inspect the isolated validator exit after `make clean`.
3. Manually execute the first failed/missing validation command in `E:\z-src\colibri-chain1-validate\c`.
4. Repair the validator only if the command itself is valid and the wrapper is at fault.
5. Complete the validation ladder.
6. Commit/push the CHAIN=1 step-1 source change only after all gates pass.
7. Update this cursor and execution ledger.
8. Advance to U01 CHAIN=2 requalification.

Independent work allowed while U00/U01 inference is active:
- offline U40 residency simulation improvements;
- analysis/documentation of U20 crossover experiment;
- benchmark-manifest preparation;
- no rebuild or DLL replacement while inference is active.

## Wake prompt

Resume WRX80 native-Windows GLM-5.3 optimization from `docs/experiments/wrx80-glm53-utilization-execution-cursor.md`. Inspect HEAD/origin/status, active inference/marshal/validator state and GPU use first. Preserve useful runs and treat dirty `c/glm53.c` and `c/backend_cuda.cu` as owned work. Advance the ACTIVE unit with the smallest validated change, checkpoint source and docs separately, push frequently, then update the cursor before moving on.


### U00 progress checkpoint — 2026-10-06 14:27 local

- The original validator exited after `make clean`; no CPU/CUDA build logs were produced. Treat this as a wrapper/build-environment failure, not a model failure.
- Manual isolated-worktree CPU build using absolute w64devkit `make.exe` passed for the staged conservative CHAIN=1 source.
- Evidence: `E:\\z-results\\glm53-native-2026-10-06\\chain1-validation\\cpu-build-manual.log` ends with `CPU_BUILD_PASS`.
- ACTIVE next action: run CUDA DLL build in the isolated worktree with explicit CUDA/MSVC environment, then CUDA-linked GLM build, RTX backend numerical suite, and loader fixture. Do not commit `c/glm53.c` until all remaining gates pass.


### U00.2 CUDA DLL validation — 2026-10-06

- Isolated worktree: `E:\\z-src\\colibri-chain1-validate`.
- Explicit native toolchain: w64devkit 2.10.0, CUDA 12.9, VS 2022 x64 `cl.exe`, `CUDA_ARCH=sm_89`.
- `make cuda-dll CUDA_ARCH=sm_89` PASSED for the staged conservative CHAIN=1 source.
- Evidence: `E:\\z-results\\glm53-native-2026-10-06\\chain1-validation\\cuda-dll-build-manual.log` ends with `CUDA_DLL_BUILD_PASS`.
- Only existing nvcc warnings were observed; no new compiler/linker error.
- ACTIVE next action: U00.3, build CUDA-linked native `glm53.exe`, then run the RTX backend numerical suite and loader fixture. Do not commit `c/glm53.c` until those gates pass.


### U00.3 remaining validation gates — PASS

- CUDA-linked native `glm53.exe` build PASSED in the isolated worktree. Evidence: `E:\\z-results\\glm53-native-2026-10-06\\chain1-validation\\cuda-glm-build-manual.log`.
- RTX backend numerical binary PASSED directly with exit code 0: `q8/q4/q2/f32/e8 correctness ok on 1 device(s)`. Evidence: `backend-test-direct.log`.
- The first `make cuda-test` wrapper attempt was not a numerical failure: PowerShell `$ErrorActionPreference=Stop` promoted the test's normal `[CUDA] device 0...` stderr banner into a `NativeCommandError`. Direct execution proved the binary itself passes.
- Loader stub fixture initially skipped because the direct Python command lacked w64devkit `gcc`/`objdump` on PATH. Rerun with explicit w64devkit PATH executed the intended fixture: **12/12 tests PASS** in 4.425 s. Evidence: `loader-test-direct.log`.
- U00 validation is now complete: diff check, CPU build, CUDA DLL build, CUDA-linked GLM build, RTX numerical suite, and loader ABI fixture all pass.
- ACTIVE next action: U00.4, verify the main-checkout `c/glm53.c` diff is byte/semantic-equivalent to the validated isolated-worktree patch, commit only that source file, push, then checkpoint docs and advance the cursor to U01 CHAIN=2 short requalification.
