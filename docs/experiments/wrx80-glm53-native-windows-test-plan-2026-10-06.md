# WRX80 GLM-5.3 Native Windows Iteration Plan — 2026-10-06

## Goal

Qualify and optimize the native Windows GLM-5.3 Flash path on WRX80 using the CUDA-resident branch as the production candidate. Maximize warmed S=1 decode wall-clock throughput and sustained useful GPU utilization while preserving deterministic CPU fallback and measured numerical correctness.

Canonical branch: `wrx80/glm53-cuda-resident`

Starting cursor: `f3e8994d`

Model: `E:\z-models\GLM-5.3-Flash-colibri-int4-g64`

Do not launch a competing WSL model workload while native qualification is active.

## Operating discipline

1. Inspect branch status and concurrent dirt before every source change.
2. Keep source changes narrowly scoped and commit them separately from documentation.
3. After CUDA-source changes run the RTX CUDA numerical suite, CPU GLM build, CUDA-linked/native Windows GLM build as relevant, and `git diff --check`.
4. Push every useful checkpoint.
5. Preserve experimental logs under `E:\z-results\glm53-native-2026-10-06\`.
6. Use short deterministic probes first. Increase prompt/context or generated-token count only after the preceding gate is clean.
7. Qualification-mode throughput is not production throughput when verification work is enabled.

## Run 0 — short native parity probe

Configuration:

- OMP_NUM_THREADS=16
- COLI_CUDA=1
- COLI_CUDA_RESIDENT_EXPERT_GB=18
- COLI_CUDA_RESIDENT_RESERVE_GB=3
- GLM53_PREWARM_EXPERTS=1
- GLM53_EXPERT_GB=175
- COLI_CUDA_GLM53_CHAIN=2
- COLI_CUDA_GLM53_ROUTER=2
- COLI_CUDA_GLM53_INDEXER=2
- GLM53_PROFILE_VRAM=1
- GLM53_PROF_EVERY=1
- PROF=1
- deterministic 3400-character systems prompt
- greedy 2-token generation

Observed:

- native CUDA initialization succeeded;
- resident matrix prewarm: 552 matrices, 4.33 GiB VRAM;
- router prewarm: 42 layers, 252 MiB VRAM;
- resident expert tier: 1142 experts, 15.06 GiB VRAM;
- initial free VRAM after prewarm: about 2.87 GiB;
- measured peak used: 21985.5 MiB;
- measured minimum free: 2578.0 MiB;
- sparse indexer mode-2 reported exact selected-index parity with zero reported mismatch/failure;
- router selected-index parity matched; reported weight drift was around 1e-7 to 1e-6;
- shared and resident MoE verification drift was tiny, typically absolute 1e-8 to 1e-7;
- decode: 2 tokens in 2.1 s = 0.934 tok/s despite qualification overhead;
- all-resident expert-set coverage: 0/84 decode sets under the 18 GB target;
- chain-residency continuity: 77/630 pre-sites, 12.2%;
- chain qualification exposed large mHC pre/norm and downstream drift during prefill, beginning at layer 0 site 1 while layer 0 site 0 was near numerical noise.

Log: `E:\z-results\glm53-native-2026-10-06\q1-parity-short.log`

Interpretation:

- GPU sparse indexer and router are ready for longer mode-2 qualification and likely mode-1 A/B after another clean decode run.
- Resident/shared MoE device math is numerically healthy in observed S=1 checks.
- Chain mode 1 is not qualified. Mode 2 kept the CPU path authoritative, so Run 0 remained correctness-safe.
- The chain issue should be debugged independently from router/indexer promotion.
- The 18 GB resident-expert target leaves useful safety margin but gave zero complete resident expert sets in this tiny sample, making residency policy a major optimization opportunity.

## Phase A — isolate chain-verification drift

Priority: P0 correctness.

1. Reproduce the first failing boundary at layer 0 site 0/site 1.
2. Capture CPU-vs-GPU pre/norm/post/comb maxima separately.
3. Verify `cuda_hc_site_prepare()` maps attention and FFN `fn/base/scale/norm` tensors correctly.
4. Distinguish parameter-upload mismatch from input/state-propagation mismatch.
5. Confirm real-geometry S>1 behavior independently of the synthetic small-shape CUDA test.
6. If the issue is qualification-only state propagation, fix the bridge rather than kernel math.
7. Add a regression test before enabling chain mode 1.

Exit: real-model mode-2 chain drift is within established tolerance for prefill and S=1 decode, or chain qualification is explicitly restricted to a valid scope.

## Phase B — router/indexer promotion ladder

Priority: P0 performance/correctness.

Run B1:
- CHAIN=0;
- ROUTER=2;
- INDEXER=2;
- same deterministic ~551-token prompt;
- 8–16 decode tokens;
- collect parity counts, wall time, index projection/select timing, router timing, GPU utilization, and VRAM.

If B1 is clean:

Run B2:
- ROUTER=1 authoritative;
- INDEXER=2 qualification;
- identical workload.

Run B3:
- ROUTER=1;
- INDEXER=1 authoritative;
- identical workload.

Compare warmed decode wall time. Any backend failure must fall back deterministically.

## Phase C — selected-index / compact-latent profiling split

Priority: P0 diagnosis.

Instrument one narrow counter at a time:

1. GPU indexer compute/launch.
2. selected-index D2H plus stream synchronization.
3. compact latent host gather/pack.
4. pinned staging reuse wait.
5. compact latent H2D.

Preserve aggregate `index_select` timing.

Goal: determine whether the next attention win is dominated by pooled-history scoring, index-result synchronization, host gather/packing, the ~4 MiB/layer latent upload, or downstream launch gaps.

## Phase D — resident expert coverage

Priority: P0/P1.

Current tiny-run result: 0% all-resident sets at the 18 GB target.

1. Record per-token/layer selected expert sets and residency misses.
2. Measure how many additional experts or GiB would convert partial sets to complete sets.
3. Compare current ordering with frequency and co-occurrence-aware admission.
4. Sweep residency conservatively. The current request produced 15.06 GiB actual expert residency and ~2.58 GiB minimum free VRAM.
5. Prefer changes that increase complete-set coverage over merely increasing raw resident expert count.

Exit: materially higher complete-set coverage without allocation failures or pathological streamed-expert churn.

## Phase E — utilization capture

For representative warmed runs sample at 100–500 ms cadence:

- GPU utilization;
- power;
- VRAM used/free;
- CPU utilization;
- disk throughput;
- process RSS/private bytes.

Correlate with per-forward profile lines. Look for long GPU-idle CPU sections, H2D/synchronization bubbles, streamed expert stalls, and launch-bound low-power phases.

## Phase F — longer-context native qualification

Only after short B-series correctness passes:

1. fixed medium-context run;
2. fixed long-context run;
3. preserve full profiles and environment;
4. compare to the final WSL baseline on wall time, prefill, warmed decode tok/s, attention/indexer/router/FFN buckets, VRAM peak/min-free, resident expert coverage, GPU duty/power, host RSS, and disk activity.

Do not spend a multi-hour run on a configuration whose short-run parity gate is unresolved.

## Immediate next actions

1. Commit this plan.
2. Preserve Run 0 observations in the execution ledger.
3. Debug chain mode-2 prefill drift at layer 0/site 1.
4. Prepare B1 with CHAIN disabled and ROUTER/INDEXER mode 2 so their qualification can proceed independently.
5. Add the selected-index/compact-latent timing split before the first medium/long-context run.
6. Use the next clean short run to decide whether router and indexer advance to authoritative mode 1.
