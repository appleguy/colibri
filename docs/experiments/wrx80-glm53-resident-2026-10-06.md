# WRX80 GLM-5.3 Flash resident-GPU experiment log

Living checkpoint log for the dedicated WRX80 / RTX 4090 host. Keep entries small and chronological so interrupted agent work can resume without reconstructing state.

## 2026-10-06 00:xx PDT — baseline and implementation state

- WSL2 ceiling: 244 GiB; production launcher targets WSL MemTotal - 8 GiB; supervisor emergency floor 4 GiB; swap 16 GiB and treated as a pressure signal, not working memory.
- GPU: RTX 4090, 24,564 MiB reported by nvidia-smi.
- Colibri branch: `wrx80/glm53-cuda-resident`.
- Persistent GLM-5.3 hot-expert placement and dispatch are implemented in commits `b0c49c3` and `93fe586`.
- Resident-specific grouped-int4 no-sync path committed as `96e54c3`.
- Follow-up commits add grouped-resident regression coverage, live VRAM reserve guarding, avoided-H2D accounting, resident-dispatch timing, and scratch reuse.
- A verify-first CUDA sparse-MLA path and coverage counters are also present at current HEAD.
- Conservative live bring-up at `COLI_CUDA_RESIDENT_EXPERT_GB=1`, reserve 3 GB placed 70 experts / about 0.92 GiB VRAM. Observed 14 resident grouped calls, 108 routed rows, 21 expert hits, zero fallbacks.
- Existing CUDA backend test passed on the RTX 4090, including grouped q4 correctness. The resident clamped path is bitwise checked against the generic clamped path on preuploaded fmt=4 grouped-int4 tensors.
- Pre-existing untracked `c/glm53-vk` is unrelated and must remain untouched.

## Next measurement ladder

1. Rebuild current HEAD and repeat a fixed deterministic 1 GiB resident baseline.
2. Increase persistent expert budget to 12 GiB, keeping a 3 GiB reserve; measure placement, actual VRAM, resident hit rate, avoided H2D, resident time, fallbacks, and output correctness.
3. If clean, step 16 -> 18 -> 20 GiB. Stop on CUDA allocation/refusal, output mismatch, swap growth, or live free VRAM below reserve.
4. Compare fixed prefill-heavy and decode-heavy requests across CPU-only, legacy hybrid, and resident hybrid after the residency sweep.
5. Profile sparse CUDA MLA independently; preserve verify-first behavior until correctness and coverage are established.
6. Sweep `GLM53_PREFILL_CHUNK=128/256/512` on the fixed prompt once rolling phase counters are available. The dedicated host has ample RAM; larger chunks may amortize expert/cache work substantially while remaining well inside activation-memory limits.
7. Only after the best resident/MLA/prefill-chunk profile is known, restart the expensive SC64 agent audit.

## 2026-10-06 01:xx PDT — 12 GB / MLA mode 1 live utilization

- Restarted with `COLI_CUDA_RESIDENT_EXPERT_GB=12`, reserve 3 GB, and `COLI_CUDA_GLM53_ATTN=1` so CUDA attention results are used directly instead of double-computing the CPU reference.
- The richer persisted route history now contains 2,308,320 expert selections.
- Before the fixed performance request: 12,045 MiB VRAM used / 12,098 MiB free.
- A 30-second one-second-cadence sample during the request observed only 2/30 samples with nonzero SM utilization (27% and 16%); the remaining 28 samples were 0%. Treat this as a duty-cycle indicator, not precise kernel utilization, but it demonstrates long host-side intervals between CUDA bursts.
- VRAM increased from about 13.6 GiB to 14.3 GiB during that 30-second interval as lazy CUDA state populated. No swap use was observed.
- Implication: more resident experts can still reduce weight-transfer stalls, but sustained throughput is currently limited by host-side phases / pipeline discontinuity. Device-resident pipeline work should follow the residency sweep.
- Rolling profiling from the first completed forward of the interrupted 12 GB run reported `attn=9.522s`, `ffn=194.730s`, `disk=177.677s`, `head=0.374s`. This is the strongest current bottleneck evidence: sparse CUDA MLA has made attention comparatively small for that chunk, while FFN time is overwhelmingly expert-cache/disk dominated. Prioritize the aggressive 21 GB resident tier before deeper attention work.


## 2026-10-06 01:xx PDT — 12 GB resident + sparse MLA verify qualification

- Runtime configuration had already advanced to COLI_CUDA_RESIDENT_EXPERT_GB=12, reserve 3 GB, COLI_CUDA_GLM53_ATTN=2 (verify-first).
- Startup placed 847 hot experts using 11.17 GiB persistent VRAM and reported 11.28 GiB free immediately after placement.
- During the fixed qualification request, total VRAM rose through roughly 13-15 GiB as lazy dense/attention state populated, still leaving about 9 GiB free and using no swap.
- Sparse MLA verify samples for tokens 0-15 showed max absolute error at most 7.15256e-7. Relative error can look larger near zero (observed up to 0.0426316); absolute error remains tiny.
- Important benchmark rule: COLI_CUDA_GLM53_ATTN=2 performs CUDA plus the complete CPU reference and therefore is a correctness mode, not a performance mode. Performance measurements must use mode 1 after qualification.


## 2026-10-06 01:xx PDT — benchmark hygiene and sparse-MLA projection fusion

- The first 12 GB / mode-1 fixed request (3,400 prompt chars, max_tokens=128) is invalid for performance comparison: the systemd service was deliberately restarted at 01:14:51 while the request was still active, and the API returned engine_error. Do not treat its elapsed time as model throughput.
- A shorter deterministic tuning request is now the immediate comparison workload: identical 3,400-character prompt, temperature 0, max_tokens=8.
- Commit `17dce3b` adds a CUDA primitive that performs absorbed sparse MLA plus resident `o_proj` on-device and downloads only final [S,hidden] output. Its CUDA parity test uses an identity projection and passes on the RTX 4090.
- Commit `593c021` switches GLM-5.3 attention mode 1 to that fused primitive while leaving mode 2 verify-first behavior unchanged.
- The currently running 12 GB server was started before those two commits were built into the service process, so the pending 8-token result is an old-mode-1 baseline. Restart is required before measuring the fused path.


## 2026-10-06 01:14 PDT — 12 GB performance sample invalidated by code restart

- The fixed 3.4k-character / 128-token mode-1 performance request was still active when systemd intentionally restarted Colibri at 01:14:51 to load the newly committed sparse-MLA output-projection fusion.
- The client received `engine_error` with the server log explicitly saying `colibri engine is shutting down`; this is not a model/CUDA correctness failure.
- Do not use that request for throughput comparison. Rerun the identical workload on the post-fusion HEAD before changing resident budget.
- The terminated service reported a 118.9 GiB memory peak and 0 B swap peak.


## 2026-10-06 01:xx PDT — aggressive 21 GB jump and storage observation

- Per user direction, skipped the 16/18/20 GB resident sweep and requested `COLI_CUDA_RESIDENT_EXPERT_GB=21` directly with the existing 3 GB live reserve.
- Placement is safe to probe aggressively: `cuda_resident_expert_init` re-reads live free VRAM before every expert admission and stops unless free VRAM exceeds reserve + next expert logical footprint + 64 MiB allocator margin.
- The last old 12 GB mode-1 process emitted one-forward profile data before shutdown: attention 9.522 s, FFN 194.730 s, expert disk 177.677 s, head 0.374 s. This is not a clean end-to-end throughput benchmark, but it strongly identifies expert/storage time as dominant.
- The new 21 GB process spends cold-load time in Linux D state at `p9_client_rpc` while opening shards under `/mnt/e`. The model directory is about 202 GB on the Windows E: 9p mount.
- Native WSL ext4 has about 932 GB free, enough to stage a complete model copy. After the 21 GB VRAM measurement, A/B the same fixed workload from a native-ext4 model path before further kernel micro-optimization. Do not copy during a timing run because it would contend with the model reads.


## 2026-10-06 01:27 PDT — direct 21 GB stress qualification

- Skipped the intermediate 16/18 GB ladder per user direction and restarted directly with `COLI_CUDA_RESIDENT_EXPERT_GB=21`, reserve 3 GB, sparse MLA mode 1.
- Startup placement admitted 1,483 hot experts using 19.55 GiB persistent VRAM. The live placement guard stopped with 2.90 GiB free rather than blindly reaching the nominal 21 GB budget.
- Lazy dense/MLA allocations during the first request consumed almost all of that startup reserve: total VRAM reached 24,123 MiB used with only about 20 MiB free.
- Despite that extreme occupancy, the first two forwards completed without CUDA allocation failure or WSL swap use.
- Rolling counters after forward 1: attention 16.247 s, FFN 182.439 s, expert disk 165.843 s, head 2.761 s.
- Rolling counters after forward 2: attention 34.842 s, FFN 235.208 s, expert disk 201.328 s, head 5.296 s. Incremental forward 2 was therefore about 18.595 s attention, 52.769 s FFN, 35.485 s disk, 2.535 s head.
- Decode showed bursty but real GPU utilization; a point sample reached 83% SM utilization.
- The 8-token probe was intentionally cancelled after two successful forwards instead of spending several more minutes proving the same VRAM reuse property.
- Conclusion: 21 GB is stress-viable, but the startup reserve guard is not post-lazy-aware. Keep the user-visible 21 GB target, but future policy should reserve expected lazy CUDA state before expert admission.
- The next dominant problem is host expert-cache warming: first-forward disk time is much larger than attention. On this dedicated high-RAM host, enable `GLM53_PREWARM_EXPERTS=1` and size `GLM53_EXPERT_GB` aggressively enough to hold essentially/all routed experts before re-benchmarking.


## 2026-10-06 01:3x PDT — aggressive steady-state VRAM target

- Updated target: optimize for **23+ GiB stable total VRAM use** on the 24,564 MiB RTX 4090, not a conservative 21-22 GiB ceiling.
- The 21 GB resident-budget experiment placed 1,483 experts / 19.55 GiB persistent expert VRAM and then lazy dense/attention allocations drove total use to about 24,123 MiB, leaving only about 20 MiB free.
- That state did not immediately OOM; a point sample reached 95% GPU SM utilization at the near-full VRAM state. However the service was later deliberately/repeatedly restarted by experiment control, so this does not yet prove long-duration stability at ~24.1 GiB.
- Reserve policy should therefore become **transient-aware**: reserve only enough space for the largest expected lazy/workspace allocation plus fragmentation margin. Do not preserve multiple GiB of idle VRAM if measurements show a smaller margin is stable.
- Compute remains bursty: earlier 30-second mode-1 sampling at 12 GB residency saw only 2/30 nonzero one-second samples (27% and 16%), while the 21 GB near-full run produced at least one 95% sample. This indicates the GPU kernels can saturate the card, but host/CPU phases and synchronization gaps dominate duty cycle.
- Primary performance objective: increase sustained GPU duty cycle by extending device-resident pipeline continuity (attention -> o_proj -> following operations, resident experts, reduced host round trips), while keeping total VRAM as close to physical capacity as stability allows.


## 2026-10-06 01:xx PDT — 21 GB expert-first boundary and dense-first fix

- Expert-first 21 GB placement admitted 1,483 hot experts using 19.55 GiB persistent expert VRAM and initially left 2.90 GiB free by Colibri accounting.
- During the fixed request, lazy resident-matrix uploads consumed the remaining headroom: nvidia-smi reached 24,123 MiB used / about 20 MiB free. The process did not OOM, but this is not an acceptable steady-state reserve.
- Before cancellation the stress run reached three forwards with cumulative rolling profile: attention 57.860 s, FFN 268.688 s, expert disk 217.404 s, head 7.847 s. Treat as stress/profile evidence, not a throughput benchmark because the request was intentionally interrupted.
- Root cause: startup ordered hot-expert placement before ordinary resident CUDA matrices were lazily materialized.
- Commit `60401bd` adds startup resident-matrix prewarm before CUDA expert placement, so the same requested 21 GB tier self-clamps against true remaining VRAM rather than guessed future headroom. The same commit also contained a concurrent compatible host-expert-cache prewarm implementation.
- Host expert prewarm is enabled for the next run. It fills the already-sized RAM cache from routing history after VRAM placement, while preserving runtime miss/byte telemetry.
- Historical repeated service restarts around 01:36-01:37 were delayed completions of earlier blocking systemctl restart commands; no persistent restart helper remained afterward.


## 2026-10-06 01:41 PDT — dense-first 21 GB placement result

- Resident-matrix prewarm uploaded 552 ordinary resident matrices first, consuming 4.33 GiB VRAM and leaving 18.12 GiB free.
- With the requested expert budget still set to 21 GB and reserve 3 GB, hot-expert placement then self-clamped safely at 1,158 experts / 15.27 GiB persistent expert VRAM.
- Combined startup allocation was about 20.7 GiB, leaving 2.86 GiB free by Colibri accounting (about 3.47 GB free reported by nvidia-smi at the observation point).
- This fixes the earlier expert-first behavior where lazy dense uploads later drove free VRAM to about 20 MiB. Keep the aggressive 21 GB request; dense-first ordering now determines the true safe admitted expert tier automatically.
- Host expert-cache prewarm began after VRAM placement; while active it showed p9_client_rpc waits against /mnt/e, growing RSS, fixed VRAM, and zero swap.
