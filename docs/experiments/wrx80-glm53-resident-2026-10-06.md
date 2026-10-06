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
6. Only after the best resident/MLA profile is known, restart the expensive SC64 agent audit.

## 2026-10-06 01:xx PDT — 12 GB / MLA mode 1 live utilization

- Restarted with `COLI_CUDA_RESIDENT_EXPERT_GB=12`, reserve 3 GB, and `COLI_CUDA_GLM53_ATTN=1` so CUDA attention results are used directly instead of double-computing the CPU reference.
- The richer persisted route history now contains 2,308,320 expert selections.
- Before the fixed performance request: 12,045 MiB VRAM used / 12,098 MiB free.
- A 30-second one-second-cadence sample during the request observed only 2/30 samples with nonzero SM utilization (27% and 16%); the remaining 28 samples were 0%. Treat this as a duty-cycle indicator, not precise kernel utilization, but it demonstrates long host-side intervals between CUDA bursts.
- VRAM increased from about 13.6 GiB to 14.3 GiB during that 30-second interval as lazy CUDA state populated. No swap use was observed.
- Implication: more resident experts can still reduce weight-transfer stalls, but sustained throughput is currently limited by host-side phases / pipeline discontinuity. Device-resident pipeline work should follow the residency sweep.


## 2026-10-06 01:xx PDT — 12 GB resident + sparse MLA verify qualification

- Runtime configuration had already advanced to COLI_CUDA_RESIDENT_EXPERT_GB=12, reserve 3 GB, COLI_CUDA_GLM53_ATTN=2 (verify-first).
- Startup placed 847 hot experts using 11.17 GiB persistent VRAM and reported 11.28 GiB free immediately after placement.
- During the fixed qualification request, total VRAM rose through roughly 13-15 GiB as lazy dense/attention state populated, still leaving about 9 GiB free and using no swap.
- Sparse MLA verify samples for tokens 0-15 showed max absolute error at most 7.15256e-7. Relative error can look larger near zero (observed up to 0.0426316); absolute error remains tiny.
- Important benchmark rule: COLI_CUDA_GLM53_ATTN=2 performs CUDA plus the complete CPU reference and therefore is a correctness mode, not a performance mode. Performance measurements must use mode 1 after qualification.
