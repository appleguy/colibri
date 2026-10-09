# WRX80 GLM-5.3 Flash: next native Windows optimization pass
Date: 2026-10-08
Worktree: `C:\src\colibri-kda-batch-20261008`
Branch: `experiment/glm53-kda-batch-20261008`

## Priority and operating constraints

The user explicitly prioritized **Qwen3.8 Q6**, then **DeepSeek V4 Flash**, then **GLM-5.3 Flash** for the full-repository `z-voice` audits. This note is offline analysis and a benchmark plan, not authority to interrupt either current audit. No subscription API, no WSL inference, no changes to the running model binary, and no parallel full-model loads. The live Windows launcher is outside this Git worktree.

Qwen's first 4,825-token prompt was processed at ~24 token/s by native llama.cpp and its first decode was ~8.7 token/s. These are Qwen numbers, not GLM numbers. Native GLM is idle after reprioritization.

## What previous work has already implemented

The main fork contains the resident CUDA expert tier, resident matrix/router prewarm, sparse CUDA MLA, GPU router/indexer, and the experimental chained residual/shared-expert path. Do **not** repeat this work. The separate KDA branch adds guarded batched KDA prefill projections and hoists the recurrent decay exponent per head. Its Windows GCC syntax check passed on 2026-10-08. The scalar-versus-batched numerical oracle requires an idle gateway and has **not** been rerun while Qwen is active.

Previously recorded native-Windows short decode rates were approximately 1.27–1.35 tokens/sec (frozen 551-token prompt, eight generated tokens). These are small, variable measurements, not an established full-context result.

## Fresh live GLM evidence before handoff to Qwen

The earlier native Flash engine launched with `COLI_CUDA=1`, `OMP_NUM_THREADS=16`, `GLM53_EXPERT_GB=160`, `GLM53_PREWARM_EXPERTS=1`, `COLI_CUDA_RESIDENT_EXPERT_GB=18`, reserve 3 GiB, GPU attention/router/indexer mode 1 and `COLI_CUDA_GLM53_CHAIN=0`. Its service was configured with `--ram 230 --ctx 65536 --kv-slots 1`; the cache allocation is separately constrained by the 160-GB expert env setting.

At rolling forward **41**, cumulative profiler (seconds):
- `attn=922.063`, `ffn=755.575`, `router=163.180`, `indexer=70.338`, `head=8.264`, `disk=0.250`.
- FFN subcategories: CPU routed experts `532.822`; resident GPU `17.735`; shared `29.559`; streamed GPU `0.215`. Some categories nest or overlap; **do not sum everything as independent wall-clock time**.
- GPU: ~21.5 GiB used of 24.6 GiB, ~1.9 GiB free at peak. Many GPU occupancy snapshots were single digit or low double digit. Native WRX80 GPU 4090 and 16 physical CPU cores/32 SMT threads.
- Model directory is ~201.6 GiB. The RAM budget exceeds its size, but anonymous allocation, Windows file cache, CUDA residency, and actual working set mean that `--ram 230` **does not prove 100% RAM residency**. The warmed disk timer near zero is the stronger relevant observation.

**Critical:** repeated HTTP 502s came from the **Mac peer proxy's 900-second response-header deadline**, not from a Colibri model error. The local full-repo audit uses a direct SSH tunnel to Windows localhost port 19000 for Qwen and DeepSeek. GLM's eventual long-context run should bypass that extra Mac proxy; if the Windows model router itself has a header deadline, connect directly to the native Colibri port only **after the model is correctly loaded**.

## Validated immediate improvement: fair CUDA resident-expert admission

Offline replay using the present model's `.coli_usage` with 1,142 expert slots:
| Placement policy | Overall historical selections covered | Weakest layer | Experts per layer |
|---|---:|---:|---:|
| Global heat | 24.29% | 3.78% | 5–38 |
| Round-robin fair | 24.00% | 16.31% | 27–28 |

Historical simulator also found 1,000 and 1,200-slot fair-policy improvements in the weakest layer. This is **routing-history coverage**, not verified speedup. Native Windows launcher now sets `COLI_CUDA_RESIDENT_LAYER_FAIR=1` **only** for `--model flash` on its NEXT load, with a rollback copy at `native_windows_entry.py.before-glm-fair-20261008`. Neither Qwen nor DeepSeek was modified.

## Next isolated A/B arms, in this order

1. **Fair admission baseline:** when GLM's turn comes, capture exact binary SHA/Colibri git commit, full env, engine startup prewarm, free RAM, VRAM, cache hit/miss and cold-vs-warmed timing. Compare to frozen global-heat results with identical short decode and medium/long prefill prompts. Prefer correctness and wall latency over abstract residency coverage.
2. **Host↔GPU expert cutoff:** keep baseline frozen, compare `COLI_CUDA_EXPERT_MIN_ROWS=32` (existing), 16, 8 and 4. Source `glm53.c` dispatches groups smaller than the threshold to CPU. Hypothesis: smaller CUDA groups may reduce CPU-routed MoE, but launch/transfer overhead may make them slower. Capture per-stage CPU MoE, resident/streamed GPU MoE, H2D/D2H and output parity.
3. **Prefill tile:** with the best proven cutoff, compare `GLM53_PREFILL_CHUNK=128/256/512` (and existing default) for identical prompts; seek to amortize routing and GPU expert grouping. Do not infer prefill gains from decode-only data.
4. **KDA branch:** run `run-kda-oracle.ps1` only after the gateway is idle. The tool enforces that gate. Require scalar/batched output equivalence, actual exercised batch path, and improvement on long-context prefill before promoting `GLM53_KDA_BATCH_PROJ=1` to the native Windows production branch.
5. **GPU chain:** retain `CHAIN=0` as the safe baseline. Re-qualify mode 2 numerical drift and actual device branch coverage, then compare `COLI_CUDA_GLM53_CHAIN=1` for the short deterministic decode, afterward long-context. Do not promote from source tests alone.
6. **Memory vs storage:** increase `GLM53_EXPERT_GB=160` toward historical 175 only if missed experts, disk reads, or cache residency measures warrant it. Remaining free RAM by itself is not evidence of a bottleneck. Avoid swapping and avoid double-caching VRAM-resident experts in system RAM.
7. **VRAM reserve:** measure true peak after all lazy allocations before considering lower than 3 GiB reserve. The old expert-first 21-GiB experiment left ~20 MiB free, which is too fragile. Dense-first ordering is already implemented.

## Evidence collection

A passive 30-second native Windows sampler is launched as
`C:\Users\Tech\.local\share\z-agent-inference-v2\voice_audit_perf.py`.
Its JSONL is `...\logs\voice-audit-perf.jsonl`; records which WRX80 checkpoint is active alongside CPU%, RAM, 4090 GPU utilization/VRAM/power and backend profile information. Original engine logs remain under `...\logs\*-native-backend.log`. Audit client stages persist in Mac `~/.local/state/z-voice-audits/2026-10-08/`.

Do not change the GLM binary or restart the Windows model router while the Qwen or DeepSeek audits are running. Benchmark only at the later GLM slot, keeping failed trials and numerical error results explicit.
