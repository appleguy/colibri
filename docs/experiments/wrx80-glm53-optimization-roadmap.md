# WRX80 GLM-5.3 Flash optimization roadmap

Last updated: 2026-10-06
Primary branch: `wrx80/glm53-cuda-resident`
Upstream base: `JustVugg/colibri:main`
Primary target host: WRX80, Threadripper PRO 5955WX (16C/32T), RTX 4090 24,564 MiB, WSL2 capped at 244 GiB.
Model: GLM-5.3-Flash grouped-int4 / gs=64.

This is the authoritative phase/status document for the WRX80 GLM-5.3 performance work. Keep it current as measurements or architectural decisions change. The chronological raw experiment log remains in `docs/experiments/wrx80-glm53-resident-2026-10-06.md`.

## Global objective and guardrails

Optimize for single-request agent latency and throughput while preserving exact model semantics closely enough for deterministic correctness checks.

The target is intentionally aggressive:

- Use **23+ GiB total VRAM** when stable on the 24,564 MiB RTX 4090.
- Do not reserve multiple GiB merely for comfort. Reserve the empirically required transient/workspace/fragmentation margin.
- Keep WSL system RAM highly utilized but preserve approximately 8 GiB launch headroom and the 4 GiB supervisor emergency floor.
- Treat any WSL swap growth as a pressure/failure signal, not normal operation.
- Separate prefill and decode measurements.
- Prefer fixed deterministic prompts, `temperature=0`, prefix reuse disabled for microbenchmarks, and identical workloads across A/B arms.
- Preserve CPU-only and prior-hybrid fallbacks until a new GPU path is qualified.
- Do not restart the expensive SC64 agent audit until the optimized configuration is materially faster and stable.

## Phase 0 — Measurement foundation

**Status: MOSTLY COMPLETE / ONGOING**

Purpose: make every later optimization attributable and reproducible.

Already available:
- rolling GLM-5.3 phase counters: attention, FFN, expert-disk, lm-head, forwards;
- CUDA resident-expert calls/hits/fallbacks and avoided-H2D accounting;
- VRAM/RAM/swap sampling;
- fixed short tuning probe;
- deterministic CUDA grouped-int4 correctness coverage;
- sparse CUDA MLA verify mode and direct performance mode;
- experiment log with invalidated runs explicitly marked.

Current evidence:
- one earlier first-forward profile at 12 GB residency showed approximately:
  - attention 9.522 s,
  - FFN 194.730 s,
  - expert disk 177.677 s,
  - head 0.374 s.
- 21 GB stress run forward 1:
  - attention 16.247 s,
  - FFN 182.439 s,
  - disk 165.843 s,
  - head 2.761 s.
- forward 2 incremental work was approximately:
  - attention 18.595 s,
  - FFN 52.769 s,
  - disk 35.485 s,
  - head 2.535 s.
- GPU duty cycle is bursty. Near-full VRAM state has produced point samples up to 95% SM utilization, while many one-second samples are 0%.

Remaining measurement work:
- add explicit router/indexer timing;
- split expert FFN into resident-GPU, streamed-GPU, CPU-matmul, host-cache wait/read, and synchronization buckets;
- add H2D/D2H byte/time counters per phase;
- record GPU high-water and largest transient allocation per request;
- retain exact Colibri commit and service environment with each benchmark result.

Exit criterion:
- every major wall-time bucket is directly measured rather than inferred.

## Phase 1 — Near-full VRAM residency

**Status: ACTIVE; CORE IMPLEMENTATION COMPLETE**

Purpose: use almost all useful 4090 memory for persistent hot experts and reusable dense state.

Implemented:
- persistent GLM-5.3 hot expert tier;
- route-history ranking from `.coli_usage`;
- complete gate/up/down expert triples resident in VRAM;
- grouped-int4 resident dispatch;
- resident no-sync path;
- avoided-H2D and resident timing counters;
- live free-VRAM guard during expert admission;
- prewarm of resident matrices before expert placement;
- configured resident budget accepts 21 GB.

Validated:
- 1 GB bring-up: 70 experts / ~0.92 GiB, resident hits, zero fallbacks.
- 12 GB: 847 experts / 11.17 GiB.
- expert-first 21 GB stress run admitted **1,483 experts / 19.55 GiB persistent expert VRAM**, but later lazy dense/MLA allocations drove total use to roughly **24,123 / 24,564 MiB**. It survived long enough to complete multiple forwards and reached a 95% SM-utilization point sample, but ~20 MiB free is not a robust steady-state margin.
- commit `60401bd` changed startup ordering to prewarm ordinary resident CUDA matrices before expert placement.
- dense-first 21 GB qualification uploaded **552 ordinary resident matrices / 4.33 GiB** first, then safely admitted **1,158 experts / 15.27 GiB**. Colibri reported ~2.86 GiB free after placement (nvidia-smi ~3.47 GB at observation time).
- the requested resident budget remains 21 GB; the live guard now self-clamps against the true dense-first residual capacity.

Known deficiency:
- dense-first ordering fixes the largest previously hidden allocation, but the final reserve is still a fixed policy rather than a measured transient/workspace model. Later attention/expert scratch and allocator fragmentation should be characterized before intentionally shrinking the safety margin toward the 23+ GiB total-use goal.

Next implementation:
- transient-aware residency planner:
  - track largest attention/expert scratch allocations after dense-first prewarm;
  - include allocator/fragmentation margin;
  - admit experts against **predicted steady-state high-water**, not only current free VRAM.
- experimentally reduce the remaining reserve only after long-run evidence; target the highest stable occupancy, ideally 23–24 GiB total.

Exit criterion:
- repeated long prefill + decode workloads complete without allocation failures, corruption, or performance collapse at the chosen near-full VRAM target.

## Phase 2 — Eliminate storage from steady state

**Status: ACTIVE**

Purpose: the dedicated host has enough system RAM that disk should not remain on the steady-state critical path.

Implemented / configured:
- `GLM53_PREWARM_EXPERTS=1`;
- `GLM53_EXPERT_GB=175`;
- parallel expert-cache prewarm;
- commit `039cd68` avoids duplicate host prewarm for every VRAM-resident expert even when the host cache could otherwise hold the full layer; CUDA failure can still lazily load the host fallback.

Observed problem:
- model lives at `/mnt/e/z-models/...`, a WSL drvfs/9p mount.
- cold model work has blocked in `p9_client_rpc`.
- model directory is approximately 202 GB.
- native WSL ext4 has approximately 932 GB free.

Next experiments:
1. complete full host-expert prewarm and verify disk time approaches zero on warmed requests;
2. copy the model once to native WSL ext4 without contending with a benchmark;
3. A/B identical cold-load, prewarm, prefill, and decode workloads from:
   - `/mnt/e` 9p/NTFS path;
   - native WSL ext4;
4. separately test native Windows in Phase 9.

Exit criterion:
- steady-state expert-disk time is negligible; storage choice no longer materially affects warmed decode.

## Phase 3 — Continuous device-resident activation pipeline

**Status: ACTIVE / PARTIAL**

Purpose: eliminate the long host gaps between high-utilization CUDA bursts.

Implemented:
- sparse GLM-5.3 MLA CUDA path;
- correctness verification mode;
- sparse MLA + resident `o_proj` fusion, downloading only final hidden output;
- backend already contains device-pointer pipeline primitives for GEMM, RMSNorm, RoPE, add, row-add, device attention, scratch, peer copy and synchronization.

Next steps:
1. retain the fused MLA output on-device rather than immediately downloading it;
2. feed it directly into the next normalization/projection/residual operations;
3. retain the residual stream on-device across layer boundaries where possible;
4. replace host round trips with explicit device dependency ordering;
5. minimize `cudaStreamSynchronize` calls and synchronize only at genuine host-consumption boundaries.

Exit criterion:
- decode shows sustained GPU duty rather than isolated 80–95% spikes separated by long 0% intervals.

## Phase 4 — GPU-native router and MoE decode

**Status: PLANNED; REUSABLE BACKEND EXISTS**

Purpose: remove serial CPU routing from the decode critical path and make resident experts valuable even at S=1.

Current GLM-5.3 path:
- router dot products over 288 experts are CPU-side;
- top-8 selection is CPU-side;
- current small-group CPU cutoff was measured when expert weights still crossed PCIe.

Reusable backend:
- `coli_cuda_pipe_router`;
- resident expert issue/take primitives;
- device-resident accumulation helpers;
- other Colibri engines already use these paths.

Next steps:
1. wire GLM-5.3 router matrix + bias into resident CUDA tensors;
2. run logits + top-k gating on GPU;
3. dispatch resident experts directly from device-side activation;
4. re-sweep small-group cutoff 0/4/8/16/32/64 after weight and activation transfers are removed;
5. retain CPU router as deterministic fallback/reference.

Exit criterion:
- S=1 decode routing/expert work remains primarily device-side and beats the current hybrid policy on wall time.

## Phase 5 — GPU-native sparse indexer and attention state

**Status: PARTIAL / PLANNED**

Purpose: keep sparse-attention state and selected-index construction on device.

Current CPU-side work includes:
- q_a / q_b projections and normalization;
- indexer queries/keys/gates/head weights;
- sparse selection;
- host arrays for latent/index state.

Next steps:
- persist latent/KV/index state in VRAM where capacity permits;
- move sparse index scoring/selection to CUDA;
- avoid uploading q/latent/selected arrays every layer;
- chain GPU indexer -> sparse MLA -> o_proj -> residual without host materialization.

Exit criterion:
- attention path requires no per-layer bulk host-device round trip during decode.

## Phase 6 — CPU utilization and CPU/GPU overlap

**Status: ACTIVE RESEARCH**

Purpose: improve unavoidable host work and use the 5955WX concurrently with the GPU rather than synchronously blocking it.

Host:
- Threadripper PRO 5955WX, 16 physical Zen 3 cores / 32 SMT threads, one NUMA node.
- service currently uses `OMP_NUM_THREADS=16`.
- representative process samples have often shown only ~2–5 fully occupied cores despite available CPU capacity.

Opportunities:
1. benchmark 16 physical threads versus 24/32 SMT threads separately for:
   - prewarm;
   - prefill;
   - warmed decode;
2. parallelize/vectorize GLM-5.3 router work across tokens/experts where CPU fallback remains;
3. improve top-k selection rather than repeatedly rescanning 288 scores;
4. parallelize remaining indexer/MLA host preparation where dependency-safe;
5. use SIMD for router/indexer reductions and normalization hot loops;
6. preserve aligned, reusable buffers and reduce malloc/free churn;
7. tune prefill chunk size (128/256/512 and larger if RAM allows) to increase CPU and GPU batch efficiency;
8. **overlap** CPU work with CUDA:
   - prepare next-layer routing/index metadata;
   - prefetch cold host experts;
   - build next dispatch while current GPU kernels execute.

Important principle:
- higher CPU utilization is not itself the objective. The objective is wall-time reduction. Avoid extra CPU threads if they only create memory-bandwidth contention.

Exit criterion:
- CPU is either concurrently useful while the GPU runs or demonstrably removed from the critical path; no obvious serial host bubble remains.

## Phase 7 — Decode specialization and graph/scratch stabilization

**Status: PLANNED**

Purpose: optimize the common S=1 / small-batch agent decode loop after residency and pipeline boundaries settle.

Candidate work:
- persistent scratch and stable device pointers;
- eliminate per-token allocation;
- CUDA graph capture where control flow and pointer stability permit;
- fuse tiny residual/norm/router kernels when launch overhead dominates;
- tune tensor-core thresholds specifically for decode versus prefill;
- revisit speculative decoding only when resident hit rate is near-full and the widened expert union no longer dominates.

Exit criterion:
- optimized S=1 decode policy is separately tuned from prefill and is stable on long generations.

## Phase 8 — Second GPU / multi-GPU

**Status: PLANNED; GLM-5.3 IS SINGLE-DEVICE TODAY**

Backend capability:
- generic CUDA backend supports multiple devices;
- peer-copy and resident expert issue/take primitives already exist;
- other engines contain multi-device head sharding and expert-parallel code;
- GLM-5.3 currently initializes exactly one CUDA device: `coli_cuda_init(&g_cuda_device, 1)`.

### If a second 4090 is installed in the same WRX80 host

This is the preferred topology.

Potential wins:
- roughly double usable expert VRAM;
- keep a much larger fraction of hot experts resident;
- partition routed experts by heat/load;
- execute expert groups concurrently across GPUs;
- optionally shard attention/dense work later.

Because RTX 4090 has no NVLink, PCIe/P2P costs matter. Prefer expert parallelism where activations/results are small relative to expert weights rather than naïve tensor parallelism that moves large tensors every layer.

Expected benefit must be measured; a perfect 2x is unlikely for end-to-end latency. The upside is strongest once CPU/storage stalls are removed.

### If the second 4090 stays in another machine

Treat this as a distributed systems project, not a normal second-GPU configuration.

Most promising uses:
1. independent second inference worker for concurrency/parallel agent tasks;
2. remote expert worker with a persistent expert bank and long-lived low-overhead RPC;
3. coarse pipeline/segment execution where network crossings are infrequent.

Avoid fine-grained tensor parallelism over ordinary Ethernet. The remote machine cannot participate in CUDA P2P, and per-layer network latency can erase compute gains.

Exit criterion:
- prototype shows a meaningful end-to-end latency or throughput gain on the actual interconnect before deeper distributed integration.

## Phase 9 — Native Windows A/B

**Status: HIGH-PRIORITY EXPERIMENT, SMALL PORTABILITY GAP**

Rationale:
- current model source is Windows E: exposed to WSL through drvfs/9p with 64 KiB `msize`;
- cold model reads have blocked in `p9_client_rpc`;
- native Windows Colibri/CUDA is an established supported configuration;
- repository benchmarks include native-Windows NVMe results up to ~10.6 GB/s on suitable hardware.

What already works in the tree:
- `glm53.exe` target exists;
- Windows host uses MinGW;
- CUDA backend builds as `coli_cuda.dll` via nvcc + MSVC;
- runtime loader already exposes many resident-pipeline, router, peer-copy and multi-GPU symbols.

Current portability gap:
- the newly added GLM-5.3 sparse-attention functions
  `coli_cuda_attention_absorbed_sparse_batch` and
  `coli_cuda_attention_absorbed_sparse_project_batch`
  exist in `backend_cuda.{h,cu}` but are not yet resolved/wrapped by `backend_loader.c`.
- native Windows therefore needs a small loader/export integration before it can test the current optimized GLM-5.3 branch equivalently.

Experiment order:
1. benchmark storage alone first:
   - Windows native iobench against E:;
   - WSL /mnt/e iobench;
   - WSL ext4 model copy;
2. patch/validate Windows loader symbols for current GLM-5.3 CUDA additions;
3. build `glm53.exe CUDA_DLL=1 ARCH=native` and `coli_cuda.dll CUDA_ARCH=sm_89`;
4. run the exact same deterministic fixed workload and profiling configuration;
5. compare cold load, host prewarm, prefill, warmed decode, CPU utilization, GPU duty, and total wall time.

Decision rule:
- keep native Windows only if it wins materially after the full host-cache warm state, not merely because cold filesystem reads are faster.

## Phase 10 — Production qualification and SC64 audit restart

**Status: BLOCKED ON PERFORMANCE WORK ABOVE**

Before restarting the expensive agent audit:
- stable near-full VRAM configuration;
- storage eliminated or understood;
- fixed CPU/current-hybrid/resident-hybrid comparisons;
- long deterministic generation without CUDA allocation errors;
- representative agent turn succeeds;
- service restarts/benchmark controllers cleaned up;
- exact launch profile checked into the appropriate automation/harness repository.

Then run:
- 12-hour wall budget;
- full profiling active;
- preserved audit artifacts;
- comparison against previous Qwen/GLM results.

## Current priority order

1. Phase 1 transient-aware near-full residency.
2. Phase 2 full host prewarm + native ext4 storage A/B.
3. Phase 3 device-resident pipeline continuation.
4. Phase 4 GPU router/resident MoE decode.
5. Phase 6 CPU overlap/thread/vectorization work in parallel.
6. Phase 9 native Windows A/B.
7. Phase 8 multi-GPU prototype when second-GPU topology is decided.
8. Phase 5 deeper GPU sparse indexer/state residency.
9. Phase 7 decode specialization.
10. Phase 10 restart expensive SC64 audit.

## Do-not-repeat / known invalid measurements

- The 12 GB / 128-token run interrupted by an intentional systemd restart is invalid for throughput.
- CUDA attention mode 2 performs GPU + CPU reference verification and must not be used as a performance arm.
- A fixed startup free-VRAM reserve is not sufficient evidence of steady-state headroom because lazy CUDA allocations occur after expert placement.
- Do not infer GPU duty from one `nvidia-smi` sample; the workload is bursty.
