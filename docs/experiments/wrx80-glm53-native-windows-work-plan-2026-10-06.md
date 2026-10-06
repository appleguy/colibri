# WRX80 GLM-5.3 Native Windows Work Plan — 2026-10-06

## Purpose

This is the durable execution cursor for the native Windows GLM-5.3 optimization program on WRX80.

Canonical branch: `wrx80/glm53-cuda-resident`

Canonical WSL checkout: `/home/tech/src/colibri`

Canonical native Windows checkout: `E:\z-src\colibri-native`

Native model: `E:\z-models\GLM-5.3-Flash-colibri-int4-g64`

Result logs: `E:\z-results\glm53-native-2026-10-06\`

Do not launch a competing WSL inference while a native Windows model run is active.

## Current experimental baseline

Run 0 completed natively with:

- `COLI_CUDA_GLM53_CHAIN=2`
- `COLI_CUDA_GLM53_ROUTER=2`
- `COLI_CUDA_GLM53_INDEXER=2`
- `COLI_CUDA_RESIDENT_EXPERT_GB=18`
- `COLI_CUDA_RESIDENT_RESERVE_GB=3`
- `GLM53_EXPERT_GB=175`
- `GLM53_PREWARM_EXPERTS=1`
- `GLM53_PROFILE_VRAM=1`
- `GLM53_PROF_EVERY=1`

Observed:

- CUDA init and model load succeeded.
- Resident matrices: 4.33 GiB VRAM.
- Router residency: 252 MiB VRAM.
- Resident expert tier: 1142 experts / 15.06 GiB VRAM.
- Peak CUDA used: 21985.5 MiB.
- Minimum CUDA free: 2578.0 MiB.
- Indexer qualification observed exact selected-index parity with no reported mismatches/failures.
- Router selected-index parity matched in reported checks; weight drift was around 1e-7 to 1e-6.
- Shared/resident MoE device-row drift was tiny.
- Decode reached 0.934 tok/s for the two-token qualification tail.
- All-resident expert-set coverage was 0/84 under the conservative resident target.
- Chain verification exposed large real-model mHC pre/norm drift during prefill after the first site.
- CPU stayed authoritative in chain mode 2, so the run remained a correctness qualification rather than an unsafe production path.

Run-0 log:

- `E:\z-results\glm53-native-2026-10-06\q1-parity-short.log`

## Unit graph

### N00 — Native run harness and artifact discipline

Class: independent infrastructure  
Priority: P0  
Status: ACTIVE

Goal:

Make every Windows run reproducible and comparable.

Work:

- script the fixed deterministic prompt and environment;
- capture exact git commit and binary timestamps;
- capture wall time, stdout/stderr, profile lines, GPU samples, CPU/RSS, disk throughput, and exit code;
- retain one directory per run;
- record environment variables and native/WSL service state;
- make the harness refuse to start when another full GLM workload is active.

Dependencies: none.

Can proceed in parallel with N10, N20, N30, N40.

Exit:

One command launches a guarded, logged native test run and emits a compact summary.

---

### N10 — Chain mode-2 real-model drift isolation

Class: correctness  
Priority: P0  
Status: ACTIVE / FIRST FAILURE OBSERVED

Goal:

Find why real-model mHC chain verification diverges during prefill, starting at layer 0/site 1 while layer 0/site 0 is near numerical noise.

Work units:

- N10.1: reproduce the smallest failing layer/site with extra diagnostic output;
- N10.2: verify `cuda_hc_site_prepare()` maps site-0/site-1 fn/base/scale/norm tensors correctly;
- N10.3: distinguish uploaded-parameter mismatch from input/residual-state mismatch;
- N10.4: test full real geometry at S=1 and S=128 outside the full model loop if practical;
- N10.5: determine whether resident-stream continuity is contaminating later verification comparisons;
- N10.6: add a regression test for the actual bug;
- N10.7: fix one root cause at a time and re-run chain mode 2.

Dependencies:

N10.1 -> N10.2/N10.3 -> N10.4/N10.5 -> N10.6 -> N10.7.

Independent of router/indexer promotion as long as chain mode remains disabled or verification-only.

Exit:

Real-model chain verification is clean enough to justify a production-mode experiment, or the valid qualification scope is narrowed and documented.

---

### N20 — Router/indexer short-run qualification ladder

Class: correctness + performance  
Priority: P0  
Status: READY

Goal:

Promote already-healthy decode pieces independently of the chain issue.

Serialized runs:

- N20.1 B1: `CHAIN=0, ROUTER=2, INDEXER=2`, fixed 551-token prompt, 8–16 decode tokens;
- N20.2 B2: `CHAIN=0, ROUTER=1, INDEXER=2`, identical workload;
- N20.3 B3: `CHAIN=0, ROUTER=1, INDEXER=1`, identical workload.

For each:

- require zero indexer mismatches/failures in mode 2;
- require router selected-index parity and bounded weight drift in mode 2;
- record warmed decode wall time and tok/s;
- record fallback counts;
- preserve exact output text/token ids if available;
- collect utilization and VRAM telemetry.

Dependencies:

N20.1 -> N20.2 -> N20.3.

Independent of N10 after chain mode is disabled.

Exit:

Router and indexer authoritative modes are either qualified or blocked by a reproducible defect.

---

### N30 — Selected-index / compact-latent boundary profiling

Class: instrumentation  
Priority: P0  
Status: READY

Goal:

Quantify the remaining synchronization bubble around the GPU indexer and compact latent staging.

Add counters one at a time for:

- N30.1 GPU indexer compute/launch;
- N30.2 selected-index D2H + stream synchronize;
- N30.3 compact latent host gather/pack;
- N30.4 pinned staging reuse wait;
- N30.5 compact latent H2D;
- N30.6 aggregate check that subphases sum sensibly to current `index_select` / attention time.

Dependencies:

None for source instrumentation.  
Use N20 runs to gather representative data.

Independent of N10 and N40.

Exit:

The next attention optimization target is selected from measured wall-time contribution rather than intuition.

---

### N40 — Resident expert coverage optimization

Class: performance  
Priority: P0/P1  
Status: READY / CURRENT COVERAGE 0%

Goal:

Increase the probability that every routed expert needed by an S=1 decode layer is already resident.

Work:

- N40.1 log selected expert sets and which member caused each non-all-resident set;
- N40.2 compute marginal coverage gain from additional experts/bytes;
- N40.3 compare static historical ordering against frequency-aware ordering;
- N40.4 compare against co-occurrence-aware admission;
- N40.5 conservative VRAM sweep, preserving transient/cache headroom;
- N40.6 measure streamed fallback frequency and churn for each policy.

Important baseline:

18 GB requested resident target produced 15.06 GiB actual expert residency and ~2.58 GiB minimum free CUDA memory.

Dependencies:

N40.1 -> N40.2 -> N40.3/N40.4 -> N40.5/N40.6.

Can proceed independently of N10/N20/N30.

Exit:

Materially improved all-resident expert-set coverage without allocation failures, excessive eviction, or worse wall time.

---

### N50 — Native utilization timeline

Class: profiling  
Priority: P1  
Status: READY

Goal:

Correlate GPU bubbles with CPU, disk, H2D, and synchronization phases.

Capture at 100–500 ms cadence:

- GPU utilization;
- GPU power;
- VRAM used/free;
- process CPU;
- RSS/private bytes;
- disk read throughput;
- optional per-thread CPU if useful.

Correlate with GLM rolling profile timestamps.

Dependencies:

Best paired with N20 and later N70 runs.

Independent implementation.

Exit:

Major GPU-idle regions have a plausible causal phase attribution.

---

### N60 — Launch-granularity and tiny-kernel audit

Class: performance  
Priority: P1/P2  
Status: BLOCKED ON N30/N50 DATA

Goal:

Only after large transfer/sync boundaries are measured, inspect launch-bound decode work.

Candidates:

- sparse score/top-pool selection;
- mHC pre/rmsnorm;
- residual adds;
- router micro-stages;
- expert bookkeeping.

Dependencies:

N30 + N50.

Exit:

Only kernels proven launch-bound are fused or restructured.

---

### N70 — Medium-context native qualification

Class: serialized experiment  
Priority: P1  
Status: BLOCKED ON SHORT-RUN GATES

Goal:

Run a deterministic medium-context workload using the best qualified configuration.

Required before start:

- N20 authoritative choices settled;
- no unresolved correctness issue on enabled paths;
- N30 telemetry available;
- adequate VRAM headroom measured.

Capture:

- total wall;
- prefill wall;
- decode wall/tok/s;
- indexer/router/attention/FFN buckets;
- resident coverage;
- utilization timeline;
- VRAM peak/min-free;
- fallbacks and diagnostics.

Dependencies:

N20, N30; N40 policy may be included if already qualified.

Exit:

A stable medium-context native baseline suitable for comparison.

---

### N80 — Long-context native qualification

Class: serialized expensive experiment  
Priority: P1  
Status: BLOCKED ON N70

Goal:

Exercise the intended S=1 long-context regime without wasting a multi-hour run on an unqualified configuration.

Dependencies:

N70 must be clean.

Exit:

Measured native Windows long-context performance and utilization with preserved full logs.

---

### N90 — Native vs preserved WSL comparison

Class: analysis  
Priority: P1  
Status: BLOCKED ON N70/N80

Compare against:

- `docs/experiments/wrx80-glm53-final-wsl-profile-2026-10-06.json`
- `docs/experiments/wrx80-glm53-final-wsl-environment-2026-10-06.txt`

Metrics:

- warmed decode wall/tok/s;
- prefill wall;
- attention/indexer/router/FFN contribution;
- expert disk time;
- GPU duty/power;
- host CPU/RSS;
- VRAM peak/min-free.

Dependencies:

N70, preferably N80.

Exit:

Platform decision is based on warmed inference, not only storage/load speed.

## Parallel lanes

Lane A — correctness:

N10 chain drift isolation.

Lane B — short decode qualification:

N20 router/indexer ladder.

Lane C — telemetry:

N30 selected-index boundary + N50 utilization timeline.

Lane D — residency:

N40 expert coverage/admission.

These four lanes are substantially independent and can advance while another lane is blocked.

## Serialized critical path

N00 run harness
-> N20.1 mode-2 short gate
-> N20.2 router authoritative
-> N20.3 router+indexer authoritative
-> N70 medium context
-> N80 long context
-> N90 native-vs-WSL decision

N10 joins the critical path only before chain mode 1 is enabled.

N30 should finish before N70 so medium/long runs carry useful subphase telemetry.

N40 can join at N70 only after a short-run policy A/B proves it does not harm stability.

## Current cursor

Highest-value independent next actions:

1. N00: finish the reusable native run harness.
2. N20.1: run CHAIN=0 / ROUTER=2 / INDEXER=2 for 8–16 decode tokens.
3. N10.1/N10.2: isolate layer 0 site-1 chain drift.
4. N30.1–N30.5: add selected-index / compact-latent timing counters.
5. N40.1: log exactly why each decode expert set is not fully resident.

Do not start N70/N80 yet.

## Wake prompt

Resume the WRX80 native-Windows GLM-5.3 plan from `docs/experiments/wrx80-glm53-native-windows-work-plan-2026-10-06.md` and the execution ledger. If inference is stalled or idle, verify that first, preserve any useful run output, then advance the highest-priority independent unit (N10, N20, N30, or N40) with a tiny validated commit and update the plan/ledger before moving on.


## Progress checkpoint — 2026-10-06 12:36 local

- N20.1 B1 PASSED: CHAIN=0 / ROUTER=2 / INDEXER=2, 8 decode tokens, 0.975 tok/s, clean reported indexer/router parity.
- N20.2 B2 PASSED: CHAIN=0 / ROUTER=1 / INDEXER=2, 8 decode tokens, 0.982 tok/s, 8/8 reported exact indexer checks, zero mismatch/failure.
- The first marshaled N20.3 attempt was invalid because the PowerShell runner split the multiword prompt into separate argv words. It exited before model loading. `c70ae1f` fixes and smoke-tests prompt quoting.
- N20.3 B3 is now the active serialized gate: CHAIN=0 / ROUTER=1 / INDEXER=1, same 551-token prompt and 8-token tail. Require authoritative indexer success and zero fallback.
- N30 and N40 remain independent work if B3 is running or blocked.
