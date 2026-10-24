# RTX 3080 GEMM stall analysis, 2026-10-24

C is analysis only. No compiler, runtime, GEMM emitter or model changes were made. A and B1 remain assigned to the 4090 agent. Stop for the user's decision before changing the GEMM emitter.

The current evidence does **not** support a general main-loop throughput deficit. Two different effects account for the measured gap:

1. CUTLASS selects a smaller tile for `o` and Stream-K for `down`, reducing partially filled SM waves. Installed fixed-configuration controls isolate approximately **5-6 ms per 42-layer step** from these choices at 1695 MHz.
2. With matching ordinary tiles, Strict spends more work on strided scale/bias setup and its shared-memory epilogue. The remaining measured fixed-tile gap is approximately **8-10 ms per step**, principally qkv/up. PC samples locate the excess outside the main loop. These figures are headroom estimates from isolated kernels, not demonstrated full-model improvements or independently additive speedup promises.

## Conditions and checks

- Strict `4d1cafc9`, StrictComfy `26fc436`, driver 621.01, RTX 3080 10 GB / 68 SMs, comfy-kitchen 0.2.39, embedded Python 3.13.12.
- Final timings use `nvidia-smi -lgc 1695,1695` and `-lmc 9756,9756`. Actual CUDA/P2 clocks were **1695 / 9506 MHz**. Requested memory clock and actual P2 clock must not be confused. The user's Afterburner offsets, voltage, power and temperature limits were unchanged.
- Two telemetry repeats sampled every 100 ms: **156 high-load samples, all 1695/9506 MHz**, maximum GPU temperature 63.28 C, neither software nor hardware thermal slowdown active. NCU effective graphics-clock measurements were approximately 1.684-1.691 GHz. Initial 1800-MHz-target timings gave a 13-14 ms gap, but later telemetry caught 1755-1770 MHz dips, so final comparisons below use the lower lock. An initial cold down capture also measured 1.230 GHz and is excluded.
- Reused the existing graph-checked prequantized GEMM runner. Each graph contains 50 launches; six rounds alternate forward/reverse variant order. CUDA events measure graph execution, excluding Python gaps, quantization and weight packing.
- Identical seeded INT8 inputs/weights and numerical scales, represented in each backend's required layout; BF16 output, no effective bias. FP32 reference, output tolerance and Strict tail sentinel pass for every shape/configuration. This diagnostic uses constant scales and does not establish bitwise equivalence for arbitrary model data.
- Strict entry is the current `strict_device_wholes_fast`, not the historical generic `strict_device_wholes`. `NativeRuntimeTokens.mlir` has no changes since its archived build. PTX SHA-256: `9c3a31a3c6ccaccc737c62f2b14c51ab3cfac757d30855a9992546413c5b8ad5`.

## Timings and actual dispatch

Dimensions are M tokens, K reduction width, N output channels. Final six-round comparison without concurrent NVML polling, median microseconds:

| Projection | M x K x N | Strict | Installed CUTLASS | Difference x 42 |
|---|---|---:|---:|---:|
| qkv | 3740 x 3072 x 9216 | 1198.8 | 1125.2 | 3.09 ms |
| o | 3740 x 3072 x 3072 | 443.2 | 392.0 | 2.15 ms |
| up | 3740 x 3072 x 16384 | 2143.8 | 2040.8 | 4.33 ms |
| down | 3740 x 8192 x 3072 | 1068.2 | 936.6 | 5.53 ms |
| Sum per layer / step | | 4854.0 | 4494.6 | **15.09 ms** |

The two independent telemetry batches give 17.40 and 15.47 ms equivalent gaps. Stable clocks do not remove all Windows/graph timing variation; treat the isolated total as approximately **15-17 ms** at this lock. It cannot be directly subtracted from an unlocked full-model baseline.

| Kernel | Tile / stages | Threads | Registers | Dynamic shared | Blocks / waves |
|---|---|---:|---:|---:|---:|
| Strict, all shapes | 128 x 256 x 64 / 3 | 256 | 254 | 73728 B | qkv 1080/15.88; o/down 360/5.29; up 1920/28.24 |
| CUTLASS qkv/up, config 0 | 128 x 256 x 64 / 3 | 256 | 226 | 73728 B | same as Strict |
| CUTLASS o, config 1 | 128 x 128 x 64 / 4 | 128 | 230 | 65536 B | 720/10.59 |
| CUTLASS down, config 13 | 128 x 256 x 64 / 3, LeanStreamK | 256 | 254 | 73728 B | 340/5.00 |

All have one resident CTA/SM. Achieved occupancy is approximately 16.66% except o's four-warp CUTLASS tile, approximately 8.33%. Thus the earlier same-tile/stages/occupancy statement holds for qkv/up, **not all four installed dispatches**. No dynamically executed register-spill instructions were reported for these captures. Strict's allocated 24 local bytes are not evidence of a hot-loop spill.

Installed `cutlass_int8_dequant_config` allows a controlled comparison without rebuilding either backend. All configurations below pass the same numerical check. Two telemetry batches and the final clean batch, microseconds:

| Projection / batch | Strict | CUTLASS config 0 | Config 1 | Config 13 |
|---|---:|---:|---:|---:|
| qkv 1 | 1245.0 | 1189.2 | 1201.2 | 1186.0 |
| o 1 | 468.4 | 444.0 | 424.7 | 405.9 |
| up 1 | 2226.5 | 2083.2 | 2144.6 | 2046.7 |
| down 1 | 1097.6 | 1072.7 | 998.2 | 972.2 |
| qkv 2 | 1218.7 | 1147.6 | 1183.1 | 1157.0 |
| o 2 | 443.7 | 437.1 | 405.8 | 406.8 |
| up 2 | 2216.6 | 2072.3 | 2136.2 | 2050.2 |
| down 2 | 1093.8 | 1073.3 | 1004.1 | 965.4 |
| qkv clean | 1198.8 | 1133.4 | 1144.4 | 1118.6 |
| o clean | 443.2 | 427.5 | 396.9 | 406.9 |
| up clean | 2143.8 | 2041.4 | 2111.1 | 2025.8 |
| down clean | 1068.2 | 1052.0 | 970.1 | 946.3 |

For down, changing only CUTLASS config 0 -> 13 saves **100.5-107.9 us**, or **4.22-4.53 ms/step**. Config 0 -> 1 on o saves 19.3-31.3 us, or 0.81-1.31 ms/step. This is dispatch/load balance evidence, not an inferred warp scheduling fix.

Down has 360 output tiles. Ordinary scheduling needs six waves; the last has only 20/68 SMs occupied. LeanStreamK assigns 272 ordinary tiles plus the remaining 88 tiles' K iterations across 68 blocks: 340 blocks total, five waves, with reductions for split tiles. Idealized tail headroom is `6 / (360/68) - 1 = 13.33%` before overhead. Ordinary CUTLASS 1052.0 versus Stream-K 946.3 us is 11.17% faster, consistent with this explanation. An earlier 1800-MHz-target changed-shape check also corroborates this: at 4352 tokens, 408 tiles fill six waves exactly and down's gap shrinks to 2.25%. That supplementary check is not a final locked-clock model timing.

For o, wave rounding costs `ceil(360/68)/(360/68) = 1.1333` with the large tile versus `ceil(720/68)/(720/68) = 1.0389` with the smaller tile. The matching release explicitly implements this `wave_guard`; the installed kernel name/resources confirm the selection. Stream-K is not automatically best for qkv: config 13 is slower there.

Source checked against installed release: [INT8 dispatch/configurations](https://github.com/Comfy-Org/comfy-kitchen/blob/v0.2.39/comfy_kitchen/backends/cuda/ops/cutlass_gemm_int8.cu), [LeanStreamK distribution](https://github.com/Comfy-Org/comfy-kitchen/blob/v0.2.39/comfy_kitchen/backends/cuda/ops/cutlass_gemm_common.cuh). Actual installed SASS and counters, rather than upstream defaults, determine the resource table above.

## Warp stalls: qkv, matching ordinary tiles

NCU `WarpStateStats`: average warp cycles in each state per issued instruction. These are not percentages or directly additive elapsed-time savings.

| State | Strict | CUTLASS |
|---|---:|---:|
| Total cycles / issued instruction | **14.208** | **13.059** |
| math_pipe_throttle | 8.635 | 8.445 |
| wait | 2.555 | 2.227 |
| long_scoreboard | 0.608 | 0.341 |
| not_selected | 0.480 | 0.416 |
| mio_throttle | 0.321 | 0.163 |
| short_scoreboard | 0.140 | 0.079 |
| barrier | 0.171 | 0.251 |
| lg_throttle | 0.119 | 0.094 |
| branch_resolving | 0.066 | 0.022 |
| no_instruction | 0.0253 | 0.0036 |
| drain | 0.0195 | 0.0072 |
| dispatch_stall | 0.0188 | 0.0107 |
| imc_miss | 0.0049 | 0.0084 |
| misc | 0.0059 | <0.0001 |
| membar / sleeping / tex_throttle | 0 | 0 |
| selected | 1 | 1 |

Current Strict executes **71.253M** instructions versus **73.568M**, rather than the old generic entry's nearly equal counts. Strict's main loop is 131 SASS instructions per K=64 iteration versus CUTLASS's 150. Both contain 64 IMMA, 16 LDSM.x4 and six 16-byte async copies. The remaining addressing/control work is 38 versus 57 instructions after excluding the identical memory/compute/synchronization instructions and three predicated-off LDS instructions. Strict is not doing more address arithmetic in this loop.

The denominator matters: qkv's total math-pipe stalled warp cycles are about **615.3M Strict / 621.4M CUTLASS**, despite Strict's larger cycles-per-instruction ratio. Main-loop sampled math+wait totals are **93742 / 93933**, essentially equal. A large math-pipe category does not establish an avoidable Strict-only tensor-pipe stall.

Source-correlated PC samples, interval 1024 cycles (NCU interval setting 5), same settings for both kernels, zero dropped samples/buffer overflow:

| Region | Strict executed instructions | CUTLASS executed instructions | Strict samples, all states | CUTLASS samples, all states |
|---|---:|---:|---:|---:|
| Setup | 4.552M | 3.966M | 9667 | 5677 |
| Main loop | 54.328M | 62.208M | 104433 | 104993 |
| Epilogue | 12.372M | 7.395M | 9251 | 6635 |

Setup's long_scoreboard samples are **4673/2435**, mio_throttle **1249/96**. Epilogue short_scoreboard is **1032/604**, wait **2418/1553**. Main-loop long_scoreboard is **173/279**, barrier **142/109**, short_scoreboard **0/0**. Up shows the same distribution: setup 18513/9181 samples, loop 185528/186784, epilogue 16708/11790. This locates the ordinary-tile excess outside the bulk tensor loop; sampling does not independently measure the time saved by deleting a particular instruction.

## Main loops and specific excess work

Addresses below are relative to kernel entry; absolute PCs change between loads. qkv loop: Strict `+0x2bc0..+0x33e0`, CUTLASS `+0x1e00..+0x2750`.

| Feature | Strict | CUTLASS config 0 |
|---|---|---|
| Shared operand loads | `LDSM.16.M88.4` (`ldmatrix.x4`) | same; not plain LDS |
| Tensor instruction | `IMMA.16832.S8.S8` | same with `.SAT`; model K values cannot overflow INT32 |
| Fragment loading | eight LDSM interleaved with early MMAs in each half | same broad pattern |
| Commit/wait | `LDGDEPBAR`, `DEPBAR.LE SB0, 1`, one block barrier per stage | same depth and wait value |
| Wait placement in actual SASS | after MMA 9, `+0x2d60` | after MMA 21, `+0x23f0` |
| Next async copies | after MMAs 33,34,35,36,38,40; commit after 42 | after MMAs 9,13,14,15,16,17; commit after 18 |
| B fragment traversal | ascending in each row group | alternates direction across row groups |

Strict preloads three stages before entering the loop; CUTLASS preloads two and feeds the pipeline inside the loop. Strict's early wait and later commit are real differences, but the tiny main-loop barrier/scoreboard samples and essentially equal tensor-loop totals do **not** establish them as the measured gap's cause. Do not repeat a speculative wait/stage rewrite on these results.

**Setup:** Strict fetches row/token scale words with 4-byte `cp.async.ca`, one scale per distant packed row/group. For qkv, weight scale addresses are 3456 bytes apart, token scale addresses 384 bytes apart. CUTLASS takes separate contiguous scale arrays. Strict also decodes the packed header and bias format before computation. Top Strict setup PCs include header-dependent `LOP3` at `+0x4e0` (1027 long_scoreboard samples), scale `LDGSTS.E.LTC128B` at `+0x1850` (659 mio_throttle samples), and the bias-dependent `STS` at `+0x23e0` (1472 long_scoreboard samples). These are latency/queue costs of metadata handling, not missing vector matrix loads. qkv L2 read traffic is 1.289 GB versus 1.267 GB; the former unaligned-row 2.14 GB problem has not returned.

**Static 2560 B:** `strict_device_whole_scales` is 640 FP32 entries: 256 row scales (1024 B), 256 biases (1024 B), 128 token scales (512 B). It is epilogue metadata, not a fourth K stage. Bias values are parked here before the loop rather than kept live in registers. The separate three-stage operand buffer uses 73728 dynamic shared bytes. Removing static metadata alone cannot add a resident CTA: the dynamic allocation already excludes a second one.

**Epilogue:** Strict converts/scales groups of 64 accumulators per thread, stages FP32 with scalar `STS`, reads `LDS.128`, then packs BF16 and writes `STG.E.64`. CUTLASS stages groups of 16 **INT32 accumulators**, two per `STS.64`, reads `LDS.128`, and only then converts/scales/packs BF16 for `STG.E.128`. Both stage four bytes per accumulator; CUTLASS is not halving shared bytes by staging BF16. Strict repeats the uniform result-format branch and 64-bit output-address work across unrolled stores. qkv executes 1.106M scalar shared stores versus CUTLASS's 0.553M paired stores, and epilogue instructions are 67% higher. The smaller consumed fragments and vectorized stores are relevant to register lifetime and instruction cost. Changing accumulation precision is not indicated. CUTLASS also contracts scale/bias arithmetic to FFMA; copying that transformation is not an exactness-preserving optimization by assumption.

**Banks:** Source-correlated LDSM wavefronts are exactly ideal for both: **26,818,560 actual / ideal, zero excessive**. Direct shared-load bank-conflict counter is 132619 Strict versus 0 CUTLASS, shared-store 32456 versus 813. These are small compared with tens of millions of shared wavefronts; main-loop short_scoreboard is negligible. Async-copy source wavefront excess is present in both layouts, so it must not be mistaken for a Strict-only ldmatrix bank conflict. No evidence supports changing the matrix-load swizzle as the first fix.

Down's matching ordinary-tile control reinforces the distinction: warmed Strict/CUTLASS config 0 take 1003.0/987.6 us in NCU while reporting **15.052/13.539 cycles per issued instruction**. Their executed counts are 53.921M/59.083M. The large normalized stall ratio coexists with only 1.56% elapsed difference. Default Stream-K takes 887.1 us on this capture.

## B2 and decision

Strict already uses 254 registers/thread. The loop carries 128 INT32 accumulators and operand fragments. The epilogue then reads eight row scales, eight biases and sixteen token scales while accumulator values remain live. B2 must avoid extending gate/residual registers through the K loop or retaining two complete output tiles; load them only as each epilogue fragment is consumed. Measure executed spills, not just allocated local bytes.

Preserve the existing BF16 rounding of the GEMM update before gating, then perform the gate multiply and residual addition in the current FP32 residual precision. Fusion can remove the intermediate buffer without removing its rounding point. CUTLASS's already-installed residual epilogue also explicitly rounds before gating, but its output dtype does not automatically match Strict's FP32 residual. Do not introduce FMA/reassociation or a lower-precision residual as an incidental speed change.

Suggested review choice: tackle metadata/setup and the epilogue's repeated format/address work, paired shared stores and fragment lifetimes first, coordinating the latter with B2; separately evaluate wave-aware tile selection / integer Stream-K reduction for o/down. The former has an **8-10 ms/step fixed-tile ceiling**, the latter **5-6 ms/step demonstrated dispatch headroom** at 1695 MHz. No implementation has been attempted. Full-model paired timing and exact error/PNG checks remain required before claiming any savings. The isolated 15-17 ms total is not a guarantee of beating ComfyUI's 363-366 ms evaluations.

A has not landed in origin/master or origin/main at the final fetch. Its 3080 validation is pending: archive current baseline, pull/rebuild when committed, alternate paired full runs at these same locked clocks, report steady steps and `busy.py` idle, and compare exact errors/PNG SHA-256. Do not mix the old unlocked 374-382 ms baseline with a newly locked candidate.

## Evidence and reproduction

All diagnostic code/reports are outside production at `D:/StrictDumps/gemm-stalls-3080/`. `runner.py` derives from the existing `codex-elementwise/format-diagnostic/benchmark_existing_gemm.py`; `CONFIGS=0,1,13` adds installed fixed-config controls, `SHAPE` selects a projection, `TOKENS` is only for the full-wave control. Input archive: `D:/StrictDumps/codex-elementwise/production-gemm-fast/`.

- Final timings: `config-1695-{1,2,clean}.log`, `telemetry-1695-{1,2}.csv`. Initial/supplementary 1800-MHz-target evidence: `timing.log`, `config-control.log`, `config-repeat-1800.log`, `telemetry-1800.csv`, `o-full-waves.log`, `down-full-waves.log`.
- Final counters: `{qkv,o,up,down}-1695.ncu-rep` (down includes Strict, default CUTLASS, ordinary CUTLASS config 0).
- Per-PC instructions/stalls: matching `*-strict.json`, `*-cutlass.json`, `*-cutlass-config0.json`, and `*-compact.sass`, produced with NCU's bundled `ncu_report` Python API (`analyze.py`). The original `down.ncu-rep` cold-clock duration must not be used.
- Profiler runner: `profile-1695.py`; `--clock-control none --cache-control none`, external clock locks, 12 warmups before each captured launch, `cuProfilerStart/Stop`, sampling interval 1024 cycles, 18 replay passes. Sections: SpeedOfLight, LaunchStats, Occupancy, SchedulerStats, WarpStateStats, InstructionStats, SourceCounters, MemoryWorkloadAnalysis; explicit shared bank-conflict and L2 sector counters. Do not substitute replay duration for graph timing.

```powershell
$env:CONFIGS = '0,1,13'
& D:/ComfyUI_windows_portable/python_embeded/python.exe D:/StrictDumps/gemm-stalls-3080/runner.py D:/StrictDumps/codex-elementwise/production-gemm-fast
& D:/ComfyUI_windows_portable/python_embeded/python.exe D:/StrictDumps/gemm-stalls-3080/analyze.py qkv-1695
```

Run timing only after applying the clock locks. Graphics/memory locks were reset after this session with `nvidia-smi -rgc` / `-rmc`, both exit 0 (`clock-restore.log`); Afterburner settings remain intact.
