# Handoff: Moss elementwise kernels, step wall time below ComfyUI (RTX 3080)

Goal: MossImage at 720x1280, 8 steps, merged LoRA model `moss-image-int8-convrot-dmd-lenovo-finger.safetensors`, faster per step than ComfyUI on the local RTX 3080 (10 GB). Build the ComfyUI node only once Strict is faster. Rules: `AGENTS.md` here and in `../Strict` (TDD, tabs, CRLF, `strict check` 0 issues, no handwritten CUDA without an emitter, commit only your own hunks in `../Strict`, other sessions edit it in parallel).

## Where it stands (2026-10-23, Strict 61506496, StrictComfy 9e9389b)
Per step, from nsys kernel windows (`cats.py` below):

| | Strict | ComfyUI |
|---|---|---|
| GEMM (int8, strict_device_wholes) | 185-188 ms | 164 ms |
| Attention | 114-116 ms | 130 ms |
| Elementwise (generated loops) | 78-79 ms | 29-30 ms |
| Quantize / Hadamard | 22 ms | 34 ms |
| Kernels / wall | 398-405 / 411-423 ms | 358-360 / 370-373 ms |

The gap is 40-50 ms of wall time: elementwise +49, GEMM +22, host gaps ~15 (Strict kernels cover 95-97 % of the step, ComfyUI 97 %); attention and quantize are ahead.

Elementwise per layer call (42 layers, µs, 3740 tokens x 3072, FFN 8192):

| Kernel (MossBlock) | µs | per step | ComfyUI counterpart |
|---|---|---|---|
| Rotated (q and k, head RMS inline, 32 bit out) | 2 x 287 | 23.8 ms | rope_kernel ~9 ms for q+k |
| Activated (Situ, bfloat16 out) | 455 | 18.7 ms | fused / vectorized |
| Gated (residual + update * gate) | 2 x 170 | 14.5 ms | at::native elementwise ~8.6 |
| Modulated (token RMS inline, adaLN, bfloat16 out) | 2 x 167 | 13.9 ms | adaln_kernel ~7 |
| Batch.Columns (v out of qkv) | 110 | 4.7 ms | none (a view) |

ncu (clocks locked): these kernels are instruction bound, 120-330 SASS instructions per number at 72-84 % SM against 36-78 % DRAM throughput. ComfyUI's are bandwidth bound with 16 byte vector loads of 8 bfloat16. Nothing exotic is wrong; the generated code does too much per number:
- one scalar load per list read, each with a bounds check (compare, branch, failure store) and a format branch (`strict_device_floats`: 32, 16 or bfloat16 at run time);
- exact/rational shadow math (the `_e`/`_q` parts) where the exact `sx` kernel can not prove it away;
- grouped calls (`Qwen3Block.HeadScale` inside Rotated/Modulated) go through the generic dot path with int8/float8 branches (see the PTX of MossBlock_Rotated_5_gpu14sx);
- one element per thread, so index math (divisions by width/size) is paid per number.

## Done this round (read the commit messages, they hold the measurements)
- Strict 453d73e8: packed int8 rows start on a 128 byte line (GEMM -15 ms).
- Strict c2577213: `E ^ x` is one exp2.
- Strict 9037087f, 232e77b5: grouped loops run one block (>1024 numbers, 256 lanes) or one warp (<=1024) per group, each lane loops over its numbers and calls the group sum once (`MlirGrouped`: Body, Ending, Remembered); exact `sx` kernels also for grouped and warp summed loops. StrictComfy f80ae71, 8eacc7b: Modulated and Rotated compute their RMS scale inline, `MossBlock.Scales` is gone.
- Strict 8eaa1041: exact kernels assume divisor parameters in [1, 2^31] (`MlirPassed.Divisors/Bounded`, `llvm.intr.assume`; the host only picks the `sx` kernel when they have a magic number, guards mask bit 32 + index) and grouped kernels assume %at < %count < 2^32, so `index / p` and `index % p` share one magic multiply without sign checks or slow path.
- Strict 61506496: `NeuralNetworks/BFloat16.strict` (Float8 moved there too); a GPU loop returning `BFloat16s` stores format 17 (from 2^20 numbers on). StrictComfy 9e9389b: Modulated and Activated return BFloat16s (-5 ms).
- Strict HEAD: a dot whose two reads are the same list from the same start (sum of squares, `Qwen3Block.HeadScale`) emits `strict_device_squares` (`MlirDeviceDot.SquaredEmitted`): one read per number, `ld.v4.b32` when 32 bit and 16 byte aligned. Rotated 351.4 -> 341.0 µs isolated (6 alternating runs), per step below noise: the 128 element group spread over 32 lanes already read 4 numbers per lane, so only the per-read bounds check and format branch went away (the earlier 30 % figure came from a stub that also dropped the reduction, keep these measurements apart).

## Follow-up on this checkout (2026-10-23)
Starting heads: Strict `74e512aa`, StrictComfy `6837de0` (includes h3 commit `c94f93d`). Local device is still the RTX 3080; no 4090 measurements were made. The changes below are working-tree changes, not commits.

- `MossBlock.Rotated` computes token with `((index / size).Floor / heads).Floor`, channel with `index % size`, and head start with `index - within`. `MlirGroups` now recognizes `index - index % size` as a grouped start, including a remainder saved to a local variable. Without that compiler change the rewrite loses grouping and repeats RMS per element: full steps regressed to 1.38 s. One focused compiler regression exercises the saved remainder.
- Checked isolated RoPE medians (six alternating CUDA-graph rounds): f32 inputs 248.3 -> 246.1 us; bf16 inputs 274.2 -> 264.8 us. This is a small isolated gain, **not a demonstrated wall-time improvement**. The second full alternating pair had median steps 1–6 of 433.5 -> 433.0 ms; desktop variation is larger. Both paths give exactly the same logged step errors, latent error 0.32765268 and PSNR 27 dB. Executable size stays 1,201,664 bytes; PTX grows by 1,289 bytes.
- Scalar batching was tried and reverted: a rolled four-element loop, forced unrolling, `noalias` outputs/failure flags, and a separate load phase before computation. All passed format/tail tests, but none made Activated faster. ncu on the same bf16 Activated input: baseline 417.89 us / 17 registers, load-ahead 439.81 us / 34 registers; SM throughput 71.93 -> 79.09 %, DRAM 59.56 -> 56.96 %. More scalar instructions and registers did not fix the bottleneck. LLVM's unroll hint must be on `llvm.cond_br` with `loop_annotation`; this installed cf-to-LLVM conversion leaves `llvm.loop_annotation` as an unused attribute.
- GEMM row-tile grouping 4 -> 8 was also reverted. With CUDA graphs and model-like packed row strides, qkv was 1161.7 -> 1162.6 us and FFN up/gate 2115.1 -> 2110.1 us, within noise. Real-workload ncu of qkv: 1.06 ms, 78.07 % SM, 32.68 % DRAM, 255 registers/thread, 73,728 bytes dynamic + 2,560 bytes static shared memory, one CTA/SM, 16.65 % achieved occupancy. Register pressure **and** shared memory constrain occupancy; lowering only one will not allow a second CTA.
- H3 decoder validation exposed a separate loader blocker: `TensorFile.c` rejected its five-dimensional `post_quant_conv.weight`. Flat tensor reads now accumulate rank/height/width while parsing instead of storing at most four dimensions. Size/offset checks remain, and matrices still require rank 2. H3Decoder's existing 23 failing tests now pass, including the native GPU run. H3Block's native GPU tests also pass.

`benchmark_elementwise.py` loads real emitted PTX and its MLIR signature. It clears output to a sentinel for **each variant**, checks numerical output, failure flag and a guard after the result, then times 50 captured launches with CUDA graphs in six alternating rounds. It covers f32/bf16 Gated, Activated, Columns and RoPE, plus int8 GEMM and partial tiles. Examples (ComfyUI `python_embeded`):

```powershell
python benchmark_elementwise.py D:/StrictDumps/codex-elementwise/baseline D:/StrictDumps/codex-elementwise/grouped-rotated
$env:STRICT_BENCH_GEMM = '1'
python benchmark_elementwise.py baseline-directory candidate-directory
```

`STRICT_BENCH_KERNEL=Rotated` selects one kernel; `STRICT_BENCH_PROFILE=1` only checks one full-size bf16 launch per variant for ncu. GEMM defaults to bf16 output; `STRICT_BENCH_BITS=32` selects f32. Its launch requires 73,728 bytes of dynamic shared memory and `cuFuncSetAttribute` attribute 8, just like `strict_gpu_launch`.

Artifacts and complete logs are in `D:/StrictDumps/codex-elementwise`. Reverted emitter source/patch is in `emitter-experiments`; PTX snapshots are `baseline`, `batched`, `unrolled`, `noalias`, `prefetched`, `group8`, and `grouped-rotated`. **Do not treat `rotated` or `rotated2` as speedups:** they were diagnostic variants that lost grouping. The older `elementwise.py` shares output between variants, so a partly launched candidate can pass by leaving baseline values in unwritten positions. That caused a false 3x result, caught by the new sentinel checks and full run.

Existing implementations reviewed: [PyTorch CUDA loops](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Loops.cuh) and [memory access policies](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/MemoryAccess.cuh) separate loads, compute and stores and use aligned vector loads. Those are useful emitter designs; scalar unrolling alone did not reproduce their benefit here. [NVIDIA's vector-load guidance](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/) also calls out alignment and register pressure.

Verification: changed compiler VM/native tests and related MlirGrouped/MlirDevice/MlirGuards VM tests pass; native GPU tests pass for Batch, Attention, Normalization, Qwen3Block, Gpt2Block, Float8, MossBlock, H3Block and H3Decoder. `strict check` reports 0 issues for Compiler, NeuralNetworks and StrictComfy. The pre-existing uncommitted `BFloat16(0.1).IsNearlyEqual(...)` test still fails (unsupported native Number call, empty VM result); its source was left untouched. Grouping of the new aligned expression is conservatively disabled for ranged loops, whose start may not align to the group size.

## Format specialization: hypothesis resolved (2026-10-23)
Checkout heads at the start of the format experiment: Strict `51c57f87`, StrictComfy `4498aee`; the concurrent Gguf/H3 commits were left intact. The new "How to get unstuck" experiment produced a useful result within the hour. Resolving input formats removes a substantial fraction of RoPE's runtime. True vector elementwise emission is still unimplemented; no new scalar batching experiment was retained.

Diagnostic scripts, modified emitted MLIR, PTX and logs live outside production at `D:/StrictDumps/codex-elementwise/format-diagnostic`. They retain the complete grouped reduction, bounds checks, output coverage and original arithmetic. The isolated mixed-format diagnostic measured 259.7 -> 172.1 us; specializing only the sum-of-squares reader measured 248.9 -> 234.9 us, against 170.5 us for the fully resolved kernel in that alternating round. An in-kernel dispatch recovered some benefit but raised register use to 56, so it was not retained.

Production changes use the existing host selection scheme instead. `MlirFormats` emits three extra copies of each exact kernel: `sxf0` for all f32 free lists, `sxf1` for all bf16, and `sxf2` for a first bf16 free list followed by f32 lists. Only the incoming format stores become constants. The launcher checks the actual staged formats after conversion/quantization, chooses a matching copy and caches its handle. Other signatures, including f16 and Moss Gated's f32/bf16/f32 mix, retain `sx`. Kernel argument ABI, output format, numerical/index guards and reduction geometry remain unchanged. Name/cache arrays grow from 3 to 6 entries for exact kernels.

Checked CUDA-graph comparisons against `grouped-rotated` (six alternating rounds, us):

| Input / kernel | Baseline | Specialized |
|---|---:|---:|
| RoPE f32 | 249.6 | 177.8 |
| RoPE bf16 | 269.7 | 173.1 |
| RoPE model mix bf16/f32/f32 | 266.6 | 178.2 |
| Activated bf16 | 442.8 | 338.1 |
| Gated all bf16 | 164.3 | 135.7 |
| Columns bf16 | 107.4 | 109.6 |

A targeted diagnostic for Gated's actual f32/bf16/f32 mix also stayed flat: 174.9 -> 176.9 us, confirmed at 168.6 -> 169.0 us with numerical, bounds and tail checks (`gated-mixed-confirmed.log`). No fourth format variant was added.

f32 Activated/Gated/Columns were essentially flat. f16 takes the generic fallback and remained within timing variation. A second RoPE run with graph-node assertions measured mixed-format 260.8 -> 179.6 us (`graphs-checked-rotated.log`).

Actual emitted `sxf2` ncu: **251.78 -> 172.42 us**, **107,083,661 -> 70,461,581 executed instructions**, 38 -> 40 registers/thread, zero spills in either kernel. Static PTX instructions in the entry fall from 505 to 283, branches 57 -> 28. The reduction is present; the removed paths are format/packed-reader dispatch. See `production-ncu.log`. `MossImage.exe` grows 1,201,664 -> 1,207,296 bytes; PTX 18,045,472 -> 18,820,124 bytes. This trades code size for speed, not a size optimization.

Full-model accuracy logs are exactly unchanged: embedder/layer errors, step errors, latent error 0.3276526817290422 and PSNR 27 dB. Alternating normal runs (`full-baseline1`, `full-formatted1`, `full-formatted2`, `full-baseline2`) give median steps **2-6: 417 -> 408 ms**. This excludes a real early stall: candidate step 1 was 489/492 ms, whereas baseline was 413/416 ms. Summed logged steps 0-7 were baseline 3,117/3,102 ms and candidate 3,137/3,133 ms. **Do not claim an eight-step wall-time win yet.** A further normal candidate moved the stall to step 0 (685 ms, subsequent steps 404-412 ms). An eager-loading diagnostic removed the step-1 spike, but the node trace later reproduced it and shows only 2.64 ms total in `cuModuleGetFunction`; eager loading alone is not an established explanation or fix.

`formatted-nodes.sqlite` confirms actual host selection: RoPE `sxf2`, Activated `sxf1`, Modulated `sxf0`, Columns `sxf1`, Gated `sx`. One early window has ~90 ms of GPU idle time between Hadamard and GEMM, not slow new kernels. A separately compiled host-MLIR GC probe (no diagnostic code added to production) measured one 96.955 ms full collection near the first sampling step and a 377.422 ms full collection during VAE decoding. That establishes expensive collections, but does not prove the exact profiled idle gap was GC. Keep host/runtime investigation separate from kernel speed claims.

The benchmark now covers f32, bf16 and f16, the mixed RoPE signature, tails and sentinels, and optionally invalid input lengths (`STRICT_BENCH_BOUNDS=1`). It selects available format variants as the host does; `STRICT_BENCH_GENERIC=1` forces the generic copy. Captured graphs must contain at least 50 nodes, so wrong-stream/empty captures fail instead of reporting an apparent speedup. The empty-graph check was watched fail before that assertion was added, then pass by rejecting the capture.

Validation after the change: rebuilt NativeCompiler and strict; fresh VM tests for MlirDevice, MlirGuards, MlirFormats, MlirGrouped and MlirGpu; native MlirDevice/MlirGuards tests; native GPU Batch, Attention, Normalization, Qwen3Block, Gpt2Block, Float8, Products, Layer, Softmax, FeatureMap and MossBlock all pass. Compiler, NeuralNetworks and StrictComfy checks report 0 issues. The existing BFloat16 Number-method failure remains outside this change. Temporary fixture copies were removed.

Existing GEMM implementations were also measured using the installed `comfy_kitchen` CUDA extension, prequantized identical numbers/scales and checked outputs. No new dependency or DLL integration was added. See `existing-gemm-graphs-checked.log`; **ignore `existing-gemm.log` and `existing-gemm-checked.log`**, whose first attempts used a stream outside capture. The corrected runner resolves the current capture stream on every launch and asserts nonempty graphs.

| Shape M x K x N | Strict us | Installed CUTLASS us | cuBLAS + dequant us |
|---|---:|---:|---:|
| 3740 x 3072 x 9216 | 1147.6 | 1049.5 | 1404.5 |
| 3740 x 3072 x 3072 | 422.3 | 359.9 | 518.9 |
| 3740 x 3072 x 16384 | 2068.5 | 1903.6 | 2515.2 |
| 3740 x 8192 x 3072 | 1032.3 | 875.4 | 1097.1 |

These are the confirming alternating run with graph-node assertions. CUTLASS receives contiguous weights, while Strict receives its real padded packed-row layout. The comparison includes scaling/output conversion but excludes quantization/ConvRot in both paths. cuBLAS is not a promising substitute here; CUTLASS remains a useful 8-17% target and potential library integration, with weight-layout/setup costs still to account for.

Further GEMM counters narrow the next hypothesis. Installed CUTLASS uses the same **128 x 256 x 64 tile, 64 x 64 warp tile, 3 stages, 256 threads, 73,728 bytes dynamic shared and one CTA/SM**, with 226 rather than 255 registers and no static shared allocation (Strict has 2,560 bytes). Achieved occupancy is the same ~16.66%. It is faster without a second CTA. In `gemm-paired-ncu.log`, identical qkv numbers execute 73.260M instructions in Strict, 73.122M with no bias, and 73.568M in CUTLASS; cycles between issued instructions are 14.64, 14.62 and 13.17. This points toward issue scheduling/data movement rather than fewer total instructions; it does not establish a particular root cause.

Removing the zero-bias argument was flat on the large qkv and expansion shapes (`bias-gemm.log`), so no production bias special case was added. Matching CUTLASS's identity tile traversal and trying row groups of 2 also gave no useful gain. Explicit register caps of 224/240 were 2-4% slower on large shapes; driver attributes confirm the caps took effect and local allocation grew from 168 to 224/208 bytes per thread. All these candidates passed numerical and partial-tile checks and remain diagnostic only. Logs/source/PTX: `D:/StrictDumps/codex-elementwise/gemm-tuning`.

## Strided attention values (2026-10-23 follow-up)
Scheduled upstream sync: `git pull --no-rebase --no-edit origin master` at 11:42:05 and 12:12:05 UTC (30/60 minutes after the request) both returned **Already up to date**, Strict `c868127c`. Local changes were backed up, stashed and restored without conflicts. The fixes for improvements items 1-3 had not arrived on origin/master at either check. Schedule/results: `D:/StrictDumps/codex-elementwise/upstream_merge_schedule.json`.

`Attention` now accepts optional `valueOffset` and `valueStride` in numbers; zero stride keeps the ordinary packed layout. Moss reads V directly from the interleaved qkv result at offset `2 * width`, stride `3 * width`. This removes `Batch.Columns` without rounding the residual or changing attention math. Host checks require whole, nonnegative, aligned offsets/strides and a complete view within the list and signed 32 bit device indexes; other layouts use the ordinary implementation. The D128 f32-key/bf16-value prompt branch resolves formats before the tiled loop. Keeping mixed formats generic made attention ~900 us slower, so that variant was rejected.

Artifacts: `D:/StrictDumps/codex-elementwise/attention-view`. The production archive is `strided-production`; `benchmark_production.py` checks it against `baseline`, and `production-compact` runs the new ABI with ordinary contiguous values. D128 decode, D64 grouped/multisequence decode, prompt tails and the full 3740-token prompt all matched the original output bit for bit with f32 and bf16 values. Independent PyTorch SDPA checks and output sentinel checks pass. The compiled host guard passes 2,012 boundary/random checks (`check_view_guard.py`). VM/native Attention tests include unaligned fallback and an aligned nonzero-offset view; native GPU MossBlock, MossImage, Qwen3Block and Gpt2Block pass. Compiler, NeuralNetworks and StrictComfy checks report 0 issues.

Fresh nsys traces (`baseline-trace.sqlite`, `strided-trace.sqlite`) show all 378 Moss Columns launches removed: 42.715 ms of copies disappear while attention rises 12.862 ms across the run, about **3.3 ms saved per 42-layer pass**. Isolated graphs agree: full-size attention 2702.6 -> 2732.9 us for f32 V and 2725.8 -> 2757.9 us for bf16 V, against ~110 us for the removed copy. Contiguous attention with the new ABI measured 0.3-1.3% slower across alternating comparisons; a diagnostic extra packed-layout branch did not recover this and remains outside production (`packed-specialized`).

Two full-model pairs, with order reversed in pair 2:

| Run | Steps 0-7, ms | Steps 2-6 mean, ms |
|---|---|---:|
| Strided 1 | 510, 386, 397, 392, 395, 396, 396, 9 | 395.2 |
| Baseline 1 | 425, 488, 397, 399, 407, 393, 396, 7 | 398.4 |
| Baseline 2 | 429, 489, 395, 391, 395, 393, 398, 11 | 394.4 |
| Strided 2 | 532, 391, 399, 391, 395, 394, 390, 17 | 393.8 |

All logged errors were identical: step 0 `0.007934923826421877`, latent `0.3276526817290422`, PSNR 27 dB. The steady improvement is small and noisy; the eight-step total is not a demonstrated win because the early host stall moves between steps. Executable size is 1,203,712 bytes against 1,207,296 for the formatted baseline; PTX grows 18,820,124 -> 18,950,559 bytes.

The separate bf16 Gated/residual-output experiment is rejected: step 0 error rose to 0.00857781, latent to 0.4703, PSNR fell to 23.39 dB (`D:/StrictDumps/codex-elementwise/Gated16`). No residual precision change is retained.

## Vector-load evidence (2026-10-23, diagnostic only)
Actual vector loads now have a checked result, unlike the previous scalar unrolling experiments. A Python diagnostic emitter under `D:/StrictDumps/codex-elementwise` builds MLIR with vector loads/stores and ordinary scalar tails; it does not change Strict's compiler or production kernels. `build_vector_activated.py` emits the same Situ expression for scalar and vector types, using the existing exp2 approximation and bf16 conversion. This is a proof of the codegen opportunity, not a production implementation.

| Real input mix | Scalar graph median, us | Vector graph median, us | Executed instructions | DRAM throughput |
|---|---:|---:|---|---|
| Gated f32/bf16/f32 | 165.2 | 167.1 | 41.649M -> 8.168M | 92.77% -> 91.60% |
| Activated bf16, pair 1 | 336.4 | 265.3 | 133.085M -> 89.521M | 75.93% -> 92.57% |
| Activated bf16, reversed pair | 327.7 | 268.3 | same kernels | same kernels |
| Activated bf16, alignment guard + full bounds checks | 325.0 | 271.1 | added uniform alignment guard | not reprofiled |

Gated is already at the memory bandwidth limit on this mix: five times fewer instructions did not make it faster. Activated improves **17-21%** and approaches that limit. The ncu single-launch times were 324.19 -> 262.21 us, registers 17 -> 29, DRAM bytes 184.36 -> 181.76 MB. Driver attributes confirm zero local bytes per thread for both kernels (`attributes.log`). Emitted PTX has two `ld.global.v4.b16` and a `st.global.v2.b32` per four outputs; the baseline is scalar. Every checked output was bitwise identical, including counts 1, 255, 257, 1023, 1025, full minus 3 and full (30,638,080). Bounds failures and output sentinels pass. Width 7 correctly takes the scalar tail/fallback and matches bit for bit; it is slower (3.0 vs 1.8 us), so production selection must preserve the old kernel for unsuitable/small shapes. The vector path checks width alignment before vector loads. Inputs scaled by 10,000 also match bit for bit and pass the same shortened-list/sentinel checks (`large-values-results.log`); their timings are 535.3 -> 453.4 us, not the normal-input timing quoted above. Width 4096 also matches exactly and improves 163.8 -> 134.5 us (`width4096-results.log`). Full-model performance has not been measured with this diagnostic.

Logs and lowered source: `vector-gated/{results,counters}.log` and `vector-activated/{results,reversed-results,counters,width7-results,width4096-results,full-bounds-results}.log`. The next implementation should recognize eligible contiguous paired-column reads, emit checked four-wide loads/math/stores, and retain the original format/shape fallback. Do not copy these model-specific diagnostic kernels into the runtime or repeat scalar unrolling. Gated is not a useful first performance target after format specialization.

## Next steps, by expected gain
1. **Integrate vector elementwise codegen**: Activated's checked four-wide diagnostic is 17-21% faster (section above). Implement this through the general emitter for eligible contiguous paired-column reads, with shared vector bounds checks, the existing scalar/format fallback, unchanged arithmetic and output coverage. Reuse the diagnostic for emitted-instruction and alternating graph comparisons, then verify full-model accuracy/performance. Gated already reaches ~93% DRAM throughput and did not benefit. Scalar unrolling, load-ahead and noalias alone were flat or slower.
2. **Format coverage**: the common signatures are implemented and measured above. Gated's f32/bf16/f32 specialization measured 168.6 -> 169 us and was not retained. In-kernel dispatch also lost to register pressure. Preserve f16/packed fallbacks. Do not repeat either experiment without new evidence.
3. **Remaining division work**: the committed RoPE rewrite removes the product numerator, preserves grouped RMS and gives only a small isolated gain. The old `floored.patch` scratchpad is absent and was never applied or validated here. Inspect current PTX before assuming another division rewrite will matter.
4. **Rotated as bfloat16**: halves its write and the attention's key reads, but `strict_device_attentions` got slower with bfloat16 queries/keys (3193 -> 3586 µs): ncu it; the fix is probably bf16 mma (m16n8k16 bf16) instead of converting each bf16 pair to f16 in `strict_device_attention_stored` (`Compiler/NativeRuntimeAttention.mlir`), or Rotated returning f16 lists.
5. **Fusions** (need a cross-method view: an inliner for loop methods, `Bytecode/Inliner.strict` only inlines small loop-free ones): Gated into the o/down GEMM epilogue (`strict_device_wholes`), Gated + next Modulated in one grouped kernel, Columns elimination is now implemented by strided attention (above). Gated/GEMM fusion remains open.
6. **GEMM ~20 ms of remaining headroom**: the installed CUTLASS comparison above is 8-17% faster, but uses the same tile, stages and one CTA/SM. Compare inner-loop load/compute scheduling, data layout and epilogue register lifetimes before redesigning for two CTAs. Identity/2/8 tile grouping, forced lower register limits and removal of the zero-bias argument have now been tested without a useful large-shape gain. A CUTLASS library integration remains an option; its standalone ABI and packed-weight layout/setup costs still need implementation and measurement. The old scratchpad DLL prototype (`ext/strict_cuda`) is absent here.
7. **Host gaps ~15 ms per step**: nsys `busy.py` shows 95-97 % busy; look at launches between layers.

## How to build, run, measure
- Rebuild after `../Strict/Compiler` changes: `strict build Compiler/NativeCompiler.strict` twice, delete `Strict.strictbinary`, `strict build Strict.strict` (from `../Strict`). Runtime `.mlir` changes need no rebuild, programs read them at build time. mlir-translate crashes about every tenth build (0xC0000005), just retry. Right after strict.exe is replaced a call can exit 127, retry after a second.
- Native builds launched from StrictComfy currently fail (ggml flags paths, `ggml-cpu/quants.h` not found; the 4090 agent has a fix on branch native-flags-root, not merged yet). Workaround: build from `../Strict`: `strict build ../StrictComfy/MossImage.strict` and run executables with StrictComfy as working directory. A file whose VM tests fail on relative model paths (`MossBlock.strict` reads `tinymoss.safetensors`) skips the native build silently (`NativeGate.Measured`): copy `tiny*.safetensors` into `../Strict` for the build, then delete them.
- Run: `MossImage.exe D:/StrictDumps/mix720 D:/ComfyUI_windows_portable/ComfyUI/models/diffusion_models/moss-image-int8-convrot-dmd-lenovo-finger.safetensors "" D:/ComfyUI_windows_portable/ComfyUI/models/vae/flux2-vae.safetensors` from StrictComfy (~70 s). Latest RTX 3080 formatted/strided runs: steady steps ~390-407 ms, step 0 error 0.00793492, latent 0.32765268, PSNR 27 dB (paired results above). The 8 step trajectory is chaotic to last-bit changes; judge accuracy on the embedder, layer 0/16 and step 0 errors and by looking at the image.
- GPU clocks drift between runs by up to 30 ms per step (desktop apps use the GPU): compare builds with alternating runs. Copy `MossImage.exe` + `MossImage.ptx` into a folder per variant (the exe checks the PTX hash next to it) and alternate them.
- Scratchpad of the previous session (scripts): `C:\Users\Benja\AppData\Local\Temp\claude\C--code-GitHub-strict-lang-StrictComfy\8fa75453-dac0-48f2-a606-c61392e20e36\scratchpad`
  - `prof.ps1 <name> 150`: nsys profile through `moss.bat` (nsys must start a .bat that cds first) and the top kernels.
  - `cats.py <sqlite> <attention kernel substring>`: per step window kernel time by category (`strict_device_attentions` for Strict, `fprop_wmma` for ComfyUI's `prof\comfy720.sqlite`).
  - `busy.py <sqlite>`: GPU busy per step.
  - `kb\`: extract one kernel from MossImage.ptx and launch it on synthetic data (`extract.py`, `bench.py`), for measuring PTX edits.
  - ncu: `C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.0\target\windows-desktop-win7-x64\ncu.exe --kernel-name regex:"MossBlock_" --launch-skip 40 --launch-count 5 ... cmd /c moss.bat` (counters are enabled for all users).
- Tests per change: VM tests of the changed compiler files, native GPU tests (`STRICT_GPU=all`, run from `../Strict`) of Attention, Batch, Normalization, Gpt2Block, Qwen3Block, Products, Layer, Softmax, FeatureMap, BFloat16, Float8 and StrictComfy MossBlock (run `MossBlock.test.exe` from StrictComfy). DeltaChunks has 6 native GPU failures that predate this work.

## Known open bugs
- VM: Number methods on a BFloat16 value give an empty result (`x.IsNearlyEqual(...)` of a BFloat16 or a BFloat16s element); MossBlock is unaffected because its VM lists hold plain numbers.
- A bfloat16 MossText.Activated result going into `down.Sums` gave NaN (method-name experiment, improvements.md item 2); MossText does not use BFloat16s now.

## How to get unstuck
a bit stuck now after a day of work, some new guidelines: Spend at most 60 minutes testing one hypothesis: whether resolving input formats removes a meaningful fraction of Rotated’s runtime. Reuse benchmark_elementwise.py. Preserve reduction work, bounds checks, output coverage and numerical validation. Diagnostic changes stay outside production. Report alternating baseline/candidate timings and inspect emitted instructions to explain any difference. Stop with evidence if inconclusive. No compiler redesign or unrelated fixes.

then continue with a fix, the goal is to get the Elementwise work reduced so it can run faster, not to do some microoptimization on one instruction or loop, it seems to be doing too much work that is not necessary as we can see from the comfyui implementation, it can run much faster on the same hardware. feel free to look for other implementations if this goes nowhere. then continue with general optimizations.