# Handoff: Moss elementwise kernels, step wall time below ComfyUI (RTX 3080)

Goal: MossImage at 720x1280, 8 steps, merged LoRA model `moss-image-int8-convrot-dmd-lenovo-finger.safetensors`, faster per step than ComfyUI on the local RTX 3080 (10 GB). Build the ComfyUI node only once Strict is faster. Rules: `AGENTS.md` here and in `../Strict` (TDD, tabs, CRLF, `strict check` 0 issues, no handwritten CUDA without an emitter, commit only your own hunks in `../Strict`, other sessions edit it in parallel).

## Current handoff: resident timing and phase-boundary memory (2026-10-24)

The user confirmed ComfyUI's 6-6.5 s is at 720x1280 with the model already loaded. The earlier fresh-process 9.7 s comparison was inappropriate for that case. MossGenerate now retains both model and decoder for optional resident repetitions; compare_moss.py exposes --resident-runs, records every sampling/full-generation timing, and labels total-process time separately. Default remains one image per process.

Measured production warm eight-step runs: 4.098-4.179 s including decode, PNG saving and cleanup between passes. Sampling is 3.155-3.171 s. Matched headless ComfyUI is 3.511-3.516 s total, sampling 3.159-3.164 s. Both use saved conditioning/noise and the same merged DMD, at locked 1695/9506 MHz. The remaining ~0.6 s backend gap is mainly VAE decode (Strict ~0.63 s / Comfy ~0.27) and PNG saving (~0.28 / ~0.08). A fresh Strict process is still 9.057 s. Do not imply a cold-start or matched-backend win.

The concrete memory issue: after the VAE, free pooled buffers made Moss re-upload about 2 GB of weights per steady evaluation. Explicit GpuMemory.Released between resident generations frees unused large pool blocks and preserves live cached tensor copies; steady uploads drop to 2048 bytes. The first evaluation still reloads some weights. The global H3 allocation/eviction order is unchanged. Global pool-first and spilled-row free-space alternatives remain rejected diagnostics. The new runtime hook has a native cache/lifetime regression and is documented in ../Strict/NeuralNetworks/README.md.

PNG compression alone cost about 0.93 s. ImageSaver now supports compressed=false via existing Deflate.Stored; its global default stays true. MossGenerate selects stored PNGs by default, saving about 0.65 s with identical decoded pixels and larger files (3.69 MB versus 2.99 MB on the portrait). --compressed restores compression.

Next: close the matched ~0.6 s gap in VAE decode and image encoding, then cold model loading (~3 s versus ComfyUI's lazy ~0.11 s construction, with upload work counted during sampling). Keep resident reuse when integrating the ComfyUI node. Grouped Modulated/Rotated work and the index miscompile remain open; no GEMM emitter changes were made. Leave H3 copy-stream work to the 4090 agent.

Evidence and tests: D:/StrictDumps/moss-end-to-end; Performance.md's end-to-end correction has stage timings and before/after counters. Public harness, VM/native/GPU checks, decoded-pixel checks and compiler fixpoint pass. Benchmark clock limits are restored; Afterburner settings were untouched.

## Current handoff: sampling presets and high-level speed (2026-10-24)

Completed the Spectrum audit and 36-image comparison requested by the user. See [Generation.md](Generation.md) for the runnable harness and settings, [Performance.md](Performance.md) for full timings, and [Spectrum.md](Spectrum.md) for upstream coverage. Local gallery: D:/StrictDumps/moss-speed-comparison/index.html. Three prompts, two seeds and six variants all completed; image size/link checks pass. Current compiler is Strict 7e61d34e including the H3 fixed-budget allocator.

Median sampling times: base25 20.423 s, Spectrum25 13.448 s, turbo8 4.413 s, Spectrum8 4.147 s, akhaliq4 2.892 s, darask0 rapid4 2.922 s. These include initial uploads/text projection and exclude model loading/decode/save. Whole-process medians are 25.75 / 18.37 / 9.70 / 9.58 / 8.18 / 8.29 s. All use int8 Convrot and bf16 residual/QK; the four-step speed variants merge/requantize their LoRAs. Outputs are expected to differ; no equality requirement is imposed across these presets.

The 25-step comparison is explicitly ordinary Euler CFG 3 with zero conditioning, not the release's negative-block skip or ER-SDE preset. darask0 needs its own data-space loop and replacement time weights; akhaliq's current root LoRA uses stock Euler. Both are implemented. Unmerged akhaliq is much slower (48 s sampling); unmerged darask0 timed out after 240 s, so neither is the speed default. Both int8 merges matched 252 projections. Pinned sources are in Generation.md.

A writable per-process CUDA cache under the comparison root eliminates repeated ~28-second setup compilation after warmup. Do not mix uncached runs with the reported matrix. CLI progress timestamps mark host submission, not individual completed GPU evaluations. Benchmark clock limits have been reset; Afterburner settings were untouched. All 476 high-load samples held 1695/9506 MHz, without thermal slowdown.

H3 fixed-budget accounting is already merged and tested here. Two alternating pairs averaged 395.50 versus 399.92 ms steady evaluations; noisy/modest, not a 20% Moss improvement. No copy-stream prefetch commit was available at the final fetch; leave that work to the 4090 agent.

Next priorities:
1. Let the user review the gallery and extend comparison.json with preferred prompts/seeds/LoRA stacks. Keep four-step models as comparison options until quality is accepted. Do not reject faster methods merely because their images differ.
2. Improve end-to-end cost: model load is around 2.6 s and decode/save around 2 s per fresh process. Profile reusable model/decoder execution and phase transitions; the generation runner already removes reference-comparison overhead.
3. Spectrum's predictor is implemented, but adaptive p95/p99 drift control, expanding-window scheduling, detail/repair guards and ER-SDE solver integration are not. Implement/measure these if pursuing the upstream 2-2.5x claims; current periodic 25-step preset is about 1.52x faster, eight-step about 1.06x.
4. Continue grouped Modulated/Rotated vectorization and the full-size index miscompile below. Keep integer addressing precise. B1/B3 remain rejected on 3080 and the GEMM emitter remains unchanged.
5. If dynamic unmerged LoRA use matters, profile low-rank loading/products and memory pressure. The existing path loses the few-step speed benefit here; do not silently claim upstream fp32-LoRA fidelity from the int8 merge.

Validation: VM/native/forced-GPU Guided and channel-major Q mixing checks pass, StrictComfy check reports 0 issues, both compiler executables rebuilt with the merged runtime, native tiny CFG/Spectrum probe passed, and all 36 full-size generation cases completed. Diagnostics, failed trials and a tiny probe are retained outside production under D:/StrictDumps/moss-speed-comparison.

## RTX 3080 bf16 residual and QK (2026-10-24, validated)

Precision policy is in ../Strict/Agents.md: prefer supported int8 Convrot projections, bf16 activation/residual/attention storage, and f32 where precision or measured performance requires it, including H3 audio VAE. Keep necessary precise accumulators and exact integer address/index arithmetic. Do not lower the global bf16 list threshold: that regressed VAE convolutions.

Implemented: native bf16 QK prompt attention for 128-wide heads on Ampere, bf16 Gated/Rotated outputs, explicit Gated width, and four-wide flat loops accepting any number of safe independent linear reads. A noinline attention specialization prevents both predicated bf16 and fp16 MMA paths from being issued; isolated native bf16 attention is 3.009 ms against f32 3.038 ms. Grouped Modulated/Rotated vectorization remains open. GEMM emitter is unchanged.

Three alternating pairs at locked 1695 MHz core / 9506 MHz actual CUDA memory: f32 405.50 / 406.00 / 406.17 ms, bf16 394.67 / 395.00 / 396.00 ms (steps 1-6). Means: 405.89 -> 395.22 ms, saving 10.67 ms (2.63%). Fresh same-clock ComfyUI is 402.43 / 402.56 ms: Strict's steady evaluation is 7.28 ms (1.81%) faster. These clocks differ from historical unlocked 363-366 ms measurements.

Quality is a deliberate tradeoff: PSNR against ComfyUI 28.74 -> 23.30 dB; latent error 0.2691597633635198 -> 0.4578933680652566. F32 PNG SHA256 is 50ab7c7f2c0798e57ac40097061c9d778a6401dfe7b101d82f7dea841eda4913; bf16 is 4aaf91a74757676d7f0a6dafabad43a58e94ffff4c13cc500fea78f44055d972, identical across repeated candidate runs. QK-only bf16 gave 27.08 dB. Images were visually checked on this fixture; broader quality is not established.

Important limitation: the entire diagnostic harness is slower, 24.3-25.0 s bf16 against 16.0-16.5 s f32. Its per-layer CPU comparisons and phase transitions remain unresolved; this is a steady denoising win, not an end-to-end wall-time win. Peak VRAM is about 9604-9606 MiB bf16 versus 8658-8788 MiB f32. Unused large pool blocks are now trimmed under activation allocation pressure, without changing weight-cache policy or global reserve. Eager lending, finer size classes, a global 1 GiB reserve and comparison materialization were rejected; patches and logs remain outside production.

nsys steady windows: f32 410.37 ms / bf16 399.19 ms, including profiler overhead; busy.py target idle 7.37 / 7.75 ms. Gated 14.17 -> 8.27 ms per step, Modulated 12.95 -> 10.89, Rotated 14.19 -> 14.03. Steady copies remain 2048 bytes HtoD and 1843200 bytes DtoH for the harness latent check.

Next, in order:
1. Vectorize grouped Modulated and Rotated through the existing emitter: four contiguous outputs per lane, reuse group reductions and hoist provable bounds checks. Preserve exact integer indices and reduction order. Require raw formats, invalid lengths, tails and full-model validation.
2. Fix the full-size Rotated miscompile recorded in ../Strict/improvements.md: inlining half = size / 2 corrupts 24.3% of raw outputs despite passing tiny tests. Keep the original expression until the cause is isolated; restoring it restores the exact bf16 hash. Reproducer: model-pool and rotated-inline-check.log.
3. Compare ComfyUI gate/modulation rounding independently; its gate is bf16 while Strict's small modulation values remain f32. Preserve normalization/softmax precision and avoid global threshold changes.
4. Isolate the diagnostic comparison/phase-transition wall-time regression. The Readable materialization experiment changed error measurements and did not provide a convincing gain; it is reverted.
5. Have the 4090 agent validate shared runtime changes on H3 before extrapolating these results. Its H3 optimization work remains separate. B1/B3 stay rejected on 3080; do not rebuild them. C has no accepted GEMM emitter change; coordinate any future epilogue work with B2.

Compiler changes are pushed as Strict c1829a08. Final restored production build reproduces the exact bf16 errors/PNG (393.33 ms steady); its new artifact location incurred cold initialization/JIT, so its 73.88 s process time is excluded from warmed paired timing. Benchmark clock limits were reset after timing; Afterburner settings were untouched.

Validation: focused VM/native/forced-GPU checks, large-range bf16 attention, cache trim/lifetime, residency and H3 Hadamard regressions pass; compiler stage 2/3 MLIR fixpoint is identical. Diagnostics: D:/StrictDumps/bf16-attention, paired artifacts model-trim and model-trim-base, baseline worktree ../StrictComfyPrecisionBase at 502da91. Server C:/ is V:/ here; X:/ is server C:/code/. Older A-pending/C-awaiting-review sections below are historical.

## 3080 results for A, B1, C (2026-10-24, locked 1695/9756)
Tables in [Performance.md](Performance.md), section "RTX 3080: A confirmed, B1 rejected, C inconclusive".
- **A works on the 3080**: idle per step 12.3 -> 3.4 ms, steps 1-6 about 5-10 ms faster (415.9 -> 408.3 ms), identical errors and PNG (`50ab7c7f...`). ComfyUI at the same locks: 403.4 ms, Strict is 3-7 ms behind.
- **B1 rejected on the 3080**: fused 632.8 vs separate 587.5 us per layer. B3 on the 3080 not measured yet.
- **C**: no GEMM change wins. Small tile checkpoint only on `codex/gemm-small-tile-3080`. Real stall/SASS analysis of qkv still open, see [GemmStalls3080.md](GemmStalls3080.md); BF16-output entry ("brain") measured against baseline and CUTLASS on all four shapes in one alternating run: within noise of baseline, rejected.
- Open: 3.4 ms idle left. The ncu/SASS analysis of qkv is in GemmStalls3080.md (commit 22a1bf0); it does not name a single cause, see its qkv/up scale/bias setup and epilogue finding.

## B3 diagnostic, and both fusions to measure on the 3080 (2026-10-24)
B3 fuses Gated into the next Modulated. The prototype is a copy of the generated Modulated kernel (`sxf0`, one 256-lane block per token row):
- its `HeadScale` sums the squares of the Gated residual computed in place, `input + update * modulation`, with the same loop, contract flags and lane reduction as `strict_device_squares`;
- its numbers use the same Gated value, and it writes the residual once.

It is bit-identical: 0 residual and 0 normed bits differ (3631 tokens x 3072, the in-layer pair, Gated part 2, Modulated part 3).

On the 4090 at 2400/10251 it saves only 0.65 ms per step for one pair per layer: 161.5 -> 146.0 us per pair, about 1.3 ms with both pairs. That is below noise, so it is not built for the 4090. The fused kernel already moves ~134 MB in 146 us, near DRAM peak, but the separate path loses less than estimated: the 4090's 72 MB L2 already serves much of Modulated's re-read of the residual.

The RTX 3080 has a 5 MB L2, so every re-read of an intermediate goes to DRAM (760 GB/s), and both fusions may be worth far more there. By bytes, B3 could save about 5 ms per step with both pairs, and B1 up to about 6-7 ms, unless their instructions dominate.

**For the 3080 agent:** measure before anyone builds them:

```powershell
& D:/ComfyUI_windows_portable/python_embeded/python.exe benchmark_fusion.py <folder with your MossImage.kernels.mlir> 3740
```

Run it with the clocks locked, from a current MossImage build. It writes `<folder>/fusion`, and the first PTX load JITs for minutes. It prints, for B1 and B3, the bytes that differ (must be 0) and separate against fused us, with ms per step for one pair per layer.

The general implementation would build an element function from the producer's loop body and defer the producer's launch until its list is read; it is about a day of compiler and runtime work. Worth it only if the 3080 shows at least 2-3 ms per step per fusion. B2 (Gated in the o/down GEMM epilogue) stays with the 3080 agent after C.

## B1 diagnostic: fusing Activated into Hadamard/quantization is not worth building on the 4090 (2026-10-24)
The fused kernel was hand-assembled outside production in `C:/StrictDumps/exp/b1` (`diag.kernels.mlir`, `fused.py`). It is r2's generated module plus a copy of `strict_device_hadamard_tokens_run` whose eight source reads compute situ per element: the same `MossBlock_Situ_2_devices`, bf16 reads of `expanded`, then the same `truncf` to bf16 as Activated's store.

It is bit-identical to Activated + `strict_device_hadamard_tokens`: 0 of 33.5 MB of int8 numbers and scales differ, 3631 tokens x 8192.

Locked 2400/10251, CUDA graphs of 20 layers, 8 alternating rounds:

| | Separate | Fused | Saving per step |
|---|---:|---:|---:|
| Element index by 64-bit division | 289.6 us | 255.5 us | 1.4 ms |
| Best case, shift and mask | 287.9 us | 236.2 us | 2.2 ms |

The fused kernel moves ~138 MB at about 585 GB/s, against Activated's 945 GB/s. The Hadamard kernel's shape (1024-thread blocks, shuffles, eight situ evaluations per thread) limits it. With the emitter's magic-number division the gain would be about 1.5-2 ms per step, at the noise floor. The general version (element functions, a generated dispatcher, deferred producer launches in the runtime) is about a day of compiler and runtime work, so it is not built.

On the 3080 the bytes count more (760 GB/s, 5 MB L2), but Activated is instruction bound there and fusion keeps its instructions. Measure it there with `benchmark_fusion.py` (below) before deciding.

## A done on the 4090, ready to commit (2026-10-24)
Step-boundary host round trips are gone. Outputs are identical: step 0 error 0.006325429288679978, latent 0.2228712467784174, PSNR 32.36 dB, PNG SHA-256 prefix `a439d0b78f51c944` in every run.

Measured on Strict 4d1cafc9 + this change, clocks locked at 2400/10251 MHz:

| | Baseline (StrictComfy HEAD, same compiler) | Candidate |
|---|---:|---:|
| Steps 1-6, three alternating pairs (model changes only) | 179.3 / 182.2 / 176.8 ms | 165.8 / 164.3 / 167.7 ms |
| Steps 1-6, three alternating pairs (full change) | 187.0 / 182.7 / 187.8 ms | 169.8 / 161.7 / 163.5 ms |
| GPU idle per step (nsys, `idle.py`) | 17-33 ms, mean 23.2 | 3.0-7.3 ms, mean 5.0 |
| Uploads per step | 11.2 MB | 0 |
| Graph re-instantiation idle per step | up to 7.5 ms | 0 |

The only sync left in a step is the harness's own latent download for its error log (1.8 MB, 3-5 ms). Production runs would not have it.

The general rule behind the round trips: a loop with no 32-bit-readable input list runs its 64-bit ("double") kernel, and a double kernel's result always went to the host after its failure flag was checked. A 64-bit Power or Logarithm doesn't exist on the GPU and fails on every launch, so the CPU runs the loop instead. `Sinusoid` (time embedding, Logarithm/Exponential) and `MossBlock.Rotations` (`10000 ^ x`) failed on the GPU every step. Their results and flags came back, the CPU recomputed them, and they were uploaded again. `Euler.Sigmas` (9 numbers) launched and synced every step.

The change:
- **StrictComfy**:
  - `Euler.Stepped(latent, output, sigmas, step)` takes the per-run sigmas.
  - `Moss.Rotation(latent, text, width)` builds the rotation table once per run; `Output`/`Features` take it.
  - `MossImage`, `MossSpectrum`/`MossSpectrumImage` and `MossLora` are updated.
  - `H3Video` calls `Euler.Stepped` and still recomputes its 9 sigmas per step, now on the CPU.
- **Strict, emitter** (new `Compiler/MlirDoubles.strict`):
  - `Failing`: a loop whose double kernel needs a 64-bit Power or Logarithm (also in callees, depth 3) gets no double kernel, so the CPU runs it without a failed launch, sync or download.
  - `Kept`: a collecting loop that reads no list and has no calls or Power passes result format 64, so its doubles stay on the GPU.
  - Both use existing launch parameters, so old binaries keep working.
- **Strict, runtime** (`NativeRuntime.mlir`):
  - a loop of fewer than 256 numbers reading no list stays on the CPU, also with `STRICT_GPU=all`;
  - format 64 results stay resident; the CPU downloads them only when it reads them (new 64-bit unpack);
  - a 32-bit kernel reading a big resident double list gets a device-side f32 copy (`strict_device_singles`), rounded exactly like the existing host copy;
  - `STRICT_GPU_DOWNLOADS=1` prints every list the GPU sends to the host.
- **Regression check:** `Compiler/check_gpu_residency.py` builds and runs a small program. It fails if a numbers-only table or a Power loop round-trips; it also checks a 32-bit consumer of a resident table and a host read of one.

Validation:
- VM tests of every changed file;
- `strict check` 0 issues for Compiler and StrictComfy;
- strict rebuilt, fixpoint identical (stage 2 = stage 3 `Strict.mlir`);
- native GPU tests of Batch, Normalization, Gpt2Block, Qwen3Block, Softmax, Layer, Attention, Products, FeatureMap, Float8 and MossBlock. BFloat16 fails only on the known VM bug below, unrelated. One Attention build hit the sporadic mlir-opt crash and passed on retry.

**For the 3080 agent, once A is pushed:**
1. Archive the current baseline build.
2. Pull and rebuild.
3. Alternate paired full runs at 1695/9756 MHz.
4. Report steady steps and `busy.py` idle, and compare errors and PNG SHA-256.

The removed costs are host-side, so expect roughly the same absolute saving there (about 15-20 ms per step).

Fixed since: a `Bytes` result computed on the GPU was downloaded (`16000 numbers in 9 bits` in the check) before a GPU loop read it. The method reading it hoisted its byte pointer for the CPU loop before deciding to run on the GPU, and that hoist downloaded the list. The hoist now leaves bytes that are only on the GPU there (`strict_list_host_bytes`). It's not on the Moss path.

**B1 estimate (next):**
- **Cost today, from the single-pass counters:** Activated is 174 us per layer at 94 % of DRAM peak; the Hadamard/quantization after it is 107 us.
- **Fused cost:** it would read `expanded` (bf16, ~110 MB from DRAM) instead of `activated`: about 153 us at Activated's 945 GB/s, 216 us at the Hadamard kernel's current 640 GB/s.
- **Saving:** **2.7-5.4 ms per step on the 4090**, more on the 3080, where Activated is instruction bound (455 us per call).

Planned through the general emitter, with unchanged rounding:
1. An elementwise loop also gets an element device function. It returns element i rounded to the loop's stored format (bf16 for `BFloat16s`), and the loop's own kernel calls the same function.
2. The emitter instantiates the runtime's `hadamard_tokens` kernel with its eight source reads replaced by calls to that function.
3. The runtime launches that variant when the list's next reader is the lazy rotation and quantization. It materializes the list first for any other reader, for example `MossLora.Applied` with LoRAs.

## RTX 4090 baseline and plan (2026-10-24)
Full tables, method and scripts: [Performance.md](Performance.md), section "Locked-clock baseline". Strict master e1cb907b (+ build fix 23e61750), clocks locked at 2400/10251 MHz for every timing run, same fixture/noise/conditioning as ComfyUI. The lock is for A/B timing only, production runs stay unlocked. 2400 is the highest lock tested that holds steadily: locked at 2550 the clock reads 2550 but bf16 GEMM throughput swings between 139 and 160 TFLOPS, against 151-153 at 2400 and 163-166 unlocked. Memory is not slowed by the lock: under CUDA the driver uses the P2 point (10451 MHz effective with +200) whether locked or not (`-lmc 10501` changes nothing); 10701 MHz for CUDA would need NVIDIA Profile Inspector's "CUDA - Force P2 State" off.

GPU: after the user's repaste and new pads (overclock +100 MHz core, +200 MHz memory), 270 s of bf16 GEMM load gave 166 -> 160 TFLOPS. Hotspot reached 100-101 °C at a core of 76 °C, results were bitwise stable, and the only limit was the 450 W power cap. Before the repaste the hotspot hit 108-112 °C and throughput fell 190 -> 145 TFLOPS. The driver counts 0 µs of thermal slowdown since. Under CUDA load the memory runs at the P2 point (10451 MHz effective).

| Per step, ms | Strict | ComfyUI |
|---|---:|---:|
| int8 GEMM | 68.9 | 68.3 |
| Attention | 49.2 | 55.3 |
| Elementwise | 29.1 | 11.1 |
| Quantize / Hadamard | 9.1 | 14.5 |
| Kernel sum | 156.4 | 149.2 |
| Idle | ~21 | ~75-116 |
| Wall, plain runs | 177 (3 runs: 178.7 / 176.8 / 176.8) | 224-265 |

What differs from the 3080 picture:
- The elementwise kernels are DRAM bound here (Activated 94 %, Gated 91 %, Modulated 72-82 %, Rotated 78-80 % of the 1003 GB/s peak).
- Gated is not L2 bound (its working set is 111 MB).
- The in-model GEMM equals ComfyUI's; per shape CUTLASS is 3.5-7 % faster, at most 3 ms per step.
- Idle is 13 % of a step, all at step boundaries:
  - sigma sync, 6.5 ms;
  - rotation table recomputed and copied to the host and back, 10.5 ms;
  - time embedding readback, 2.0 ms;
  - the harness's latent check, 2.3 ms.

Ranked (expected ms per step on the 4090, confidence, work):
1. **A, host round trips at step boundaries: 18-21 ms**, high confidence, small to medium work. Host sigma schedule, no time embedding readback, rotation table once per run and GPU resident. Plus a runtime rule (with a regression test) so lists only GPU kernels use never go to the host and back, and fewer graph re-instantiations after syncs. A whole-step CUDA graph is not needed.
2. **B1, Activated (situ) into Hadamard + quantization: 4.8-5.9 ms**, medium-high confidence, large work. It goes through the general emitter: a generated element function becomes the runtime kernel's read and rounds to bf16 like the stored list.
3. **B3, Gated + next Modulated: 3.2-4.2 ms** (in-layer pair alone 1.6-2.1 ms, below noise), medium confidence.
4. **B2, Gated into the GEMM epilogue: 2.1-2.9 ms by bytes**, low-medium confidence, competes with B3.
5. **C, GEMM stall gap against CUTLASS: ≤ 3 ms here.** Assigned to the RTX 3080 agent, where the gap is 8-17 %.

Decision: A then B1 on the 4090. **For the 3080 agent: C** (ncu warp stall reasons and side-by-side SASS of both main loops, name the difference before changing code); when A lands, repeat one paired run there.

### C analysis completed on 3080, 2026-10-24

[GemmStalls3080.md](GemmStalls3080.md) records all four matched-input graph timings, NCU stall breakdowns, per-PC SASS and installed fixed-config controls. Final clocks 1695/9506 MHz, 156 high-load telemetry samples stable, no thermal slowdown. Isolated total gap is 15-17 ms per 42-layer step at this lock. About 5-6 ms is tile/work distribution: CUTLASS uses 128x128/4 stages for o and Stream-K for down (340 blocks / five full waves versus Strict's 360 / partial sixth wave). Ordinary matching-tile down is within about 2% in NCU. The remaining 8-10 ms headroom is principally qkv/up scale/bias setup and epilogue work, not an established inner-loop scheduling deficit. Both use ldmatrix.x4 with ideal source-correlated wavefront counts; qkv main-loop samples are essentially equal. Current Fast has fewer issued instructions, so the 14.21 vs 13.06 cycles/instruction comparison needs its denominator.

No production changes. Stop for user review before changing GEMM. B2 must account for 254 registers/thread, consume small epilogue fragments and preserve BF16 update rounding plus FP32 residual precision. A is not yet on origin/master or origin/main; locked paired full-model/idle/error/PNG validation remains pending. Plan.md now explicitly records this shared split above its historical phase lists.

Fast division (f32 `afn` divides plus exact integer reduction of fraction positions, so indices never depend on float rounding) is parked uncommitted in `../StrictVector`. It was flat at full clocks. At 1350 MHz it gave Rotated −5 % and Modulated −1 %, but f32/f16 Rotated was 7-10 % slower. Grouped batching stays a patch (`C:/StrictDumps/exp/grouped-batching.patch`): its register use (80-126) made Rotated slower.

## Current checkpoint, 2026-10-24

The debugging limit is now 20 edits in both repositories. The Spectrum adapter is merged into main, opt-in, with exact inference still the default. Strict's safe cold-cache eviction fixes the standard split/ThinLTO build's VAE stall: GPU argument slots remain protected through nested dispatch; cold dedicated host-backed copies can be evicted after queued work finishes; device-only activations and heap roots stay valid. Core implementation was checkpointed in Strict e1cb907b. Forcing only the outer model object dead did not fix the issue; no model-specific noinline or linker workaround was adopted.

Both NativeCompiler and strict were rebuilt from current source. Standard `strict build ../StrictComfySpectrumWork/MossSpectrumImage.strict` completes full-size off/on runs. Off PNG is exactly `986b307e47af99d345eb09f44d6dce168d6fb1cec5e13d85dd61172067a3502e`; on PNG is exactly the prior live-prediction result `031781db011b2613a310688f9e79992ad6dd1387ae031c9acc081095c8bd8d25`. On mode skips step 5, forecast 9 ms in this run, same previously measured latent/PSNR. Off/on decoder times 4864/1416 ms still exceed standalone 463 ms; GC/cache transition overhead remains a performance issue. Full command wall times include diagnostics/loading: 54.12/48.11 s, not sampling latency. Evidence: `D:/StrictDumps/spectrum-moss/verified-{off,on}.{log,json,png}` and telemetry CSVs.

Regression check `Compiler/check_gpu_cache.py` fails without active-input protection and passes with nested/outer/restored scopes, all-inputs-pinned and pooled-copy exclusions. VM/native GPU checks of Batch, Products, Convolved, Attention, GroupNorm, FeatureMap, Gpt2Block and Qwen3Block pass; Compiler and adapter source checks report zero. See Spectrum.md for schedule, CLI, quality limitations and the original speed comparison. The old blocked-build and ten-edit-stop notes below are historical, superseded by this checkpoint.


## Driver/overclock check and acceleration research (2026-10-23)
Driver 621.01 with user settings core +82 MHz, memory +255 MHz, voltage slider +8%, power 106%, temp target 86 C: two runs of the identical validated Moss executable averaged 379.6/379.0 ms for steps 2-6, with identical errors and PNG hashes. No sampled thermal throttle; peak core 66.12 C. The approximately 0.55% timing difference from the earlier settings is within observed variation; driver and overclock effects are not separated. Evidence: `D:/StrictDumps/moss-driver621-oc/summary.json`. [InferenceAcceleration.md](InferenceAcceleration.md) assesses Spectrum-MOSS, concrete Strict integration/quality tests, existing Flash/MTP/MoE results, and Next-Concept Modeling. No approximate inference path has been enabled.

## Latest completed change: cached uniform-scale GEMM (2026-10-23)
Strict `43df39f6` implements a separate f32-input GEMM entry that removes the mixed-scale fallback only after the existing row-upload scan has proven every row uniform. Cache kind 4 records that proof and remains compatible with ordinary row lookups (kind 2). Cold, uncached, small and mixed-scale matrices keep the original entry. Float64 entry remains unchanged. Kernel math, tiling, MMA order and quantization are unchanged; no dependency was added. Diagnostic reordering of the second operand load was slower and rejected.

RTX 3080 CUDA-graph BF16 output timings, old -> fast, microseconds: 3740x3072x9216 **1111.8 -> 1057.1**, x3072 **400.7 -> 391.2**, x16384 **2023.0 -> 1956.9**, 3740x8192x3072 **1005.8 -> 961.1**. Float32 output also passes and improves. Registers/thread 255 -> 254; local storage/thread **168 -> 24 bytes**. Model PTX grows 19,396,501 -> 20,502,879 bytes because both compatible and specialized entries remain available.

| Full run | Steps 2-6 average, ms | Eight-step total, ms |
|---|---:|---:|
| GEMM fast 1 | 380.6 | 2777 |
| Vector-only 1 | 388.6 | 2855 |
| Vector-only 2 (order reversed) | 386.4 | 2845 |
| GEMM fast 2 | 382.2 | 2802 |

All four logs have identical errors and PNG SHA-256 `986b307e47af99d345eb09f44d6dce168d6fb1cec5e13d85dd61172067a3502e`. Archives `production-linear` and `production-gemm-fast`; evidence `D:/StrictDumps/codex-elementwise/gemm-scheduling/{full-validation.json,guarded-results.log,f32-results.log}`. The 370 ms goal remains open, with approximately 381 ms steady steps now measured on this 3080. No fresh 4090 results.

Runnable checks: from Strict, `D:/ComfyUI_windows_portable/python_embeded/python.exe Compiler/check_gpu_cache.py` compiles the real cache lookup and verifies uniform/ordinary/quantized cache compatibility. From StrictComfy, set `STRICT_BENCH_GEMM=1`, `STRICT_BENCH_GEMM_FAST=1`, `STRICT_BENCH_WHOLE_GUARD=1`, then run `benchmark_elementwise.py` against the two archives. The legacy entry must accept per-block scales; the specialized entry must reject a missing uniform header flag. `STRICT_BENCH_BITS=32` repeats float32 output checks. Do not globally remove the legacy fallback: first-row host classification alone does not prove later rows uniform.

Validation: shared NativeCompiler and strict rebuilt, native GPU Batch/Qwen3Block/Gpt2Block passed, native cache contract passed, Compiler and NeuralNetworks checks report 0 issues. Full native MossImage runs exercise actual dispatch: nsys recorded 168 cold `strict_device_wholes` calls followed by 1344 `strict_device_wholes_fast` calls (`gemm-scheduling/fast-trace.sqlite`). Float32/BF16 numerical, layout-guard and sentinel checks passed.

Next useful elementwise target: grouped RMS/RoPE and modulation, whose reductions currently exclude the generic four-output emitter. Gated already reaches about 93% DRAM throughput; vectorization cut its instruction count but did not improve time. Keep actual generated graph timing, numeric/tail checks and full-model paired runs as the acceptance test. H3 can pull the shared compiler vector change and reuse it without model-specific kernels.

## Completed change: generated vector elementwise (2026-10-23)
Strict `56e9b2fa` on master implements the vector path through the ordinary compiler; this is no longer only a diagnostic. Strict master is clean and its compiler/VM were rebuilt. The earlier attention change is `67ad9d56`; upstream H3/compiler fixes were merged in `9381a701`. The intermediate emitter checkpoints remain on `codex/vector-elementwise` (isolated checkout `../StrictVector`), but the validated final source was squash-merged into master. The shared Comfy MossBlock strided-V call is committed in `8a7a9d3`.

The emitter batches four outputs of eligible paired reads from one free list. A bytecode variation/alignment/sign proof allows one address calculation per span, guarded whole/positive/four-aligned parameters, checked vector loads/stores, unchanged scalar arithmetic, and scalar tails/parameter fallbacks. Unproven gathers, positional dependencies, reductions and packed outputs retain their original kernels. Negative signed modulo has a regression and is excluded when nonnegativity cannot be proved. No model-name recognition or handwritten activation kernel was added.

Actual emitted Activated graphs on RTX 3080: **bf16 331.8 -> 261.8 us (21%)**, reversed 331.1 -> 262.6; f16 425.3 -> 277.1; f32 flat (437.9 -> 442.0). ncu (`vector-activated/final-counters.log`): 133.085M -> 102.925M executed instructions, 75.89 -> 94.01% DRAM throughput, 323.39 -> 260.64 us, registers 17 -> 40, no PTX local allocations. Executable size stays 1,204,736 bytes; PTX 19,105,727 -> 19,396,501 bytes (+1.5%).

The full-model baseline includes the same merged upstream fixes and the same strided-V source. Archives: `D:/StrictDumps/codex-elementwise/production-upstream` and `production-linear`.

| Full run | Steps 2-6 average, ms | Eight-step total, ms |
|---|---:|---:|
| Baseline 1 | 395.6 | 2901 |
| Vector 1 | 388.2 | 2849 |
| Vector 2 (order reversed) | 390.4 | 2864 |
| Baseline 2 | 393.6 | 2879 |

All four runs have identical logged errors and identical saved PNG SHA-256 `986b307e47af99d345eb09f44d6dce168d6fb1cec5e13d85dd61172067a3502e`. Accuracy remains step 0 error 0.007934923826421877, latent 0.3276526817290422, PSNR 27 dB. Details: `vector-activated/full-validation.json`. The final signed-modulo safety change leaves the model's emitted MLIR/PTX byte-identical to the measured candidate. This is a real elementwise and full-step improvement; **the 370 ms goal is still open**. No new 4090 timings are claimed.

Validation: compiler/VM rebuilt; VM and native proof regressions; native GPU MossImage, Qwen3Block and Gpt2Block; generated f32/bf16/f16 numerical, tail, sentinel and invalid-length checks; bitwise bf16 comparisons and width-7 fallback. Compute Sanitizer (`final-memcheck.log`) reports **0 errors** on tail/format/bounds checks. All changed package checks report 0 issues. A fresh isolated VM compiler-fixture load prints the upstream `InstructionEntry.LoadBoolean` unsupported-native-call warning; the native proof tests pass, and this warning was not changed here.

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

## Committed checkpoints and emitter integration (2026-10-23)
Validated Strict changes are committed as `67ad9d56` on master (strided Attention values, ABI, regression tests and measured improvements). The unfinished emitter integration is checkpointed separately as `9e300b2` on `codex/vector-elementwise`; it is not merged into master. All Strict source changes are committed and the shared master tree is clean. `git fetch --all` and another master pull still see only origin/master at c868127c; the user-reported remote fixes are not visible on that remote yet. No push has been performed.

The general emitter experiment batches paired reads from the same free list, caches vector loads, preserves existing scalar arithmetic and emits vector output stores. Read results that feed later indexes are excluded with `MlirFeeding.IsFeeding`; grouped/reduction loops and packed outputs remain on their original path. It currently computes all four read addresses and checks consecutiveness at runtime. Variable-trip-count loops left local arrays and failed a one-element tail in Compute Sanitizer. Constant four-iteration loops with inactive-lane guards remove all PTX local allocations and pass all checked f32/bf16/f16 outputs, tails, sentinels and invalid lengths. Yet actual bf16 Activated is flat (334.8 -> 333.2 us), f32 flat/slightly slower (440.8 -> 447.9), f16 faster (436.0 -> 341.9). Do not merge this checkpoint as a bf16 speed improvement. Logs: `D:/StrictDumps/codex-elementwise/vector-activated/{build-native-unroll,build-moss-unroll,unrolled-results}.log`; emitted application archive `D:/StrictDumps/codex-elementwise/production-vector`.

Next root cause: eliminate repeated index computation for a provably contiguous vector span rather than computing four addresses and testing them. Keep the adaptive checked path for gathers/unaligned cases. A small bytecode variation/alignment analysis can prove `index`, index plus uniform offsets, and `floor(index / width) * stride + index % width` are contiguous when the relevant width and vector origin are aligned. Guard those uniform parameters at runtime, compute the first address once and derive the next three. The arithmetic backend itself need not change.

## Next steps, by expected gain
1. **Extend vector coverage where measured**: The paired-read path is now integrated (latest completed change above). Earlier diagnostic plan: Activated's checked four-wide diagnostic is 17-21% faster (section above). Implement this through the general emitter for eligible contiguous paired-column reads, with shared vector bounds checks, the existing scalar/format fallback, unchanged arithmetic and output coverage. Reuse the diagnostic for emitted-instruction and alternating graph comparisons, then verify full-model accuracy/performance. Gated already reaches ~93% DRAM throughput and did not benefit. Scalar unrolling, load-ahead and noalias alone were flat or slower.
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

## RTX 3080 checkpoint: grouped indexing and Spectrum, 2026-10-23

Strict master commits `3a721f7f` (grouped quotient/remainder reuse) and `3fb5ea6f` (generic spectral fitting/history) are validated and pushed. Compiler and strict rebuilt; compiler/NeuralNetworks checks zero. Grouped indexing preserves scalar arithmetic, bounds and fallback paths. RoPE mixed-format graph median improves 174.5 -> 157.1 us; ncu instruction count 70.46M -> 56.73M, registers 40 -> 42. Full paired steady steps: baseline 379.8, candidate 378.4; reverse candidate 378.6, baseline 382.8 ms. All logged errors and PNG hashes match. Modulated did not show a consistent gain. Artifact directory: `D:/StrictDumps/codex-elementwise/group-index`. `benchmark_elementwise.py` now includes Modulated format/tail validation. Situ exponential reuse was flat and rejected.

Fresh ComfyUI, same saved noise/conditioning, merged model, Euler eight-step schedule and driver: two warm sampling runs 3013.9/2972.1 ms; individual transformer evaluations approximately 363-366 ms. All three runs including warmup reproduce saved Comfy latent exactly. Exact Strict remains behind. Evidence: `D:/StrictDumps/spectrum-moss/comfy-fresh.{json,log}`.

Timing correction: historical final-step 11-12 ms logs omitted GPU completion; the final latent comparison paid that work. Historical sums of logged step durations must be called subtotals. The last evaluation is not intrinsically cheap. Middle-step comparisons remain useful. InferenceAcceleration.md corrected.

Spectrum core is implemented with Chebyshev/ridge fitting, linear blending, real-anchor bounded history, finite/shape checks and native/GPU tests. Adapter branch `codex/spectrum-moss` is checkpointed in `StrictComfySpectrumWork`; latest diagnostic source is in non-Git `StrictComfySpectrum`. Defaults off; shadow computes real features and prediction error; on forecasts only scheduled middle steps, reruns current-time head, never stores forecasts as anchors. Full-size off run currently stalls around VAE decoding; do not merge or claim end-to-end validation. Full-size shadow/live quality and memory still unmeasured. Diagnostic logs/builds: `D:/StrictDumps/spectrum-moss`.

Next larger exact-performance lead: installed ComfyUI routes `linear_input_act(..., "situ")` through fused activation + convrot64 + INT8 quantization. Strict materializes Activated before Hadamard/quantization. A general producer fusion emitter could remove this buffer; preserve rounding semantics and measure generated code. Existing whole GEMM already accumulates INT32 across K before applying scales, and Hadamard quantization already uses token scales: neither is missing. Await 4090 findings before repeating rejected scheduling experiments.


## Spectrum full-size validation and native build issue (2026-10-23, later checkpoint)

Adapter source and report are checkpointed on `codex/spectrum-moss` (`Spectrum.md`, latest code checkpoint 6fc2ae6). Shared main stays exact; do not merge the adapter yet. Full-size off/shadow/on runs now complete with a diagnostic monolithic host link, using the same generated MLIR/PTX. Off and shadow reproduce the historical PNG exactly. One live forecast at step 5 takes 17 ms instead of roughly 380 ms; sampling sums off/shadow/on are 3020/3258/2767 ms. This single sequential comparison gives 8.4% less sampling time, with clock variation; it is not a calibrated preset or an exact-inference win. Live PNG is 36.20 dB against off, mean absolute byte difference 1.92/255, max 146; visual review is close for this one prompt. Latent reference error changes 0.32765268 -> 0.34812940; decoder reference PSNR about 27 -> 26.84 dB. Shadow feature relative errors 0.699/0.491/0.825 at steps 5/6/7. Multiple prompts/seeds and base-model 25/50 steps remain unvalidated. Generic periodic scheduling is implemented, upstream adaptive sampled-token/ER-SDE integration is not.

The public adapter branch's standard split/ThinLTO build reproduces a full-size off-mode VAE stall (90-second timeout after latent comparison); native fixture and VM tests/checks pass. Standalone VAE takes 463 ms. nsys sees a late 450 MiB download then CPU-only execution. Relinking identical host MLIR/PTX monolithically completes VAE in 429 ms; preventing inlining of MossSpectrumImage.Generated also makes the split host link finish (797 ms in that run). Removing Compared does not fix it. Scalar timing reads changed tiny rounding and were removed; completion now uses the existing reference-error traversal and restores exact baseline latent error. STRICT_RC_CHECK reports two reachable 16-byte objects that reclamation would otherwise free near VAE load, but keeping them does not resolve the stall; causality is unproven. Leading hypothesis: host inlining/root lifetimes retain model/intermediate storage into the high-memory decoder. Do not label this an LLVM bug without further reduction. Proper next fix is precise lifetime/root handling at generation scope, with this full-size run as the regression; monolithic linking/noinline are diagnostic controls, not production fixes.

Debugging reached AGENTS.md's ten-edit limit for this reproduction; stop source-fix attempts pending a new prompt. Evidence and generated-host controls: `D:/StrictDumps/spectrum-moss/{validation.json,work-off.log,rc-check.log,off-profile.sqlite,monolithic.log,noinline.log,shadow.log,on.log}` and `build_{dispatch,monolithic,noinline}.py`. Working trees are committed; diagnostic source changes remain only in the non-Git snapshot.
