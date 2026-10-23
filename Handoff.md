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

## Next steps, by expected gain
1. **Vector elementwise codegen** (biggest, general): let a generated GPU loop process 4 or 8 consecutive numbers per thread when its list reads are `list(index + c)` / `list(index % w + c)` patterns, with one 16 byte load per list (8 bf16 or 4 f32), one bounds check per vector and the format resolved once. Start in `Compiler/MlirDevice.strict` (Kernel, Started, Emitted), `MlirDeviceRows`, `MlirLayout.Stored`, runtime `strict_device_floats` / `strict_device_stored` in `Compiler/NativeRuntimeDevice.mlir`, launch thread count in `strict_gpu_run` (`Compiler/NativeRuntime.mlir`). Gated and Columns are the simplest first targets (pure index reads), then Activated, Modulated, Rotated. Step 0 is the unroll alone: `MossBlock_Activated_2_gpu6sx` is ~90 PTX instructions with 2 global loads and no local-memory traffic (ptxas promotes the `%r` slots), so the limit is memory-level parallelism (ncu: 50 % long-scoreboard, 2 loads in flight, ~45 % of DRAM). Four elements per thread gives 8 independent loads; the tail needs a guard because `count % 4` is not always 0. Then batch the reads/writes per format.
2. **Format specialization**: the format branch per read measured 69 µs of Rotated's 279 (improvements.md item 5). Either the vector path above hoists it, or emit kernel variants per format signature chosen on the host like the exact `sx` kernels (`MlirGuards`, `strict_gpu_guarded`).
3. **Remaining division slow path**: Rotated's `token = (head * size / width).Floor` (a product numerator) keeps the 64 bit slow path. Uncommitted, VM tested only: the floor correction moved onto the slow path, so magic-path quotients stay provably 32 bit and chained divisions fold: scratchpad `floored.patch` (see path below); then write `token = (head / heads).Floor`.
4. **Rotated as bfloat16**: halves its write and the attention's key reads, but `strict_device_attentions` got slower with bfloat16 queries/keys (3193 -> 3586 µs): ncu it; the fix is probably bf16 mma (m16n8k16 bf16) instead of converting each bf16 pair to f16 in `strict_device_attention_stored` (`Compiler/NativeRuntimeAttention.mlir`), or Rotated returning f16 lists.
5. **Fusions** (need a cross-method view: an inliner for loop methods, `Bytecode/Inliner.strict` only inlines small loop-free ones): Gated into the o/down GEMM epilogue (`strict_device_wholes`), Gated + next Modulated in one grouped kernel, Columns removed by letting Attention read values strided out of the qkv result.
6. **GEMM 22 ms**: within ~5 % of comfy-kitchen CUTLASS per call in isolation; the rest is occupancy/tiles per shape. A CUTLASS DLL prototype exists (scratchpad `ext/strict_cuda`) but was not worth a dependency.
7. **Host gaps ~15 ms per step**: nsys `busy.py` shows 95-97 % busy; look at launches between layers.

## How to build, run, measure
- Rebuild after `../Strict/Compiler` changes: `strict build Compiler/NativeCompiler.strict` twice, delete `Strict.strictbinary`, `strict build Strict.strict` (from `../Strict`). Runtime `.mlir` changes need no rebuild, programs read them at build time. mlir-translate crashes about every tenth build (0xC0000005), just retry. Right after strict.exe is replaced a call can exit 127, retry after a second.
- Native builds launched from StrictComfy currently fail (ggml flags paths, `ggml-cpu/quants.h` not found; the 4090 agent has a fix on branch native-flags-root, not merged yet). Workaround: build from `../Strict`: `strict build ../StrictComfy/MossImage.strict` and run executables with StrictComfy as working directory. A file whose VM tests fail on relative model paths (`MossBlock.strict` reads `tinymoss.safetensors`) skips the native build silently (`NativeGate.Measured`): copy `tiny*.safetensors` into `../Strict` for the build, then delete them.
- Run: `MossImage.exe D:/StrictDumps/mix720 D:/ComfyUI_windows_portable/ComfyUI/models/diffusion_models/moss-image-int8-convrot-dmd-lenovo-finger.safetensors "" D:/ComfyUI_windows_portable/ComfyUI/models/vae/flux2-vae.safetensors` from StrictComfy (~70 s). Current: steps 411-417 ms, step 0 error 0.00800, latent 0.4115, PSNR 24.85 dB. The 8 step trajectory is chaotic to last-bit changes; judge accuracy on the embedder, layer 0/16 and step 0 errors and by looking at the image.
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
