# Moss speed and quality comparisons

## Generate from a prompt file

Build the launcher and backend once, from StrictComfy:

```powershell
../Strict/strict.exe build MossGenerate.strict
../Strict/strict.exe build MossImage.strict
```

Then run from any working folder containing your UTF-8 `prompt.txt`:

```powershell
C:/code/GitHub/strict-lang/StrictComfy/MossImage.exe --model moss-image-sft-dit-int8-convrot-dmd.safetensors
```

Weighted stacks use repeated `--lora` arguments. Paths may be absolute, relative to the working folder, or relative to ComfyUI's `models/loras`. Model names also resolve under `models/diffusion_models`.

```powershell
./MossImage.exe --prompt prompt.txt --model moss.safetensors --steps 4 --lora somelora1.safetensors:0.5 --lora somelora2.safetensors:0.31 --output MossImage.jpg
```

Defaults: `--prompt prompt.txt`, `--steps 8`, `--output MossImage.jpg`, 720x1280, guidance 1, JPEG quality 85, random seed. `--seed 42` fixes both prompt choices and initial noise. `--width`, `--height`, `--guidance`, `--quality`, `--vae` and `--te` override those settings; dimensions must be multiples of 16. Choose an int8 Convrot Moss checkpoint. Eight steps assumes a turbo checkpoint or an explicitly supplied turbo LoRA; the command does not silently add one. Four steps does not automatically select a four-step adapter. Rapid-MOSS's separate time/data-space adapter remains available through the comparison harness, not this simple command.

Every output gets at least four digits: `MossImage0001.jpg`, `MossImage0002.jpg`, etc. Numbering starts after the largest existing number for that base/extension, then reserves a free file exclusively; concurrent commands cannot overwrite each other's images. Existing unnumbered files are untouched. `.jpg`, `.jpeg` and `.png` are accepted; JPEG uses the Windows codec. Each image has a `.jpg.json` / `.png.json` sidecar with its resolved prompt, seed, model, weighted adapters and settings.

Prompt alternatives use `{choice|choice|}`. The trailing empty choice is real: `{a woman|a man|}` selects `a woman`, `a man` or nothing. Multiple and nested groups are supported; `|` outside a group remains literal. Unbalanced braces report an error. The prompt file itself is not modified.

The command reuses the installed ComfyUI text encoder in a worker that exits before Strict inference, freeing its GPU memory. Windows portable installations on D: and C: are detected. For another installation, pass `--comfyui C:/path/ComfyUI --python C:/path/python_embeded/python.exe`; `COMFYUI` also selects the ComfyUI directory. The default encoder is `qwen3.5_9b_int8_convrot.safetensors` and VAE `flux2-vae.safetensors`. Keep `moss.py`, `merge.py`, `dump.py`, `MossGenerate.exe` and its matching PTX files next to the launcher.

Resolved prompt conditioning, weighted merged models and CUDA compilation are cached under `StrictComfy/artifacts/moss`; `--cache D:/StrictDumps/moss-cli/cache` changes the location. A new LoRA stack creates a full cached checkpoint (about 5.4 GB for the tested model). All weighted deltas are added before a single int8 requantization; source files are unchanged. Kohya down/up and PEFT A/B adapters are supported, including fused qkv split into q/k/v, plus matching floating-point dense targets. LoKR and other adapter formats are not supported by this merger; a completely unmatched adapter fails instead of silently doing nothing. Zero strengths leave the base unchanged; finite negative strengths are accepted. This is the existing int8 merge quality tradeoff, not unmerged floating-point LoRA evaluation.

First use includes text encoding, LoRA merging and driver compilation. Cached calls still load a fresh native model: the tested default eight-step command took 10.81 s end to end, with 5.66 s inside generation. This is not the earlier resident 3.47 s timing. A ComfyUI sampler node is deferred until a resident backend and text-encoder/Strict VRAM handoff can avoid those cold loads; the local command is usable now. Existing positional `MossImage.exe dumps model lora vae` diagnostics still work.

Command checks (no full model needed): `D:/ComfyUI_windows_portable/python_embeded/python.exe test_moss.py`.

## Comparison harness

`MossGenerate` generates images without the per-layer and per-step reference comparisons in `MossImage`. `MossSampling` supports the existing flow Euler schedule, ordinary CFG with zero conditioning, separate Spectrum histories for its positive/negative branches, and Rapid-MOSS data-space iteration. No error or PNG equality requirement is imposed across these methods.

`compare_moss.py` prepares identical conditioning/noise within each prompt/seed case, runs the variants and writes a local HTML gallery. Default cases: lighthouse, portrait with hands, and a cafe menu; seeds 42 and 1234; 720x1280. Edit `comparison.json` in the chosen output root to use your own prompts, seeds or settings, then prepare again.

## Run

From StrictComfy, using ComfyUI's installed Python and existing base model/text encoder/VAE:

```powershell
../Strict/strict.exe build MossGenerate.strict
& D:/ComfyUI_windows_portable/python_embeded/python.exe compare_moss.py --download
& D:/ComfyUI_windows_portable/python_embeded/python.exe compare_moss.py --merge
& D:/ComfyUI_windows_portable/python_embeded/python.exe compare_moss.py --prepare
& D:/ComfyUI_windows_portable/python_embeded/python.exe compare_moss.py --run --report
```

For interactive-style timing, keep the model and decoder loaded across generations:

```powershell
& D:/ComfyUI_windows_portable/python_embeded/python.exe compare_moss.py --run --cases portrait-42 --variants turbo8 --resident-runs 3 --report
```

`--repeat` starts separate processes; `--resident-runs` reruns inference inside one process. Each resident pass executes the sampler again with the saved input, rather than reusing an image. Logs and JSON retain every pass, including the first one. A fresh Spectrum history is created for each image. `GpuMemory.Released` runs between resident generations after saving, freeing unused VAE workspace without changing the global H3 allocation policy.

Output defaults to JPEG quality 85 on Windows, using its built-in codec. `--quality 0..100` selects JPEG quality. `--format png` selects lossless stored-DEFLATE PNGs; add `--compressed` for compressed PNGs. The shared `ImageSaver` still defaults to compressed PNGs. JPEG accepts flat RGB values directly, avoiding the temporary ColorValue list. On the 720x1280 portrait, JPEG was 95,022 bytes versus 3,688,028 for stored PNG, and saving took 25-30 ms. JPEG decoded pixels differ: 42.22 dB PSNR and mean absolute error 1.45/255 against the PNG. This is an explicit output quality tradeoff, independent of model precision. Native JPEG currently requires Windows; direct MossGenerate defaults to PNG on other platforms.

Run `--prepare` separately so the text encoder process releases its GPU memory. Default output root is `D:/StrictDumps/moss-speed-comparison`; `--root` overrides it. `index.html` links full-resolution images; `summary.json`, per-run JSON/logs and `comparison.json` record timings and settings. Models and images are not checked in. The existing eight-step merged model is expected at the path in the manifest; `merge.py` can create it from the base model and your existing DMD LoRA.

For a small comparison or repeat:

```powershell
& D:/ComfyUI_windows_portable/python_embeded/python.exe compare_moss.py --run --cases portrait-42 --variants base25 turbo8 akhaliq4 rapid4 --repeat 2 --report
```

The executable itself accepts `dumps model lora vae mode adapter`. `generation.safetensors` contains `noise` [1,128,H/16,W/16], `context` and scalar `steps`, `guidance`, `warmup`, `period`; optional `runs` defaults to one, `compressed` to zero, `jpeg` to one on Windows and zero elsewhere, and `quality` to 85. An empty adapter path selects Euler; a Rapid adapter folder selects its replacement timestep weights, biases, mixing matrix and step codes. Mode is off/shadow/on for Euler. Data-space currently does not apply Spectrum.

## What each column means

| Variant | Sampling | Transformer evaluations |
|---|---|---:|
| base25 | Base int8 Convrot, Euler shift 5, ordinary CFG 3 | 50 |
| spectrum25 | Same, warmup 8, period 3, degree 4, actual final step | 28 actual, 22 forecasts across two branches |
| turbo8 | Existing merged DMD, Euler shift 5, CFG 1 | 8 |
| spectrum8 | Same, warmup 5, period 2 | 7 actual, 1 forecast |
| akhaliq4 | Current Euler-native four-step LoRA, merged int8, CFG 1 | 4 |
| rapid4 | darask0 data-space adapter, merged int8, CFG 1 | 4 |

The 25-step preset uses zero conditioning for CFG and no skipped negative block. It is not the release pipeline's default negative-block-22 skip, nor ComfyUI's ER-SDE preset. It provides an explicitly labeled baseline rather than claiming the best attainable 25-step quality. The eight-step model is the existing Euler-compatible DMD setup; it is not switched to a different checkpoint or sampler during comparison.

Rapid-MOSS requires all three files. Its iteration is `x = z + Q*x0; x0 = z - model(x, code[r])`, with fixed original noise, not four Euler steps. The custom path uses the exact code's sinusoid and replacement f32 time embedding/biases. Other small dense weights keep the existing model loader's precision. In contrast, akhaliq's current root adapter is Euler-native; its old data-space checkpoint instructions do not apply to this revision.

Both four-step speed variants use the existing `merge.py`: form the LoRA delta in f32, rotate it and requantize projections to int8 once. This is a quality tradeoff: small deltas can be lost, and upstream evaluates unmerged floating-point adapters. Optional `akhaliq4-unmerged` / `rapid4-unmerged` variants exercise Strict's existing 16-bit low-rank loading path; they do not claim upstream's fp32-LoRA fidelity. The unmerged tests are excluded by default because they are much slower and use more memory here.

Sources and pinned adapter revisions:

- [Rapid-MOSS](https://huggingface.co/darask0/MOSS-Image_4-step_adapter_lora), c88f33bcb3251eb39943d0635e74b650732977c8.
- [akhaliq four-step](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps), 2ded95c37a029360245926fe72e475e30039bbfb.
- [RunComfy workflow](https://www.runcomfy.com/comfyui-workflows/moss-image-rapid-moss-sft-4-step-comfyui-text-to-image). The adapter's own current sampler instructions take precedence over the workflow's generic guidance description.
- Spectrum feature coverage and missing upstream features: [Spectrum.md](Spectrum.md).

## Timing

`sampling_ms` includes text projection, initial uploads, all evaluations and final latent synchronization, but excludes loading and decode/save. `denoising_ms` starts at evaluation 0 and includes its initial uploads. Neither is a resident-model steady-step-only measurement. Progress lines mark host submission; they do not individually synchronize GPU completion. Process wall time includes loading, decoding, saving and startup, across all resident passes. `generation_runs_ms` measures each complete sampling/decode/save pass (and boundary cleanup when another pass follows); `sampling_runs_ms` retains its sampler timings. Gallery cells label the last generation and the total process separately.

The harness uses a writable CUDA JIT cache under its output root, with a 4 GiB cap. A cold compiled module may add tens of seconds; warm it before comparing timings. This is a per-process setting, not a global driver change. RTX 3080 results use locked 1695 MHz graphics / 9506 MHz actual CUDA memory, with clock limits restored afterwards. Results and quality observations are in [Performance.md](Performance.md).
