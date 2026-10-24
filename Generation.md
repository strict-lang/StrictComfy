# Moss generation and quality comparisons

## Generate from a prompt file

Build the launcher/backend from StrictComfy:

```powershell
../Strict/strict.exe build MossGenerate.strict
../Strict/strict.exe build MossImage.strict
```

From any folder containing UTF-8 `prompt.txt`:

```powershell
C:/code/GitHub/strict-lang/StrictComfy/MossImage.exe --count 10
```

`--help` lists defaults; `-h` is unsupported.

| Option | Default |
|---|---|
| `--prompt` | `prompt.txt` in the working folder |
| `--output` | `MossImage.jpg`, with an automatic four-digit number |
| `--count` | 1 |
| `--width`, `--height` | 720, 1280 (multiples of 16) |
| `--steps` | 8; automatically includes DMD turbo at strength 1 |
| `--lora_mode` | `merged`; cached stochastic INT8 merge; `runtime` retains floating adapters |
| `--spectrum` | `off`; `spectrum8_w3p3` enables three forecasts at eight steps |
| `--merge_only` | off; prepare/print the merged checkpoint and exit |
| `--guidance` | 1 (CFG scale) |
| `--jpg_quality` | 85, JPEG compression on a 0-100 scale; no effect on inference |
| `--json_output` | off; enable per-image JSON settings sidecars |
| `--model` | `D:/ComfyUI_windows_portable/ComfyUI/models/diffusion_models/moss-image-sft-dit-int8-convrot.safetensors` |
| `--vae` | `D:/ComfyUI_windows_portable/ComfyUI/models/vae/flux2-vae.safetensors` |
| `--te` | `D:/ComfyUI_windows_portable/ComfyUI/models/text_encoders/qwen3.5_9b_int8_convrot.safetensors` |

Paths follow `--comfyui` / `COMFYUI`; a portable installation on C: is detected too. Explicit paths/names override defaults. The default input is the **base** INT8 checkpoint. Eight steps includes `models/loras/moss/dmd_turbo_lora.safetensors` at strength 1; other counts do not automatically add it. An explicit DMD entry overrides automatic selection (including `:0`). Checkpoints produced by the current merger record their included adapters, so passing the cached checkpoint as `--model` does not add DMD again. Old premerged files without that marker are not recognized; use the original base for a fresh merge.

Weighted stacks use repeated options; relative paths also resolve under ComfyUI's `models/loras`:

```powershell
./MossImage.exe --lora moss/lenovo_moss.safetensors:0.5 --lora moss/cndamiph_Candid_Amateur_iPhone_MOSS_v1.safetensors:0.31 --count 10
```

Default `--lora_mode merged` folds the entire selected stack into one cached INT8 Convrot checkpoint, once per model/adapter/strength combination. All updates accumulate before requantization with deterministic stochastic rounding, preserving sub-quantization updates statistically. Source files stay unchanged. This is approximate: images can change relative to floating runtime adapters. The tested five-adapter stack remains coherent across three portrait seeds, but this is not universal quality acceptance. Each cached stack occupies a full model checkpoint; changing strengths creates another cache entry. `--cache` chooses its folder; obsolete `model-*.safetensors` caches are never reused.

Use `--lora_mode runtime` to retain the floating-adapter path. Weighted stacks concatenate down rows and strength/alpha-scaled up columns. Kohya down/up and PEFT A/B layer adapters, including fused qkv split into q/k/v, are supported. The merger also supports matching floating model weights. Unsupported formats such as LoKR report errors. Finite negative strengths are accepted. Four steps needs an appropriate explicitly selected adapter; Rapid-MOSS's separate time/data-space adapter remains a comparison-harness feature.

The fastest tested eight-step option is explicit:

```powershell
./MossImage.exe --count 10 --spectrum spectrum8_w3p3
```

Add your usual repeated `--lora path:strength` options. This preset uses warmup 3/period 3, five actual evaluations and three forecasts. It changes the sampling result; Spectrum remains off unless selected. Previously the CLI hardcoded it off, even though the comparison harness supported it.

For a reusable merged file without loading the encoder or generating an image:

```powershell
./MossImage.exe --lora moss/style.safetensors:0.5 --merge_only
./MossImage.exe --model 'D:/path/to/merged-stochastic-v1-....safetensors' --count 10
```

Do not repeat the already baked style options when using that checkpoint. Alternatively, keep using the same original command: its cache is reused automatically. First merge of the tested five-adapter stack took 15.35 s through the launcher (11.58 s in the isolated CPU merge). Warm native generations were 3.25 s merged, or 2.30-2.33 s with Spectrum, versus 7.18-7.23 s for the latest floating runtime path. These exclude process/model/text-encoder startup; full CLI validation and quality limits are in Performance.md.

`--count 10` keeps one model and VAE loaded. Every image gets a distinct seed and independently expanded prompt; choices may repeat by chance. `--seed 42` reproduces the batch, starting with seed 42 and deriving distinct subsequent seeds. Uncached unique prompts are encoded together with one encoder load before native inference. Completed images are published as the backend reports completion and survive a later job failure.

Outputs are `MossImage0001.jpg`, `MossImage0002.jpg`, etc. Numbering follows the largest existing number for that base/extension and reserves each file exclusively, including across concurrent commands. Existing files are not overwritten. `.jpg`, `.jpeg` and `.png` are accepted. By default **only the image** is created. `--json_output` adds its `.jpg.json` / `.png.json` settings, resolved prompt, seeds, active LoRAs, `batch_index`, `count`, `generation_ms`, and elapsed batch `seconds`.

Prompt groups `{a woman|a man|}` include the empty alternative. Multiple/nested groups work; `|` outside braces is literal. Unbalanced braces report an error. The prompt file is not changed.

## Current dependencies

The full command currently needs an installed ComfyUI source tree and its Python environment; the UI/server need not be running.

- ComfyUI: tokenization, Qwen text encoding, encoder loading/custom operations and setup reused through dump.py. Its encoder worker exits before native inference to release GPU memory.
- PyTorch/safetensors: input noise, conditioning files, cached model merging and floating adapter-stack preparation. This is not ComfyUI sampling.
- Strict native backend: Euler sampling, diffusion transformer/GEMM/attention/elementwise operations, VAE decoding and JPEG saving. With prepared conditioning, MossGenerate.exe can run without Python/ComfyUI.

Use `--comfyui C:/path/ComfyUI --python C:/path/python_embeded/python.exe` for another installation. Keep moss.py, merge.py, dump.py, MossGenerate.exe and matching PTX files beside the launcher. Prompt/model/adapter/CUDA caches default to StrictComfy/artifacts/moss; --cache changes their location. The existing positional `MossImage.exe dumps model lora vae` diagnostic mode remains available.

## Quality review and timing correction

The old default premerged DMD checkpoint is not a trustworthy turbo reference. In sampled q and FFN-down matrices its INT8 weight codes were entirely unchanged from base, with only tiny row-scale changes. Folding small updates back into coarse base quantization can discard them. Earlier fast timings, including 45.07 seconds for ten images, used that artifact and must not be presented as validated runtime-turbo quality/performance.

Review samples: 720x1280, prompt `Portrait photograph of a woman, natural daylight.`, seed 42, no Spectrum. Base 50/25 use CFG 3 without LoRA; turbo 8 uses CFG 1 and the original floating adapter:

- D:/StrictDumps/moss-quality-review/base50/strict-off.jpg
- D:/StrictDumps/moss-quality-review/base25/strict-off.jpg
- D:/StrictDumps/moss-quality-review/turbo8-runtime/strict-off.jpg

The direct native turbo run took 18.39 seconds including load (12.56 seconds generation); the full CLI check took 26.04 seconds and matched that JPEG exactly. These are slower than the old merged path. Identity scaling now preserves packed adapter rows; before that fix the same runtime experiment exceeded 180 seconds. Runtime LoRA GEMMs remain a performance target. These three images are for review, not a multi-prompt quality acceptance claim. A ComfyUI sampler node still needs coordinated encoder/Strict VRAM handoff.

Checks: `D:/ComfyUI_windows_portable/python_embeded/python.exe test_moss.py`.

## Comparison harness

New manifests use base weights plus the original runtime DMD for turbo8/spectrum8. Existing comparison.json files are not rewritten: update their model/lora fields or use a fresh output root before comparing. Older gallery results using the premerged DMD file are not validation of the corrected default.

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

Output defaults to JPEG quality 85 on Windows, using its built-in codec. `--jpg_quality 0..100` selects JPEG compression quality. `--format png` selects lossless stored-DEFLATE PNGs; add `--compressed` for compressed PNGs. The shared `ImageSaver` still defaults to compressed PNGs. JPEG accepts flat RGB values directly, avoiding the temporary ColorValue list. On the 720x1280 portrait, JPEG was 95,022 bytes versus 3,688,028 for stored PNG, and saving took 25-30 ms. JPEG decoded pixels differ: 42.22 dB PSNR and mean absolute error 1.45/255 against the PNG. This is an explicit output quality tradeoff, independent of model precision. Native JPEG currently requires Windows; direct MossGenerate defaults to PNG on other platforms.

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
| turbo8 | Base INT8 + original runtime DMD, Euler shift 5, CFG 1 | 8 |
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
