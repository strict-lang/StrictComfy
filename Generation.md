# Moss speed and quality comparisons

## Generate from a prompt file

Build the launcher and backend once, from StrictComfy:

```powershell
../Strict/strict.exe build MossGenerate.strict
../Strict/strict.exe build MossImage.strict
```

Then run from any working folder containing your UTF-8 `prompt.txt`:

```powershell
C:/code/GitHub/strict-lang/StrictComfy/MossImage.exe --count 10
```

Weighted stacks use repeated `--lora` arguments. Paths may be absolute, relative to the working folder, or relative to ComfyUI's `models/loras`. Model names also resolve under `models/diffusion_models`.

```powershell
./MossImage.exe --prompt prompt.txt --model moss.safetensors --steps 4 --lora somelora1.safetensors:0.5 --lora somelora2.safetensors:0.31 --output MossImage.jpg
```

`--help` shows the defaults; `-h` is deliberately unsupported. With no options, the command generates one image using these settings:

| Option | Default |
|---|---|
| `--prompt` | `prompt.txt` in the working folder |
| `--output` | `MossImage.jpg` (number added automatically) |
| `--count` | 1 |
| `--width`, `--height` | 720, 1280 |
| `--steps` | 8 |
| `--guidance` | 1 (CFG scale) |
| `--jpg_quality` | 85 on the JPEG 0-100 compression scale; no effect on diffusion |
| `--model` | `D:/ComfyUI_windows_portable/ComfyUI/models/diffusion_models/moss-image-sft-dit-int8-convrot-dmd.safetensors` |
| `--vae` | `D:/ComfyUI_windows_portable/ComfyUI/models/vae/flux2-vae.safetensors` |
| `--te` | `D:/ComfyUI_windows_portable/ComfyUI/models/text_encoders/qwen3.5_9b_int8_convrot.safetensors` |

These are the current 3080 test paths. File defaults follow the selected `--comfyui` / `COMFYUI` directory; a portable installation on C: is also detected. Explicit paths/names override each default. Dimensions must be multiples of 16. Eight steps uses the default premerged turbo checkpoint; choosing another base model does not silently add a turbo LoRA. Four steps does not automatically select a four-step adapter. Rapid-MOSS's separate time/data-space adapter remains available through the comparison harness, not this simple command.

`--count 10` generates ten images in one native process, keeping the model and VAE loaded. Each image gets a different seed and independently expanded prompt alternatives. Choices can repeat by chance, but noise seeds remain distinct. Without `--seed`, every batch is random; `--seed 42` reproduces the whole batch, using 42 for the first image and distinct derived seeds afterward. Uncached unique prompts are encoded together with one text-encoder load before native inference; already cached prompts are reused. Each completed image is copied to its exclusively reserved numbered output as the backend reports completion. Completed images remain if a later job fails. Per-image sidecars include `batch_index`, `count`, `generation_ms` and elapsed batch `seconds`.

Every output gets at least four digits: `MossImage0001.jpg`, `MossImage0002.jpg`, etc. Numbering starts after the largest existing number for that base/extension, then reserves a free file exclusively; concurrent commands cannot overwrite each other's images. Existing unnumbered files are untouched. `.jpg`, `.jpeg` and `.png` are accepted; JPEG uses the Windows codec. Each image has a `.jpg.json` / `.png.json` sidecar with its resolved prompt, seed, model, weighted adapters and settings.

Prompt alternatives use `{choice|choice|}`. The trailing empty choice is real: `{a woman|a man|}` selects `a woman`, `a man` or nothing. Multiple and nested groups are supported; `|` outside a group remains literal. Unbalanced braces report an error. The prompt file itself is not modified.

The command reuses the installed ComfyUI text encoder in a worker that exits before Strict inference, freeing its GPU memory. Windows portable installations on D: and C: are detected. For another installation, pass `--comfyui C:/path/ComfyUI --python C:/path/python_embeded/python.exe`; `COMFYUI` also selects the ComfyUI directory. The default encoder is `qwen3.5_9b_int8_convrot.safetensors` and VAE `flux2-vae.safetensors`. Keep `moss.py`, `merge.py`, `dump.py`, `MossGenerate.exe` and its matching PTX files next to the launcher.

Resolved prompt conditioning, weighted merged models and CUDA compilation are cached under `StrictComfy/artifacts/moss`; `--cache D:/StrictDumps/moss-cli/cache` changes the location. A new LoRA stack creates a full cached checkpoint (about 5.4 GB for the tested model). All weighted deltas are added before a single int8 requantization; source files are unchanged. Kohya down/up and PEFT A/B adapters are supported, including fused qkv split into q/k/v, plus matching floating-point dense targets. LoKR and other adapter formats are not supported by this merger; a completely unmatched adapter fails instead of silently doing nothing. Zero strengths leave the base unchanged; finite negative strengths are accepted. This is the existing int8 merge quality tradeoff, not unmerged floating-point LoRA evaluation.

First use includes text encoding, LoRA merging and driver compilation. A new CLI invocation still loads a fresh native model, but only once per batch. The cached ten-image check took 45.07 s end to end, one native load of 3.052 s, warm median generation 3.563 s including decode/save. It used two weighted adapters, ten distinct seeds and three resolved prompt variants. A batch that first encoded two missing prompts and merged the stack took 78.34 s. These normal-clock measurements are not a locked-clock ComfyUI comparison. A ComfyUI sampler node still needs an interactive resident backend and coordinated text-encoder/Strict VRAM handoff. Existing positional `MossImage.exe dumps model lora vae` diagnostics still work.

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

# H3 generation with LoRAs, step counts and Spectrum (RTX 4090)

`dump.py h3 --runs 0` writes the inputs of a prompt and seed without sampling in ComfyUI: `conditioning.safetensors` (Qwen3-VL states and token tags) and `steps.safetensors` (the noise as `video.0`/`audio.0`, `sigmas` of the default 8 steps). Text to video and audio only; image to video needs a full `dump.py h3` run for its noise augmented cond rows. `H3Video.exe` samples a folder, `dump.py h3decode` decodes every `<name>-latent.txt` in it with ComfyUI's VAEs into `<name>-video.mp4` (plus first, middle and last frame PNGs), so several variants of one prompt and seed sit side by side.

```powershell
cd C:\code\GitHub\strict-lang\StrictComfy
$python = "C:\ComfyUI_windows_portable\python_embeded\python.exe"
$models = "C:\ComfyUI_windows_portable\ComfyUI\models"
& $python dump.py h3 --runs 0 --out C:\StrictDumps\fox --prompt "A red fox trotting through fresh snow in a pine forest" --seed 7
$env:STRICT_GPU = "all"
.\H3Video.exe C:\StrictDumps\fox "$models\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors" taomate3 "$models\loras\h3\minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors=1" 3
.\H3Video.exe C:\StrictDumps\fox "$models\diffusion_models\Minimax_H3_FL2VA_PRUNED_Turbo_8step_v1.0_768p_comfyui_INT8_.safetensors" spectrum8 "" 0 12 2
& $python dump.py h3decode --out C:\StrictDumps\fox
```

Arguments after the executable: `dumps model name loras steps shift period warmup degree`, defaults `strict "" 0 12 0 1 1`.

- `name`: the output is `<name>-latent.txt`.
- `loras`: `path=strength` entries separated by `;`, a path without `=` has strength 1, `""` for none. Keys `[diffusion_model.]blocks.N.{attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2}.{lora_A,lora_B,lora_down,lora_up}.weight` with an optional `alpha` (scale alpha / rank, 1 without, times the strength), the same for `token_refiner.blocks.N`. Several LoRAs stack by rank. The int8 convrot block weights get the update merged at load like ComfyUI does (dequantized, the rotated update added, requantized per row with stochastic rounding; `Safetensors.Tuned` in ../Strict), so steps cost the same as without LoRA; the dense token refiner applies them per call. The load line reports how many of each file's tensors were applied: adaln, patch projection and final layer LoRAs and FastH3's `diff`/`diff_b` tensors are not applied yet.
- `steps`: 0 takes the count of the folder's `sigmas`; the schedule is ComfyUI's simple flow schedule `shift t / (1 + (shift - 1) t)` (8 steps match ComfyUI exactly), `shift` is the video flow shift and the audio sigmas follow from it with shift 3 like ComfyUI's MiniMax H3 shift node. Per step errors against ComfyUI print only when the folder has its references and the step count matches.
- `period`: Spectrum, 0 is off. After `warmup` actual steps every `period`-th step is actual, the others are forecast from the last actual ones, the last step is always actual: 8 steps with period 2 are A F A F A F A A (5 transformer evaluations), period 3 skips two of three. `degree` is the Chebyshev degree of the fit (ridge 0.1, blended half with the line through the last two, `SpectralFit`), it keeps `degree + 1` anchors; with one anchor the forecast holds it. H3 forecasts its output, not its features ([Spectrum.md](Spectrum.md)).

Loading the 21 GB model takes 14-20 s, merging adds about 20 s for TaoMate (rank 4-21) and 80 s for the turbo 8-step LoRA (rank 128-384). The first step of a fresh process uploads all weights once (about 12 s more), the H3 reference folder hides that in its per-layer comparison before sampling. Timings and quality numbers are in [Performance.md](Performance.md).
