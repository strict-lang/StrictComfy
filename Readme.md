# StrictComfy

ComfyUI's Moss (image DiT) and MiniMax H3 (video and audio DiT) running in [Strict](../Strict): Strict samples natively on the GPU, ComfyUI loads the models, encodes the prompts and decodes the latents. Pending work is in [Plan.md](Plan.md), measurements in [Performance.md](Performance.md), Spectrum forecasting in [Spectrum.md](Spectrum.md).

## Build

From this folder, with `../Strict/strict.exe`:

```powershell
../Strict/strict.exe build MossGenerate.strict
../Strict/strict.exe build MossImage.strict
../Strict/strict.exe build H3Video.strict
```

`MossGenerate.exe` is the Moss backend (used by the ComfyUI node and `MossImage.exe`), `H3Video.exe` is the H3 command and backend. Keep the built `.exe` and `.ptx` files next to `generate.py`, `merge.py` and `dump.py`. The first run of a new build compiles its PTX (tens of seconds), later runs use the CUDA cache in `artifacts/`.

## ComfyUI node: Strict Moss Sampler

Install once by linking this folder into ComfyUI's custom nodes and restart ComfyUI:

```powershell
New-Item -ItemType Junction -Path D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\StrictComfy -Target C:\code\GitHub\strict-lang\StrictComfy
```

Workflow, everything except the sampler is a standard ComfyUI node:

| Node | Settings |
|---|---|
| Load Diffusion Model (UNETLoader) | a Moss checkpoint: int8 Convrot like `moss-image-sft-dit-int8-convrot.safetensors`, or fp8, bf16, fp16 or fused qkv ones (converted once, cached) |
| LoraLoaderModelOnly, chained for a stack, optional | style LoRAs (the DMD turbo LoRA is added by `turbo` `auto`, a LoraLoader with it is recognized and not added twice) |
| Load CLIP | `qwen3.5_9b_int8_convrot.safetensors`, type `moss_image` |
| CLIP Text Encode | the prompt, into `positive` |
| Empty Flux 2 Latent | width, height (multiples of 16), batch size |
| Load VAE | `flux2-vae.safetensors`, decodes the result in ComfyUI |
| **Strict Moss Sampler** (`sampling/strict`) | `seed`, `steps` (8), `cfg` (1), `spectrum` (`off`), `lora_mode` (`merged`), `turbo` (`auto`) |
| Save Image | the `IMAGE` output |

What the node does: it reads the checkpoint path from the MODEL, turns the LoRA patches of the LoraLoader nodes into one adapter (cached by content in `artifacts/moss`), merges it into a cached int8 checkpoint (`merged`, first time about 15 s, then reused; the 3 most recently used merges stay in the cache, 5.4 GB each) or applies it as floating adapters (`runtime`, slower per image), unloads ComfyUI's models from the GPU and sends the run to a resident `MossGenerate.exe`, which keeps the model loaded until the model, LoRAs or Spectrum change (or ComfyUI exits), then decodes the latents with the connected VAE. Image `i` of a batch uses seed `seed + i` with the same noise as a ComfyUI sampler at batch size 1.

- `quality` picks steps, `cfg` and `spectrum` like `MossImage.exe --quality` below, `custom` uses the `steps`, `cfg` and `spectrum` inputs instead (workflows saved before `quality` existed load with `best`, set `custom` to keep their steps).
- 8 steps or fewer need the DMD turbo LoRA and `cfg` 1: `turbo` `auto` adds `loras/moss/dmd_turbo_lora.safetensors` at 8 steps or fewer unless a LoraLoader or the checkpoint's merge metadata already has it. Use `off` for checkpoints with a turbo LoRA baked in by someone else, and 25-50 steps with `cfg` 4 without turbo (turbo at 25 or 50 steps gives broken images).
- `cfg` above 1 guides away from a zero prompt (Strict's CFG), there is no negative input; every step then runs the model twice.
- `spectrum`: `off` or `spectrum<steps>_w<warmup>p<period>`, `spectrum8_w3p3` forecasts 3 of 8 steps.
- Only plain LoRA up/down patches are used, other MODEL patches (LoKR, model sampling, attention patches) are rejected or ignored; the latent input only gives the size (no img2img).
- Speed on the RTX 3080 at 720x1280, 8 steps, DMD merged: a queued prompt with an already loaded backend takes sampling + 1.5 s (VAE decode and handoff), the first one also loads the model (3-4 s). See Performance.md for the current sampling time.
- The resident backend keeps the Moss weights on the GPU between runs, so a changed prompt encodes with less free GPU memory in ComfyUI.

## Command line: MossImage.exe

From any folder with a UTF-8 `prompt.txt` (ComfyUI need not be running, its Python and code are used to encode):

```powershell
C:/code/GitHub/strict-lang/StrictComfy/MossImage.exe --count 10
C:/code/GitHub/strict-lang/StrictComfy/MossImage.exe --lora moss/lenovo_moss.safetensors:0.5 --lora moss/cndamiph_Candid_Amateur_iPhone_MOSS_v1.safetensors:0.31 --spectrum spectrum8_w3p3
```

`--quality <preset>` picks steps, guidance and Spectrum in one go, `--steps`, `--guidance` and `--spectrum` change a preset (`--quality faster --steps 10` samples 10 steps with `spectrum10_w2p3`). Sampling per image at 720x1280 on the RTX 3080, warm:

| `--quality` | Steps | Guidance, Spectrum | Sampling |
|---|---:|---|---:|
| `best` (default) | 8, DMD turbo LoRA | cfg 1 | 3.2 s |
| `best25steps` | 25, no turbo LoRA | cfg 2 (two model runs per step; 4 overbakes against the zero prompt Strict guides away from) | 18 s |
| `best50steps` | 50, no turbo LoRA | cfg 2 | 37 s |
| `fast` | 8, turbo | `spectrum8_w3p3` (5 of 8 steps evaluated) | 2.0 s |
| `faster` | 8, turbo | `spectrum8_w2p3` (4 of 8 evaluated) | 1.6 s |
| `fastest` | 4, turbo | `spectrum4_w2p2` (3 of 4 evaluated; 3 of 6 or 8 steps oversaturate or add noise) | 1.2 s |
| `fastest2steps` | 2, turbo | off (softer, about half the images usable; 3 steps are no faster than fastest) | 0.77 s (480x848: 0.32 s) |

`--help` lists every option with its default (`-h` is not supported):

| Option | Default |
|---|---|
| `--prompt` | `prompt.txt`; `{a woman\|a man\|}` picks one alternative per image (the empty one too), nesting works |
| `--output` | `MossImage.jpg` numbered `MossImage0001.jpg`, ...; never overwrites; `.jpg`, `.jpeg` or `.png` |
| `--count` | 1; images per loaded model, each with its own seed and prompt choices |
| `--seed` | random; reproduces a whole batch |
| `--width`, `--height` | 720, 1280 (multiples of 16) |
| `--quality` | `best`, see the table above |
| `--steps` | 8 (from `--quality`); 8 steps or fewer add `moss/dmd_turbo_lora.safetensors` at strength 1 unless a DMD entry is given (`:0` disables it) |
| `--lora` | repeat as `path[:strength]`, relative paths resolve under ComfyUI's `models/loras` |
| `--lora_mode` | `merged`: cached stochastic int8 merge of the whole stack (fast, approximate); `runtime`: floating adapters |
| `--merge_only` | prepare and print the merged checkpoint, then exit; it can be passed as `--model` later (its baked LoRAs are recorded) |
| `--spectrum` | `off` (from `--quality`); `spectrum8_w3p3` forecasts three of eight steps (changes the result) |
| `--guidance` | 1 (from `--quality`; CFG scale against a zero prompt) |
| `--skip` | `off`; `first:last` layers (of 42) every other evaluated step from step 2 on (never the last) reuses from the last full evaluation (MossSpan): `--quality best25steps --skip 6:36` samples in 12.3 instead of 17.9 s with slightly different colors; with 8 turbo steps Spectrum (`fast`, `faster`) is the better trade |
| `--jpg_quality` | 85 |
| `--json_output` | off; writes a `.json` sidecar with prompt, seed, LoRAs and timings per image |
| `--model`, `--vae`, `--te` | `moss-image-sft-dit-int8-convrot.safetensors`, `flux2-vae.safetensors`, `qwen3.5_9b_int8_convrot.safetensors` in ComfyUI's model folders |
| `--comfyui`, `--python` | `D:/ComfyUI_windows_portable/ComfyUI` (or C:, or `COMFYUI`) and its `python_embeded` |
| `--cache` | `artifacts/moss`: prompt conditioning, merged models, adapters, CUDA cache (a merged stack is a full checkpoint) |

Uncached prompts are encoded in one ComfyUI text encoder worker that exits before Strict starts. Strict samples, decodes with its own Flux 2 VAE port and writes the JPEG. `MossImage.exe dumps model lora vae` is the diagnostic mode comparing every layer and step against a `dump.py moss` folder.

Supported LoRAs: kohya `lora_down`/`lora_up` and PEFT `lora_A`/`lora_B` with optional `alpha`, names `diffusion_model.<module>`, `<module>` or `lora_unet_<module>`, fused `qkv` split into q/k/v (the ComfyUI node follows ComfyUI instead, which skips fused qkv on this model). Merged mode also changes matching floating weights, runtime mode only the layer projections; unsupported formats report an error.

## Tests

```powershell
D:/ComfyUI_windows_portable/python_embeded/python.exe test_generate.py
../Strict/strict.exe test MossGenerate.strict
../Strict/strict.exe check .
```

`test_generate.py` covers the commands and the node's LoRA conversion without a full model; `.strict` files test themselves against the tiny fixtures (`tiny*.safetensors`, written by `dump.py tiny...`).

## Comparison harness

New manifests use base weights plus the original runtime DMD for turbo8/spectrum8. Existing comparison.json files are not rewritten: update their model/lora fields or use a fresh output root before comparing. Older gallery results using the premerged DMD file are not validation of the corrected default.

`MossGenerate` generates images without the per-layer and per-step reference comparisons in `MossImage`. `MossSampling` supports the existing flow Euler schedule, ordinary CFG with zero conditioning, separate Spectrum histories for its positive/negative branches, and Rapid-MOSS data-space iteration. No error or PNG equality requirement is imposed across these methods.

`compare_moss.py` prepares identical conditioning/noise within each prompt/seed case, runs the variants and writes a local HTML gallery. Default cases: lighthouse, portrait with hands, and a cafe menu; seeds 42 and 1234; 720x1280. Edit `comparison.json` in the chosen output root to use your own prompts, seeds or settings, then prepare again.

### Run

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

### What each column means

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

### Timing

`sampling_ms` includes text projection, initial uploads, all evaluations and final latent synchronization, but excludes loading and decode/save. `denoising_ms` starts at evaluation 0 and includes its initial uploads. Neither is a resident-model steady-step-only measurement. Progress lines mark host submission; they do not individually synchronize GPU completion. Process wall time includes loading, decoding, saving and startup, across all resident passes. `generation_runs_ms` measures each complete sampling/decode/save pass (and boundary cleanup when another pass follows); `sampling_runs_ms` retains its sampler timings. Gallery cells label the last generation and the total process separately.

The harness uses a writable CUDA JIT cache under its output root, with a 4 GiB cap. A cold compiled module may add tens of seconds; warm it before comparing timings. This is a per-process setting, not a global driver change. RTX 3080 results use locked 1695 MHz graphics / 9506 MHz actual CUDA memory, with clock limits restored afterwards. Results and quality observations are in [Performance.md](Performance.md).

## ComfyUI node: Strict H3 Sampler

The same junction installs it (on the 4090 machine `C:\ComfyUI_windows_portable\ComfyUI\custom_nodes\StrictComfy`). It replaces the sampler group of an H3 workflow (RandomNoise, KSamplerSelect, BasicScheduler, BasicGuider, MiniMax H3 sigma shift, SamplerCustomAdvanced) and returns the same video and audio latent:

| Node | Settings |
|---|---|
| Load CLIP | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, type `minimax` |
| Load VAE, twice | `minimax_h3_video_vae_int8_convrot.safetensors` and `minimax_h3_audio_vae_fp32.safetensors` |
| MiniMax H3 Image to Video | prompt, width, height, length (124 = 5.2 s), optional first and last frame |
| **Strict H3 Sampler** (`sampling/strict`) | `positive` and `latent` of MiniMax H3 Image to Video, `quality` (`fast`), `seed`, `skip` (`5:45`), `sampler_name` (`euler`), `scheduler` (`simple`), optional `model` |
| VAE Decode, VAE Decode Audio | the latent with the video and the audio VAE |
| Video Combine (VideoHelperSuite) or Create Video + Save Video | the frames at 24 fps with the audio |

`quality` is the presets of `H3Video.exe --quality` (the table below), the model and its LoRAs come from the models folder. An optional `model` (Load Diffusion Model, LoraLoaderModelOnly for style LoRAs) replaces the preset's checkpoint, the preset's LoRAs (TaoMate for `fastest2steps`) still apply. The node unloads ComfyUI's models from the GPU and sends the run to a resident `H3Video.exe`, which keeps the model in host memory until the quality or model changes (or ComfyUI exits) and evicts its GPU copies after every run so ComfyUI can decode. On the 4090 with 124 frames: a run on the loaded model takes its sampling time (the table below) and about 9 s more for decoding and the mp4, the first run of a quality, model or LoRA set adds the model load (10-16 s, the LoRAs merged while loading like `--lora_mode runtime` below, about 20 s more for TaoMate and a style LoRA). Video `i` of a batch uses seed `seed + i` with ComfyUI's noise for that seed. With `models/vae_approx/taeh3.safetensors` (the per-frame tiny H3 decoder Model Preview Override uses) the node shows the denoised video of every step as an animation of every latent frame (about 7 per second in real time, decoded at its own size and scaled to at most 512 pixels), so a bad run can be cancelled early (about 0.3 s of GPU per evaluated step for 294 frames, a preview that comes while the last one still decodes is skipped).

### Against ComfyUI's H3 sampling

One `quality` setting picks the speed and the quality, where ComfyUI needs a model, a LoRA or an attention node for each and a round of tests for every combination. A 12 s clip (480x832, 294 frames, one prompt, seed 2 right after seed 1) on the 4090, the whole prompt from queueing to the saved mp4: text encoding, sampling, both VAE decodes and the same Video Combine h264 encode in every row (NVENC h265 takes a few seconds off each). ComfyUI runs with `--disable-dynamic-vram --disable-pinned-memory` and its default attention, the quality is rated by eye on these clips:

| Sampling | Model evaluations | Whole prompt | Against ComfyUI's default | Quality |
|---|---:|---:|---:|---|
| ComfyUI's H3 template default: the base model, 20 `res_multistep` steps | 20 | 291.0 s | 1x | ★★★★★ |
| ComfyUI, the 768p turbo 8-step checkpoint | 8 | 148.9 s (125.6 s on 2026-10-27) | 2.0x | ★★★★ |
| ComfyUI, the turbo checkpoint with 4 steps | 4 | 71.5 s | 4.1x | ★★ |
| ComfyUI, the base model with the TaoMate 3-step LoRA | 3 | 62.4 s | 4.7x | ★ breaks down, most in fast motion |
| `best` | 8 | 85.3 s | 3.4x | ★★★★ |
| `fast` (the node's default) | 8 | 58.4 s | 5.0x | ★★★★ |
| `faster` | 5 | 43.2 s | 6.7x | ★★★★ (not rated yet, assumed like `fast`) |
| `fastest4steps` | 4 | 38.2 s | 7.6x | ★★★ |
| `fastest2steps` (TaoMate) | 2 | 27.1 s | 10.7x | ★ |

With the same model and steps the node is 1.7x (`best` against ComfyUI's 8 turbo steps, 1.5x against the 125.6 s of 2026-10-27) and 1.9x (`fastest4steps` against 4 turbo steps) faster. The turbo presets use the 768p turbo 8-step checkpoint, almost as good as the 600 EMA larry turbo LoRA that wins most 8-step comparisons; a `model` input (Load Diffusion Model with any turbo LoRA through LoraLoaderModelOnly) replaces the checkpoint at the same speed, the presets add no turbo LoRA of their own on top (only `fastest2steps` adds TaoMate).

ComfyUI's attention speed-ups per step of a 10 s clip (243 frames, the turbo checkpoint, [Performance.md](Performance.md)): default attention 9.62-9.73 s, SageAttention 6.37-6.42 s, comfy_kitchen int8 5.24-5.29 s (the one that keeps the quality), Sol-Attn 3.34-3.49 s; Strict `best` 5.99-6.09 s, `fast` 3.68-3.74 s (routed attention like Sol-Attn). ComfyUI's Spectrum node failed on this ComfyUI.

## H3 videos with sound from a prompt file (RTX 4090)

`H3Video.exe --quality <preset>` picks the model, steps, attention and Spectrum in one go; `--model`, `--steps`, `--routing`, `--spectrum`, `--skip` and `--lora` change a preset (`--sampler` and `--scheduler` apply to all) (`--quality faster --steps 10` samples 10 steps with `spectrum10_w1p2`), its LoRAs come before the given ones. Sampling times for the default 124 frames on a 4090, warm:

| `--quality` | Model, LoRA | Steps | Attention, Spectrum | Sampling |
|---|---|---:|---|---:|
| `best` (default) | turbo INT8 8-step | 8 | exact | 14.0 s |
| `fast` | turbo | 8 | `--routing 1` | 11.1 s (14.8 s with `--skip off`) |
| `faster` | turbo | 8 | `--routing 1`, `spectrum8_w1p2` (5 of 8 steps evaluated) | 7.1 s |
| `fastest4steps` | turbo | 4 | `--routing 1` | 6.4 s |
| `fastest2steps` | base + TaoMate 3-step LoRA (strength 1) | 2 | `--routing 1` | 4.5 s |

Every preset runs with `--skip 5:45` (below), the times are the second of two videos of the fox prompt with one model load (2026-10-28, before: 22, 16, 11, 9 and 6 s; `fastest3steps` is gone, 3 steps skip nothing and took as long as `fastest4steps`).

It has the options of `MossImage.exe` above: `prompt.txt` with `{a|b|}` choices, `--count` resident batches with distinct seeds, `--seed`, repeated `--lora path:strength`, `--lora_mode`, `--merge_only`, `--spectrum`, numbered outputs that are never overwritten and `--json_output`. It adds `--length` (frames at 24 fps, snapped up to 17k+5 like ComfyUI, default 124 = 5.2 s) and `--shift` (12, the audio follows with shift 3), and has no `--guidance` or `--jpg_quality`. Defaults: `--quality best`, 480x832, `H3Video.mp4`, `minimax_h3_video_vae_int8_convrot`, `minimax_h3_audio_vae_fp32` and the `qwen3vl_32b_minimax_h3_nvfp4_awq` text encoder under `--comfyui`.

```powershell
cd C:\StrictDumps\videos
C:\code\GitHub\strict-lang\StrictComfy\H3Video.exe --count 4
C:\code\GitHub\strict-lang\StrictComfy\H3Video.exe --quality faster --count 4
C:\code\GitHub\strict-lang\StrictComfy\H3Video.exe --spectrum spectrum8_w1p2
C:\code\GitHub\strict-lang\StrictComfy\H3Video.exe --image first.png --count 4
C:\code\GitHub\strict-lang\StrictComfy\H3Video.exe --model minimax_h3_fl2va_pruned_int8_convrot.safetensors --steps 3 --lora h3/minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors:0.8
C:\code\GitHub\strict-lang\StrictComfy\H3Video.exe --model minimax_h3_fl2va_pruned_int8_convrot.safetensors --steps 4 --lora h3/minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors:0.6 --lora_mode runtime
```

The prompts are encoded in one ComfyUI worker (cached per prompt under `artifacts/h3`, the same Qwen3-VL states and token tags as ComfyUI's H3 node), the noise is ComfyUI's for the same seed, `H3Video.exe` samples every job with one model load, then one ComfyUI worker decodes all latents with the video and audio VAEs and ffmpeg writes `H3Video0001.mp4`, `H3Video0002.mp4`, ... `--image first.png` (and `--last_image last.png`) makes it image to video like ComfyUI's MiniMaxH3ImageToVideo node: the worker encodes the frames with the video VAE and the prompt together with them (cached per prompt, images and size), the launcher adds ComfyUI's noise augmented condition rows for each seed; the conditioning matches ComfyUI's exactly.

LoRA names are matched like ComfyUI: `diffusion_model.<module>`, `<module>` and kohya's `lora_unet_<module with _>`, `lora_A`/`lora_B` or `lora_down`/`lora_up` with an optional `alpha`. `--lora_mode runtime` (default, like ComfyUI's LoRA loaders) lets `H3Video.exe` merge the block linears and the token refiner at every load (`Safetensors.Tuned`, no cached file, the same step times; about 20 s more load for TaoMate with a small style LoRA, 110 s for rank 384, 140 s for a stack of rank 300-800); its load line counts the applied tensors. `--lora_mode merged` merges every matching tensor once into a cached int8 model with `merge.py` (4 minutes and 56 GB of RAM): the block linears, the dense token refiner, adaln and final layer, a full 21 GB file per model, LoRA and strength combination in `artifacts/h3`, so delete old ones after trying many strengths. FastH3's `diff`/`diff_b` tensors are applied by neither.

`--spectrum spectrum<steps>_w<warmup>p<period>`: after `warmup` actual steps every `period`-th step is actual, the others are forecast, the last is always actual; `spectrum8_w1p2` is A F A F A F A A (5 of 8 transformer evaluations). Moss uses `spectrum8_w3p3`.

`--skip first:last` (default `5:45` for every preset, `off` for none, the node's `skip` input): every other evaluated step after the first two (never the last) reuses what blocks first to last of the 50 added in the last full evaluation instead of computing them, the hidden state before the span plus that difference (one hidden-sized list, 0.8 GB at 480x832 and 10 s). 8 routed turbo steps of a 10 s clip on a 4090 skip steps 2, 4 and 6: `15:35` 11 %, `10:40` 19 %, `5:45` 26 % faster (a skipped step 2.3, 1.6 and 0.8 s instead of 3.6 s), the videos show the same composition, sharpness and motion. Steps with Spectrum forecasts are not evaluated and do not count, 3 or fewer steps skip nothing.

`--sampler euler|res_multistep|er_sde` and `--scheduler simple|sgm_uniform` (default `euler` and `simple`, the node's `sampler_name` and `scheduler`) are ComfyUI's k-diffusion samplers and schedulers (`Solver.strict`, `Euler.Scheduled`): `res_multistep` is second order (the default of the H3 workflows), `er_sde` is stochastic with new noise every step but the last (from the seed's generator after the initial noise, ComfyUI draws it on the GPU, so the video differs) and starts below sigma 1 like ComfyUI. Video and audio each run the sampler on their own sigmas, which is ComfyUI's result for `euler` and close to it for the others. Every sampler evaluates the model once per step, so all take as long.

### Diagnostic mode: positional arguments

`dump.py h3 --runs 0` writes the inputs of a prompt and seed without sampling in ComfyUI: `conditioning.safetensors` (Qwen3-VL states and token tags) and `steps.safetensors` (the noise as `video.0`/`audio.0`, `sigmas` of the default 8 steps); a full `dump.py h3` run (also with `--lora name:strength`) adds ComfyUI's per step references, which the positional mode compares against. `dump.py h3decode` decodes every `<name>-latent.safetensors` in a folder with ComfyUI's VAEs into `<name>-video.mp4` (plus first, middle and last frame PNGs).

```powershell
cd C:\code\GitHub\strict-lang\StrictComfy
$python = "C:\ComfyUI_windows_portable\python_embeded\python.exe"
$models = "C:\ComfyUI_windows_portable\ComfyUI\models"
& $python dump.py h3 --runs 0 --out C:\StrictDumps\fox --prompt "A red fox trotting through fresh snow in a pine forest" --seed 7
$env:STRICT_GPU = "all"
.\H3Video.exe C:\StrictDumps\fox "$models\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors" taomate3 "$models\loras\h3\minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors=1" 3
.\H3Video.exe C:\StrictDumps\fox "$models\diffusion_models\Minimax_H3_FL2VA_PRUNED_Turbo_8step_v1.0_768p_comfyui_INT8_.safetensors" spectrum8 '""' 0 12 2
& $python dump.py h3decode --out C:\StrictDumps\fox
```

Arguments after the executable: `dumps model name loras steps shift period warmup degree routing first last sampler scheduler`, defaults `strict "" 0 12 0 1 1 0 0 0 0 0`.

- `name`: the output is `<name>-latent.safetensors` (one F32 tensor `latent`: the video, then the audio numbers).
- `loras`: `path=strength` entries separated by `;`, a path without `=` has strength 1, an empty argument for none (`'""'` in Windows PowerShell 5.1, which drops `""`). Keys `[diffusion_model.]blocks.N.{attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2}.{lora_A,lora_B,lora_down,lora_up}.weight` with an optional `alpha` (scale alpha / rank, 1 without, times the strength), the same for `token_refiner.blocks.N`. Several LoRAs stack by rank. The int8 convrot block weights get the update merged at load like ComfyUI does (dequantized, the rotated update added, requantized per row with stochastic rounding; `Safetensors.Tuned` in ../Strict), so steps cost the same as without LoRA; the dense token refiner applies them per call. The load line reports how many of each file's tensors were applied: adaln, patch projection and final layer LoRAs and FastH3's `diff`/`diff_b` tensors are not applied yet.
- `steps`: 0 takes the count of the folder's `sigmas`; the schedule is ComfyUI's simple flow schedule `shift t / (1 + (shift - 1) t)` (8 steps match ComfyUI exactly), `shift` is the video flow shift and the audio sigmas follow from it with shift 3 like ComfyUI's MiniMax H3 shift node. Per step errors against ComfyUI print only when the folder has its references and the step count matches.
- `period`: Spectrum, 0 is off. After `warmup` actual steps every `period`-th step is actual, the others are forecast from the last actual ones, the last step is always actual: 8 steps with period 2 are A F A F A F A A (5 transformer evaluations), period 3 skips two of three. `degree` is the Chebyshev degree of the fit (ridge 0.1, blended half with the line through the last two, `SpectralFit`), it keeps `degree + 1` anchors; with one anchor the forecast holds it. H3 forecasts its output, not its features ([Spectrum.md](Spectrum.md)).
- `routing`: `--routing` (0 attends densely), `first` and `last`: the blocks of `--skip` (0 and 0 for none).
- `sampler`: 0 `euler`, 1 `res_multistep`, 2 `er_sde` (its noise in the job as `video.noise.<step>` and `audio.noise.<step>`, none is no noise), `scheduler`: 0 `simple`, 1 `sgm_uniform`.

Loading the 21 GB model takes 14-20 s, merging adds about 20 s for TaoMate (rank 4-21) and 80 s for the turbo 8-step LoRA (rank 128-384). The first step of a fresh process uploads all weights once (about 12 s more), the H3 reference folder hides that in its per-layer comparison before sampling. Timings and quality numbers are in [Performance.md](Performance.md).
