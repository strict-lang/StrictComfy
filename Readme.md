# StrictComfy

ComfyUI's Moss (image DiT) and MiniMax H3 (video and audio DiT) running in [Strict](../Strict): `MossImage.exe` and `H3Video.exe` make images and videos with sound from a prompt file on their own (no Python or ComfyUI, only the model files), the Strict Moss Sampler and Strict H3 Sampler nodes sample in Strict inside a ComfyUI workflow. Spectrum forecasting is in [Spectrum.md](Spectrum.md).

## Build

From this folder, with `../Strict/strict.exe`:

```powershell
../Strict/strict.exe build MossImage.strict
../Strict/strict.exe build H3Video.strict
```

`MossImage.exe` and `H3Video.exe` are the commands and also the backends of the ComfyUI nodes, which start them from this folder with `-` as the first argument (then they read one job folder per input line and keep the model loaded). To ship the nodes, copy the Python files with both executables and the GPU files next to each (`Name.fatbin` and `Name.runtime.fatbin` with the compiled kernels for sm_75, sm_80, sm_86 and sm_89, `Name.ptx`, `Name.runtime.ptx` and the `.older.ptx` variants for other GPUs). The Python files serve ComfyUI only: `__init__.py` and `generate.py` are the nodes, `merge.py` the Moss node's cached LoRA merges, `dump.py` writes ComfyUI references and the tiny test fixtures. Other GPUs compile the PTX on their first run (tens of seconds), later runs use the CUDA cache in `artifacts/`.

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

What the node does: it reads the checkpoint path from the MODEL, turns the LoRA patches of the LoraLoader nodes into one adapter (cached by content in `artifacts/moss`), merges it into a cached int8 checkpoint (`merged`, first time about 15 s, then reused; the 3 most recently used merges stay in the cache, 5.4 GB each) or applies it as floating adapters (`runtime`, slower per image), unloads ComfyUI's models from the GPU and sends the run to a resident `MossImage.exe`, which keeps the model loaded until the model, LoRAs or Spectrum change (or ComfyUI exits), then decodes the latents with the connected VAE. Image `i` of a batch uses seed `seed + i` with the same noise as a ComfyUI sampler at batch size 1.

- `quality` picks steps, `cfg` and `spectrum` like `MossImage.exe --quality` below, `custom` uses the `steps`, `cfg` and `spectrum` inputs instead (workflows saved before `quality` existed load with `best`, set `custom` to keep their steps).
- 8 steps or fewer need the DMD turbo LoRA and `cfg` 1: `turbo` `auto` adds `loras/moss/dmd_turbo_lora.safetensors` at 8 steps or fewer unless a LoraLoader or the checkpoint's merge metadata already has it. Use `off` for checkpoints with a turbo LoRA baked in by someone else, and 25-50 steps with `cfg` 4 without turbo (turbo at 25 or 50 steps gives broken images).
- `cfg` above 1 guides away from a zero prompt (Strict's CFG), there is no negative input; every step then runs the model twice.
- `spectrum`: `off` or `spectrum<steps>_w<warmup>p<period>`, `spectrum8_w3p3` forecasts 3 of 8 steps.
- Only plain LoRA up/down patches are used, other MODEL patches (LoKR, model sampling, attention patches) are rejected or ignored; the latent input only gives the size (no img2img).
- Speed on the RTX 3080 at 720x1280, 8 steps, DMD merged: a queued prompt with an already loaded backend takes sampling + 1.5 s (VAE decode and handoff), the first one also loads the model (3-4 s).
- The resident backend keeps the Moss weights on the GPU between runs, so a changed prompt encodes with less free GPU memory in ComfyUI.

## Command line: MossImage.exe

From any folder with a UTF-8 `prompt.txt`, without ComfyUI or Python (only the model files):

```powershell
C:/code/GitHub/strict-lang/StrictComfy/MossImage.exe --count 10
C:/code/GitHub/strict-lang/StrictComfy/MossImage.exe --lora moss/lenovo_moss.safetensors:0.5 --lora moss/cndamiph_Candid_Amateur_iPhone_MOSS_v1.safetensors:0.31 --spectrum spectrum8_w3p3
C:/code/GitHub/strict-lang/StrictComfy/MossImage.exe --model moss.safetensors --vae flux2-vae.safetensors --text_encoder qwen3.5_9b_int8_convrot.safetensors
```

`--quality <preset>` picks steps, guidance and Spectrum in one go, `--steps`, `--guidance` and `--spectrum` change a preset. Sampling per image at 720x1280 on the RTX 3080, warm:

| `--quality` | Steps | Guidance, Spectrum | Sampling |
|---|---:|---|---:|
| `best` (default) | 8, DMD turbo LoRA | cfg 1 | 3.2 s |
| `best25steps` | 25, no turbo LoRA | cfg 2 (two model runs per step; 4 overbakes against the zero prompt Strict guides away from) | 18 s |
| `best50steps` | 50, no turbo LoRA | cfg 2 | 37 s |
| `fast` | 8, turbo | `spectrum8_w3p3` (5 of 8 steps evaluated) | 2.0 s |
| `faster` | 8, turbo | `spectrum8_w2p3` (4 of 8 evaluated) | 1.6 s |
| `fastest` | 4, turbo | `spectrum4_w2p2` (3 of 4 evaluated; 3 of 6 or 8 steps oversaturate or add noise) | 0.93 s |
| `fastest2steps` | 2, turbo | off (softer, about half the images usable; 3 steps are no faster than fastest) | 0.77 s (480x848: 0.32 s) |

`--help` lists every option with its default and description (the options are the members of `MossFiles` and `MossOptions`, parsed by Strict's generated `Parsed`):

| Option | Default |
|---|---|
| `--model`, `--vae`, `--text_encoder` | `moss-image-sft-dit-int8-convrot.safetensors`, `flux2-vae.safetensors`, `qwen3.5_9b_int8_convrot.safetensors`: a path, or a name in the models folder; the tokenizer folder `qwen35_tokenizer` is used from next to the text encoder, else from ComfyUI's `comfy/text_encoders` |
| `--models` | `D:/ComfyUI_windows_portable/ComfyUI/models` (or C:), for files and LoRAs given by name |
| `--prompt` | `prompt.txt`; `{a woman\|a man\|}` picks one alternative per image (the empty one too), nesting works |
| `--quality` | `best`, see the table above |
| `--steps`, `--guidance`, `--spectrum` | from `--quality`; 8 steps or fewer add `moss/dmd_turbo_lora.safetensors` unless a DMD entry is given (`:0` disables it); `spectrum8_w3p3` forecasts three of eight steps |
| `--skip` | `off`; `first:last` layers (of 41) every other evaluated step from step 2 on (never the last) reuses from the last full evaluation (MossSpan): `--quality best25steps --skip 6:36` samples in 12.3 instead of 17.9 s with slightly different colors |
| `--lora` | repeat as `path[:strength]`, merged into the int8 weights at load (stochastic rounding like ComfyUI) |
| `--count` | 1; images per run, each with its own seed and prompt choices |
| `--seed` | -1 (random); the first image's seed, the others follow from it |
| `--width`, `--height` | 720, 1280 (multiples of 16) |
| `--output` | `MossImage.jpg` numbered `MossImage0001.jpg`, ...; never overwrites; `.jpg` (quality 85) or `.png` |
| `--json_output` | off; writes a `.json` sidecar with seed, prompt, steps, guidance, Spectrum, model and LoRAs per image |

Strict does everything in one process: it encodes every distinct prompt with its Qwen3.5 encoder (then frees it), merges the LoRAs, makes the noise like `torch.randn` of the seed, samples, decodes with its Flux 2 VAE port and writes the image. Loading takes about 3 s for the encoder and 3 s for Moss plus the LoRA merge (on the CPU for now, about 14 s with the turbo LoRA). `MossImage.exe dumps model lora vae` is the diagnostic mode comparing every layer and step against a `dump.py moss` folder.

Supported LoRAs: kohya `lora_down`/`lora_up` and PEFT `lora_A`/`lora_B` with optional `alpha`, names `diffusion_model.<module>`, `<module>` or `lora_unet_<module>`, fused `qkv` split into q/k/v (the ComfyUI node follows ComfyUI instead, which skips fused qkv on this model). Merged mode also changes matching floating weights, runtime mode only the layer projections; unsupported formats report an error.

## Tests

```powershell
D:/ComfyUI_windows_portable/python_embeded/python.exe test_generate.py
../Strict/strict.exe test MossImage.strict
../Strict/strict.exe check .
```

`test_generate.py` covers the nodes' presets, LoRA conversion, cached merges and backend jobs without a full model; `.strict` files test themselves against the tiny fixtures (`tiny*.safetensors`, written by `dump.py tiny...`).

## ComfyUI node: Strict H3 Sampler

The same junction installs it (on the 4090 machine `C:\ComfyUI_windows_portable\ComfyUI\custom_nodes\StrictComfy`). It replaces the sampler group of an H3 workflow (RandomNoise, KSamplerSelect, BasicScheduler, BasicGuider, MiniMax H3 sigma shift, SamplerCustomAdvanced) and both VAE decodes, it returns the images and the audio. **Strict H3 Sampler (Latent)** has the same settings and replaces only the sampler group: it returns the video and audio latent for latent nodes (upscaling, a second pass) before VAE Decode and VAE Decode Audio.

| Node | Settings |
|---|---|
| Load CLIP | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, type `minimax` |
| Load VAE | `minimax_h3_video_vae_int8_convrot.safetensors` for MiniMax H3 Image to Video (and VAE Decode after the latent variant, VAE Decode Audio takes `minimax_h3_audio_vae_fp32.safetensors`) |
| MiniMax H3 Image to Video | prompt, width, height, length (124 = 5.2 s), optional first and last frame |
| **Strict H3 Sampler** (`sampling/strict`) | `positive` and `latent` of MiniMax H3 Image to Video, `quality` (`fast`), `seed`, `skip` (`5:45`), `sampler_name` (`euler`), `scheduler` (`simple`), optional `model` |
| Create Video + Save Video, or Video Combine (VideoHelperSuite) | the images at 24 fps with the audio |

`quality` is the presets of `H3Video.exe --quality` (the table below), the model and its LoRAs come from the models folder. An optional `model` (Load Diffusion Model, LoraLoaderModelOnly for style LoRAs) replaces the preset's checkpoint, the preset's LoRAs (TaoMate for `fastest2steps`) still apply. The node unloads ComfyUI's models from the GPU and sends the run to a resident `H3Video.exe`, which keeps the model and both VAEs in host memory until the quality or model changes (or ComfyUI exits), evicts the model from the GPU after sampling, decodes the video and the audio like VAE Decode and VAE Decode Audio (`H3Vae`, `H3Audio`: the frames come back as 8 bit RGB, as from a loaded video) and evicts again, so ComfyUI has the GPU for its text encoder. The latent variant shares that process. On the 4090 with 124 frames: a run on the loaded model takes its sampling time (the table below, about 2.6 s more in the node with the previews and the model uploaded again after the decode), 4.9-5.3 s for decoding and 2.3-2.9 s for Save Video (24.3 s for `best`, 25.7 s with the latent variant and ComfyUI's warm decodes, 35.5 s for ComfyUI's first decode of a process), the first run of a quality, model or LoRA set adds the model load (10-16 s, the LoRAs merged while loading like `H3Video.exe --lora` below, about 20 s more for TaoMate and a style LoRA). Video `i` of a batch uses seed `seed + i` with ComfyUI's noise for that seed. With `models/vae_approx/taeh3.safetensors` (the per-frame tiny H3 decoder Model Preview Override uses) the node shows the denoised video of every step as an animation of every latent frame (about 7 per second in real time, decoded at its own size and scaled to at most 512 pixels), so a bad run can be cancelled early (about 0.3 s of GPU per evaluated step for 294 frames, a preview that comes while the last one still decodes is skipped).

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

ComfyUI's attention speed-ups per step of a 10 s clip (243 frames, the turbo checkpoint): default attention 9.62-9.73 s, SageAttention 6.37-6.42 s, comfy_kitchen int8 5.24-5.29 s (the one that keeps the quality), Sol-Attn 3.34-3.49 s; Strict `best` 5.99-6.09 s, `fast` 3.68-3.74 s (routed attention like Sol-Attn). ComfyUI's Spectrum node failed on this ComfyUI.

## Command line: H3Video.exe

From any folder with a UTF-8 `prompt.txt`, without ComfyUI or Python (only the model files and the Qwen3-VL tokenizer files in ComfyUI's `comfy/text_encoders/qwen25_tokenizer`, next to the models folder):

```powershell
cd C:\StrictDumps\videos
C:\code\GitHub\strict-lang\StrictComfy\H3Video.exe --count 4
C:\code\GitHub\strict-lang\StrictComfy\H3Video.exe --quality faster --count 4 --sampler res_multistep
C:\code\GitHub\strict-lang\StrictComfy\H3Video.exe --quality fastest2steps --length 243 --seed 7
C:\code\GitHub\strict-lang\StrictComfy\H3Video.exe --quality fast --skip off --scheduler beta
C:\code\GitHub\strict-lang\StrictComfy\H3Video.exe --image first.png --last_image last.jpg
```

`--quality <preset>` picks the model, steps, attention and Spectrum in one go, `--steps`, `--skip` and `--lora` change a preset (`--quality faster --steps 10` samples 10 steps with every other one forecast), the LoRAs of a preset come before the given ones. Per video at 480x832 on the RTX 4090, warm (the second video of `--count 2` with a cached prompt; the video adds the VAE decode and the mp4 to the sampling):

| `--quality` | Model, LoRA | Steps | Attention, Spectrum | 5 s (124 frames): sampling, video | 10 s (243 frames): sampling, video |
|---|---|---:|---|---:|---:|
| `best` (default) | turbo INT8 8-step | 8 | exact | 13.7 s, 19.0 s | 34.6 s, 44.7 s |
| `fast` | turbo | 8 | routed like Sol-Attn | 10.7 s, 16.2 s | 21.5 s, 31.8 s |
| `faster` | turbo | 8 | routed, `spectrum8_w1p2` (5 of 8 steps evaluated) | 6.6 s, 11.9 s | 14.1 s, 24.6 s |
| `fastest4steps` | turbo | 4 | routed | 6.6 s, 12.5 s | 13.3 s, 24.3 s |
| `fastest2steps` | base + TaoMate 3-step LoRA (strength 1) | 2 | routed | 4.2 s, 10.0 s | 9.4 s, 20.1 s |

A run of one video also loads the model (6-11 s, 14-22 s with the TaoMate merge) and the VAEs (2-3 s): 23.7 s (`faster`) to 36.3 s (`best`) for 5 s, 37.5 to 58.6 s for 10 s. A new prompt adds the text encoder (29 s to load, 7 s per prompt). Every preset runs with `--skip 5:45` (below). ComfyUI's default H3 workflow (Image to Video (MiniMax H3) with its Lightning LoRA: the base model with the turbo 8-step LoRA, 8 `res_multistep` steps) takes 38.5 s per 5 s video and 118.4 s per 10 s video.

`--help` lists every option with its default and description (the options are the members of `H3Options` and `H3Files`, parsed by Strict's generated `Parsed`):

| Option | Default |
|---|---|
| `--model`, `--vae`, `--audio_vae`, `--text_encoder` | the model of `--quality`, `minimax_h3_video_vae_int8_convrot.safetensors`, `minimax_h3_audio_vae_fp32.safetensors`, `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`: a path, or a name in the models folder |
| `--models` | `D:` or `C:/ComfyUI_windows_portable/ComfyUI/models`, for files and LoRAs given by name |
| `--prompt` | `prompt.txt`; `{a\|b\|}` picks one alternative per video (the empty one too), nesting works |
| `--quality` | `best`, see the table above |
| `--steps`, `--skip` | from `--quality`; `--skip` below |
| `--sampler`, `--scheduler` | `euler`, `simple`: ComfyUI's `sampler_name` and `scheduler`, below |
| `--lora` | repeat as `path[:strength]` (relative paths under the models folder's `loras`), merged at load |
| `--count` | 1; videos per run with one model load, each with its own seed and prompt choices |
| `--seed` | -1 (random); the first video's seed, the others follow from it (reproduces a whole batch) |
| `--width`, `--height` | 480, 832 (multiples of 16) |
| `--length` | 124 frames at 24 fps (5.2 s), snapped up to 17k+5 like ComfyUI |
| `--image`, `--last_image` | none; png or jpg first and last frame for image to video |
| `--output` | `H3Video.mp4` numbered `H3Video0001.mp4`, ...; never overwrites |

Strict does everything in one process: it expands the prompt choices, draws the seeds and the noise (MersenneTwister reproduces torch's generator: ComfyUI's noise for the same seed whenever the video latent holds a multiple of 16 numbers, the per step noise of `er_sde`, `euler_ancestral`, `dpmpp_2m_sde` and `lcm` comes from its own seeds), encodes the uncached prompts with `H3Encoder` (Qwen3-VL 32B NVFP4) before the model loads (cached per prompt, encoder and images under `artifacts/h3`, the same states and token tags as ComfyUI's H3 node), samples every video with one model load, decodes the video and the audio of every video after the last one (`H3Vae`, `H3Audio`) and writes the numbered `.mp4` files with the installed ffmpeg (libx264 crf 18 and aac 192k). `--image` and `--last_image` make image to video like ComfyUI's MiniMax H3 Image to Video node: the first image is stretched to the video size, the last one cropped to its aspect first, both with Pillow's Lanczos (`Lanczos.strict`, the same bytes), `H3Picture` encodes them with the video VAE before the model loads, every video gets them as noise augmented keyframe rows at its first and last frame (seeded like ComfyUI's) and the images go into the Qwen3-VL context like ComfyUI's (`<Picture 1>: ` and `<Picture 2>: ` before the prompt, about 16 s more encoding for two images).

LoRA names are matched like ComfyUI: `diffusion_model.<module>`, `<module>` and kohya's `lora_unet_<module with _>`, `lora_A`/`lora_B` or `lora_down`/`lora_up` with an optional `alpha`. `H3Video.exe` merges the block linears and the token refiner at every load like ComfyUI's LoRA loaders (`Safetensors.Tuned`, no cached file, the same step times; about 20 s more load for TaoMate with a small style LoRA, 110 s for rank 384, 140 s for a stack of rank 300-800); its load line counts the applied tensors. LoRAs and FastH3's `diff`/`diff_b` tensors on the float weights (patch and condition projections, final layer, norms, adaln biases) are added at load too. adaln LoRAs trained on the full form (2688 inputs) don't fit the curve form of every H3 checkpoint (8 inputs) and are skipped, as ComfyUI does ("SHAPE MISMATCH").

Spectrum (`--quality faster`): after `warmup` actual steps every `period`-th step is actual, the others are forecast, the last is always actual; `spectrum8_w1p2` is A F A F A F A A (5 of 8 transformer evaluations). Moss uses `spectrum8_w3p3`.

`--skip first:last` (default `5:45` for every preset, `off` for none, the node's `skip` input): every other evaluated step after the first two (never the last) reuses what blocks first to last of the 50 added in the last full evaluation instead of computing them, the hidden state before the span plus that difference (one hidden-sized list, 0.8 GB at 480x832 and 10 s). 8 routed turbo steps of a 10 s clip on a 4090 skip steps 2, 4 and 6: `15:35` 11 %, `10:40` 19 %, `5:45` 26 % faster (a skipped step 2.3, 1.6 and 0.8 s instead of 3.6 s), the videos show the same composition, sharpness and motion. Steps with Spectrum forecasts are not evaluated and do not count, 3 or fewer steps skip nothing.

`--sampler` and `--scheduler` (default `euler` and `simple`, the node's `sampler_name` and `scheduler`) are the ComfyUI samplers and schedulers H3 and video workflows use (counted in 170 local workflows and the Wan, LTX and Hunyuan templates), ported to `Solver.strict` and `Euler.Schedule` and checked against ComfyUI's code: samplers `euler`, `res_multistep` (second order, the default of the H3 templates), `er_sde`, `euler_ancestral`, `dpmpp_2m_sde`, `gradient_estimation` and `lcm`; schedulers `simple`, `sgm_uniform`, `normal`, `beta`, `beta57` (RES4LYF's beta with alpha 0.5 and beta 0.7), `linear_quadratic` and `hyperflow` (the 8 step grid of the HyperFlow LoRA, shifted like its node). HyperFlow's backbone runs as `--model minimax_h3_fl2va_pruned_int8_convrot.safetensors --lora <models>/hyperflow/custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors --scheduler hyperflow --skip off`: its fc1 rows come as `[value; gate]` and are swapped at load like the `comfyui-hyperflow-h3` node does, the node's two-time curve refit is not ported. `er_sde`, `euler_ancestral`, `dpmpp_2m_sde` and `lcm` add new noise every step but the last (from the seed's generator after the initial noise, ComfyUI draws it on the GPU, so the video differs), `er_sde` and `dpmpp_2m_sde` start below sigma 1 like ComfyUI. Samplers that evaluate the model more than once per step (`heun`, `dpmpp_sde`) and the large `uni_pc` and `sa_solver` are not ported. Video and audio each run the sampler on their own sigmas, which is ComfyUI's result for `euler` and close to it for the others. Every sampler evaluates the model once per step, so all take as long.

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

Arguments after the executable: `dumps model name loras steps shift period warmup degree routing first last sampler scheduler`, defaults `strict "" 0 12 0 1 1 0 0 0 0 0`. With `-` for `dumps` (the nodes' resident mode) it reads one job folder per input line and takes two more arguments, `vae audio_vae`: the video and the audio VAE, loaded once.

- `name`: the output is `<name>-latent.safetensors` (one F32 tensor `latent`: the video, then the audio numbers). In the resident mode a job with a `decode` tensor also gets `<name>-frames.safetensors` once all jobs of the folder are sampled: `frames` (the RGB frames in [0, 1], row by row) and `audio` (the waveform normalized like VAE Decode Audio, 32 kHz, the left channel, then the right).
- `loras`: `path=strength` entries separated by `;`, a path without `=` has strength 1, an empty argument for none (`'""'` in Windows PowerShell 5.1, which drops `""`). Keys `[diffusion_model.]blocks.N.{attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2}.{lora_A,lora_B,lora_down,lora_up}.weight` with an optional `alpha` (scale alpha / rank, 1 without, times the strength), the same for `token_refiner.blocks.N`. Several LoRAs stack by rank. The int8 convrot block weights get the update merged at load like ComfyUI does (dequantized, the rotated update added, requantized per row with stochastic rounding; `Safetensors.Tuned` in ../Strict), so steps cost the same as without LoRA; the dense token refiner applies them per call. LoRAs, `.diff` and `.diff_b` (FastH3) on the float weights are added to them at load, adaln LoRAs of the full form's width are skipped like in ComfyUI. The load line reports how many of each file's tensors were applied.
- `steps`: 0 takes the count of the folder's `sigmas`; the schedule is ComfyUI's simple flow schedule `shift t / (1 + (shift - 1) t)` at ComfyUI's whole timesteps of 1000 (any step count matches ComfyUI, before only those dividing 1000), `shift` is the video flow shift and the audio sigmas follow from it with shift 3 like ComfyUI's MiniMax H3 shift node. Per step errors against ComfyUI print only when the folder has its references and the step count matches.
- `period`: Spectrum, 0 is off. After `warmup` actual steps every `period`-th step is actual, the others are forecast from the last actual ones, the last step is always actual: 8 steps with period 2 are A F A F A F A A (5 transformer evaluations), period 3 skips two of three. `degree` is the Chebyshev degree of the fit (ridge 0.1, blended half with the line through the last two, `SpectralFit`), it keeps `degree + 1` anchors; with one anchor the forecast holds it. H3 forecasts its output, not its features ([Spectrum.md](Spectrum.md)).
- `routing`: `--routing` (0 attends densely), `first` and `last`: the blocks of `--skip` (0 and 0 for none).
- `sampler`: 0 `euler`, 1 `res_multistep`, 2 `er_sde`, 3 `euler_ancestral`, 4 `dpmpp_2m_sde`, 5 `gradient_estimation`, 6 `lcm` (the noise of 2, 3, 4 and 6 in the job as `video.noise.<step>` and `audio.noise.<step>`, none is no noise), `scheduler`: 0 `simple`, 1 `sgm_uniform`, 2 `normal`, 3 `beta`, 4 `beta57`, 5 `linear_quadratic`, 6 `hyperflow` (8 steps only).

Loading the 21 GB model takes 14-20 s, merging adds about 20 s for TaoMate (rank 4-21) and 80 s for the turbo 8-step LoRA (rank 128-384). The first step of a fresh process uploads all weights once (about 12 s more), the H3 reference folder hides that in its per-layer comparison before sampling.
