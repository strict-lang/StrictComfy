# StrictComfy

ComfyUI nodes that sample Moss images and MiniMax H3 videos with sound in native [Strict](https://github.com/strict-lang/Strict) programs on your NVIDIA GPU:

- **Strict Moss Sampler**: Moss images. `best` gives the same images as ComfyUI's sampler, faster; the fast presets skip work and stay close.
- **Strict H3 Sampler**: MiniMax H3 videos with sound, sampled and decoded. **Strict H3 Sampler (Latent)** returns the latents instead.

All three are in the node menu under `sampling/strict`. The same programs also run without ComfyUI, see [Command line](#command-line).

## Requirements

- Windows or Linux, 64 bit
- An NVIDIA GPU, RTX 20 series or newer, with a current driver
- ComfyUI with the models of its Moss and MiniMax H3 templates
- Memory for H3: fastest with 32 GB of VRAM, fast with 24 GB of VRAM plus 30 GB of RAM. With less VRAM it still runs, clearly slower, and may need more RAM (40-50 GB) to swap; a 10 GB RTX 3080 works. int4 or GGUF models run on really low VRAM cards, but much slower and at lower quality. Moss needs much less.

## Install

In ComfyUI-Manager search for **StrictComfy**, install it and restart ComfyUI. This puts a clone of this repository into `ComfyUI/custom_nodes`; you can also clone it there yourself.

## Strict Moss Sampler

Use it instead of the KSampler of a Moss image workflow:

| Node | Setting |
|---|---|
| Load Diffusion Model | any Moss checkpoint, like `moss-image-sft-dit-int8-convrot.safetensors` |
| LoraLoaderModelOnly (optional) | style LoRAs, chain several for a stack |
| Load CLIP | `qwen3.5_9b_int8_convrot.safetensors`, type `moss_image` |
| CLIP Text Encode | the prompt, into `positive` |
| Empty Flux 2 Latent | width, height, batch size |
| Load VAE | `flux2-vae.safetensors`; `taef.safetensors` also works and decodes much faster with slightly lower quality |
| **Strict Moss Sampler** | `quality`, `seed` |
| Save Image | the `IMAGE` output |

`quality`:

| Preset | Steps | What it does |
|---|---|---|
| `best` (default) | 8 | every step computed, with the turbo LoRA; the same images as ComfyUI, sampled faster |
| `best25steps`, `best50steps` | 25, 50 | without the turbo LoRA, cfg 2, for workflows that keep their high step count |
| `fast`, `faster` | 8 | some steps are forecast instead of computed (5 or 4 of 8 evaluated) |
| `fastest` | 4 | 4 turbo steps, 3 evaluated |
| `fastest2steps` | 2 | 2 turbo steps, softer, about half the images usable |
| `custom` | your own | the node's `steps`, `cfg` and `spectrum` inputs |

Turbo LoRA: 8 steps or fewer need one. Save lodestones' [DMD LoRA](https://huggingface.co/lodestones/MOSS-Image-Base-8-steps-dmd/blob/main/lora.safetensors) as `dmd_turbo_lora.safetensors` in `models/loras/moss`. For 4 steps or fewer, akhaliq's [4-step LoRA](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps/resolve/main/moss_image_sft_4step_v2.0.safetensors) in the same folder gives better images. With `turbo` set to `auto` the node adds them itself and leaves them out when a LoraLoader or the checkpoint already has one; set `turbo` to `off` for checkpoints with another turbo LoRA baked in.

Other inputs: `lora_mode` (`merged`: the LoRAs are merged into a cached int8 model, the first run with new LoRAs takes a few seconds longer and later runs load faster; `runtime`: floating adapters) and `skip` (layers that every other evaluated step reuses, faster, changes the result).

The model stays loaded between runs until you change the model or its LoRAs.

## Strict H3 Sampler

It replaces the sampler nodes and both VAE decodes of an H3 image to video workflow:

| Node | Setting |
|---|---|
| Load CLIP | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, type `minimax` |
| Load VAE | `minimax_h3_video_vae_int8_convrot.safetensors` |
| MiniMax H3 Image to Video | prompt, width, height, length, optional first and last frame |
| **Strict H3 Sampler** | `positive` and `latent` of MiniMax H3 Image to Video, `quality`, `seed` |
| Create Video + Save Video | the images at 24 fps with the audio |

Models, in ComfyUI's `models` folder (ComfyUI's template Templates, Video, MiniMax H3 Text to Video downloads the first four):

| File | Used for |
|---|---|
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | the H3 model |
| `loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | turbo LoRA of `best`, `fast` and `faster` |
| `loras/minimax_h3_turbo_v4_step600_ema.safetensors` (optional, recommended) | used instead of the template's turbo LoRA when it is there, it gives the best quality |
| `vae/minimax_h3_video_vae_int8_convrot.safetensors` | video decoder |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | audio decoder |
| `loras/h3/minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors` | TaoMate LoRA of `fastest4steps` and `fastest2steps` |
| `vae_approx/taeh3.safetensors` (optional) | a preview of every step in the node, so a bad video can be cancelled early |

`quality`:

| Preset | Steps | What it does |
|---|---|---|
| `best` | 8 | with the turbo LoRA, the same quality as ComfyUI, sampled faster |
| `fast` (default) | 8 | `best` with routed attention, close to `best` |
| `faster` | 8 | `fast` with forecast steps (5 of 8 evaluated) |
| `fastest4steps` | 4 | the TaoMate LoRA, not recommended for fast motion or detailed prompts |
| `fastest2steps` | 2 | the TaoMate LoRA, lowest quality, for previews |

Turbo LoRA: the node and H3Video choose it the same way. `best`, `fast` and `faster` use `minimax_h3_turbo_v4_step600_ema.safetensors` when it is in `models/loras` (or `models/loras/h3`), else the 8 step turbo LoRA of ComfyUI's template. Any other turbo LoRA is your choice: load it with a LoraLoader (on the command line `--lora`). The preset's LoRA is left out when a LoraLoader on the `model` input already has one of the two or the checkpoint name says one is merged in (turbo, dmd, 8step, 4step, hyperflow, distill or lightx2v). The fastest presets use a LoRA named turbo, 8step or taomate instead of TaoMate when you give one.

Other inputs: `skip` (blocks that every other evaluated step reuses, default `5:45`, wider spans are faster and change the result more), `sampler_name` and `scheduler` (default `euler` and `simple`), and an optional `model` input (Load Diffusion Model, with LoraLoaderModelOnly if you like) that replaces the preset's checkpoint.

## Command line

The programs in `Release/Windows` (on Linux `Release/Linux`, without `.exe`) run without ComfyUI. They read the prompt from `prompt.txt` in the current folder and the models from your ComfyUI `models` folder (`--models` names another one). `--help` lists every option of each.

```
MossImage.exe --count 4
MossImage.exe --workflow my_moss_workflow.json
H3Video.exe --quality faster --image first.jpg
H3Image.exe --prompt "a red fox in fresh snow at sunrise, photo"
H3Image.exe --image taxi.png --prompt "Replace the taxi in <Picture 1> with a pink ice-cream truck"
StrictGen.exe
```

**MossImage** makes Moss images with the same presets as the node. `--workflow` takes the model, LoRAs, prompt, size, seed and steps from a workflow saved in ComfyUI.

**H3Video** makes H3 videos with sound: text to video, image to video with `--image` (and `--last_image`), and with `--video` it continues an mp4. Useful options:

- `--vae taeh3.safetensors` (madebyollin's TAEHV for H3 in `models/vae_approx`) decodes a draft video much faster and visibly softer, mainly for previews with the fastest presets.
- `--upscale` scales every finished video up 2x with NVIDIA RTX Video Super Resolution, like 360p to 720p. It needs an RTX card and `nvidia-vfx` in ComfyUI's Python (installed by [ComfyUI_Nvidia_VFX_Nodes](https://github.com/Comfy-Org/ComfyUI_Nvidia_VFX_Nodes), or `python -m pip install nvidia-vfx`). `RtxVideo.py` runs with that Python as a hidden helper that keeps its model loaded and ends with the H3Video backend or after 15 idle minutes.
- `--no_turbo` never adds the turbo or TaoMate LoRA, for models with one merged in.
- `--commands commands.txt` makes the videos of every line of that file, loading the models once.

**H3Image** makes images with MiniMax H3 Image and with `--image` edits that image: the prompt is the edit instruction and can call the image `<Picture 1>`. It needs `diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors` and `vae/minimax_h3_image_vae_fp16.safetensors` of ComfyUI's H3 Image templates and the same text encoder as H3Video. The default is 6 steps at 2048x2048, or the edited image's size with its short side between 1024 and 2048; `--width`, `--height`, `--steps`, `--count`, `--seed` and `--lora` change that. `--vae` picks the decoder: the H3 Image VAE (default), `p-h3-image_vae_int8_convrot.safetensors` (cleaner faces, slower) or `taeh3.safetensors` (draft). Images are saved as .jpg unless `--output` ends with .png.

**StrictGen** serves a local browser page (`http://localhost:8001`, `--port` changes it) for Moss Image, H3 Video, H3 Video Extend, H3 Image to Video and H3 Image, with live step previews, time and token budget estimates, a run queue and automatic discovery of other StrictGen PCs on the local network. `StrictGen.exe stop` ends it and the running backends.

### Backends and memory

MossImage, H3Video and H3Image start a backend in the background the first time. It keeps the text encoder, the model and the decoders loaded, so the next run skips loading them, and it keeps prompts it has already encoded. The backend holds its RAM and VRAM between runs and ends after 15 minutes without a run or when RAM runs low; on a machine with too little RAM to keep everything, runs load the models themselves. `--local` runs without the backend, `MossImage.exe stop`, `H3Video.exe stop` and `H3Image.exe stop` end it.

Before loading models, each program checks that the RAM and VRAM it needs are free (at most half the RAM and three quarters of the VRAM of the machine). If the machine has them but other programs hold them, the run stops right away and lists those programs, our own backends with the command that frees them. `--force` (or `STRICT_FORCE=1`) runs anyway, which is very slow or fails while the others hold the memory.

## Speed

The measured times and the comparison with ComfyUI are on [strict-lang.org](https://strict-lang.org).

## Building from source

With [Strict](https://github.com/strict-lang/Strict) on the PATH, in this folder:

```
strict build MossImage windows
strict build H3Video windows
strict build H3Image windows
strict build StrictGen windows
```

and the same with `linux` for the Linux programs.

Tests: `python test_generate.py` with ComfyUI's Python, `strict test MossImage.strict`, `strict check .`
