# StrictComfy

Two ComfyUI nodes that make Moss images and MiniMax H3 videos with sound faster, sampled by native [Strict](https://github.com/strict-lang/Strict) programs on your NVIDIA GPU:

- **Strict Moss Sampler**: generate images with Moss faster than anything else. best keeps all steps, identical quality and results. fast, faster, fastest skip more and more work like spectrum while keeping the output close.
- **Strict H3 Sampler**: videos with sound with MiniMax H3, **Strict H3 Sampler (Latent)** returns the latents instead.

Both are in the node menu under `sampling/strict`.

## Requirements

- Windows or Linux, 64 bit
- An NVIDIA GPU (RTX 20 series or newer) with a current driver
- ComfyUI

## Install

In ComfyUI-Manager search for **StrictComfy** and install it, then restart ComfyUI. A clone of this repository is now in your `ComfyUI/custom_nodes` folder, you can also clone it yourself there.

## Strict Moss Sampler

Use it instead of the KSampler of a Moss image workflow:

| Node | Setting |
|---|---|
| Load Diffusion Model | any Moss checkpoint, like `moss-image-sft-dit-int8-convrot.safetensors` |
| LoraLoaderModelOnly (optional) | style LoRAs, chain several for a stack |
| Load CLIP | `qwen3.5_9b_int8_convrot.safetensors`, type `moss_image` |
| CLIP Text Encode | the prompt, into `positive` |
| Empty Flux 2 Latent | width, height, batch size |
| Load VAE | `flux2-vae.safetensors`, `taef.safetensors` also works and is much faster with pretty good quality |
| **Strict Moss Sampler** | `quality`, `seed` |
| Save Image | the `IMAGE` output |

`quality`:

| Preset | |
|---|---|
| `best` (default) | 8 steps, identical output, you can use any model, any 8 step or 4 step turbo lora and things stay the same and sampling is just faster. |
| `best25steps`, `best50steps` | more steps without turbo LoRA (if you want to keep your high steps of your workflow and just utilize the faster sampling without any quality loss) |
| `fast`, `faster` | 8 steps, some forecast instead of computed (spectrum like, but more dynamic and way faster) |
| `fastest`, `fastest2steps` | 4 or 2 steps (extremely fast, usually best with a 4 step turbo lora, with 8 steps you will get undercooked results). |
| `custom` | your own `steps`, `cfg` and `spectrum` |

8 steps or fewer need a turbo LoRA: put [`dmd_turbo_lora.safetensors`](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/experimental/minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors) into `models/loras/moss`. For 4 steps or fewer, akhaliq's [4-step LoRA](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps/resolve/main/moss_image_sft_4step_v2.0.safetensors) in the same folder gives better images. With `turbo` `auto` the node adds them itself, set `turbo` to `off` for checkpoints that already have one baked in (if you model already has any turbo lora, this is detected and nothing will be added).

The first run with new LoRAs prepares a merged model (about 5-10 s), later runs reuse it for faster loading speed. The model stays loaded between runs until you change the model or LoRAs.

## Strict H3 Sampler

It replaces the sampler nodes and both VAE decodes of an H3 image to video workflow:

| Node | Setting |
|---|---|
| Load CLIP | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, type `minimax` |
| Load VAE | `minimax_h3_video_vae_int8_convrot.safetensors` |
| MiniMax H3 Image to Video | prompt, width, height, length, optional first and last frame |
| **Strict H3 Sampler** | `positive` and `latent` of MiniMax H3 Image to Video, `quality`, `seed` |
| Create Video + Save Video | the images at 24 fps with the audio |

Needed in `models`: the files of ComfyUI's MiniMax H3 template (Templates, Video, MiniMax H3 Text to Video downloads them): `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`, `loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` (the turbo LoRA every quality but `fastest2steps` adds, left out when a LoraLoader on the `model` input already has it), `vae/minimax_h3_video_vae_int8_convrot.safetensors` and `vae/minimax_h3_audio_vae_fp32.safetensors`. For `fastest2steps` also `loras/h3/minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors`.

`quality`: `best` (same quality, just faster generation), `fast` (default, close and can be 30% faster), `faster`, `fastest4steps` (way faster, but not recommended for fast motion or detailed prompts), `fastest2steps` (fastest, lowest quality, for previews, low resolutions generate really fast in a couple of seconds). Both fastest presets use the TaoMate 3 step LoRA instead of the 8 step turbo LoRA, an 8 step LoRA added to them is an error. An optional `model` input (Load Diffusion Model with LoraLoaderModelOnly) replaces the preset's checkpoint. With `models/vae_approx/taeh3.safetensors` installed the node shows a preview of every step, so you can cancel a bad video early.

## Command line

`Release/Windows/MossImage.exe` and `Release/Windows/H3Video.exe` (on Linux `Release/Linux/MossImage` and `Release/Linux/H3Video`) also run without ComfyUI. They read the prompt from `prompt.txt` in the current folder and the models from your ComfyUI `models` folder:

```
MossImage.exe --count 4
MossImage.exe --workflow my_moss_workflow.json
H3Video.exe --quality faster --image first.jpg
```

`--help` lists every option.

H3Video.exe starts a backend in the background the first time, which keeps the text encoder, H3 and the VAEs loaded for the next runs: a second video with a new prompt skips loading them (on a 4090 23 s instead of 59 s). The backend keeps its RAM and VRAM between runs and ends after 15 minutes without a run or when RAM runs low. On a machine with too little RAM to keep everything, runs load the models themselves. `--local` runs without the backend, `H3Video.exe stop` ends it.

MossImage.exe does the same with the text encoder and Moss (it needs 24 GB of RAM free to start): on a 3080 a second 480p image takes 1.8 s instead of 13.6 s, 3 s with a new prompt. Prompts already encoded are kept. `--local` and `MossImage.exe stop` work like for H3Video.

Before loading models, both check that the RAM and VRAM they need are free (H3 about 23 GB RAM and up to 18 GB VRAM, Moss about 15 GB RAM and 6 GB VRAM, at most half the RAM and three quarters of the VRAM of the machine). If the machine has them but other programs hold them, the run stops right away and lists the programs using them, our own backends with the command that frees them, for example `type 'H3Video stop' to free them`. A GPU smaller than the model is fine: the run says so and keeps the rest in RAM, slower.

## Building from source

With [Strict](https://github.com/strict-lang/Strict) on the PATH, in this folder:

```
strict build MossImage windows
strict build H3Video windows
strict build MossImage linux
strict build H3Video linux
```

Tests: `python test_generate.py` with ComfyUI's Python, `strict test MossImage.strict`, `strict check .`