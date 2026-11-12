# Handoff: decoders, benchmarks and website (2026-11-12)

State on the AiServer (4090) when the GPU was taken by a training job. Everything below is committed on main.
`strict check .` has 0 issues, all 104 files pass `strict test <file> native` with Strict 9b5268d2 (obj/ was cleared,
every method without a return type changed its bytecode in that compiler, so clear obj/ on a new machine too).

## Done

- **H3Image speed** (404f359): see Readme and memory h3-image-speedups. Warm 4090: same prompt 1024 1.6 s, 2048
  4.2-4.5 s; new prompt 1024 4.8-5.1 s, 2048 6.7-7.3 s, 4096 21.0 s; edit 2048 same image + instruction 9.0 s, new
  image + instruction 16.4 s. Website (strict-lang.org 42a56b7) has an H3 Image card with 1k/2k/4k against ComfyUI
  (23/30/35 s, the user's ComfyUI numbers, not measured here).
- **Moss decoder default**: `flux2-vae.safetensors` for every preset, taef2 for `fastest` and `fastest2steps` (taef2
  0.03 s, Flux 2 VAE 0.15 s per 704x1280 image, taef2 35.6-37 dB from the Flux 2 VAE, visibly softer). MossOptions.Decoder,
  MossFiles.Fast/Tiny. StrictGen's decoder select has "auto" (default, sends no --vae). MossWeb unchanged: on the 4090 all
  three decoders take 3.0-3.2 s per 480p image, Flux 2 needs about 2 GB more VRAM and a 336 MB download, and its
  default preset is fastest anyway.
- **H3Image --vae choices** (2540967): the H3 Image VAE (default), `p-h3-image_vae_int8_convrot.safetensors` (Pruna's
  single frame decoder: 36 int8 blocks, 3D rope at time -0.857; cleanest faces on generated images, matches ComfyUI at
  36.6 dB, decode 0.68 s against 0.31 s at 1024 measured while a training job ran) and `taeh3.safetensors`
  (madebyollin's TAEHV for H3, Taeh3Decoder: the 4 frames of one latent, keeping the last; draft quality, ~0.1 s).
  H3ImageVae holds the decoder and the edit encoder (edits encode with the H3 Image VAE when --vae is taeh3).
- On the AiServer `models/vae_approx/taeh3.safetensors` is now madebyollin's (from
  github.com/madebyollin/taehv/raw/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/safetensors/taeh3.safetensors), Kijai's 2D
  preview TAE was renamed to `taeh3_kijai.safetensors` (far worse: smeared faces, 18.7 dB vs 23.9 dB). Copy madebyollin's
  to the home PC too. Downloaded to `models/vae`: p-h3-image_vae_int8_convrot (PrunaAI/P-H3-Image-Comfy),
  minimax_h3_single_frame_500k_comfy (Alissonerdx/...-500K-Comfy), minimax_h3_t1_image_vae_step1597 (Mamad8).
- **H3 decoder comparison** (ComfyUI, Kodak photos round trip through the H3 Image VAE encoder): image VAE 27.9 dB,
  video VAE 28.1, P-H3 28.9, t1 29.5, 500K with 512 px tiles 30.0, Flux 2 35.1. Faces of ~50 px break with every H3
  decoder (the 32:1 latent loses them), t1/500K only blur them; on generated images t1/500K look painted, P-H3 is best.
  t1 and 500K were not added (5.2 GB fp16, 36 blocks, softer). The 500K Comfy file has no `h3_t1_direct` metadata:
  stock ComfyUI decodes it as a video VAE, it needs still frame 3 and 512 px tiles.
- Servers: IsGranted/IsContinued return Booleans (MachineUse.Checked is one since Strict aae8af36), Served opens and
  delegates to Listened, `Backend.Clean` is the exit code of a backend that ended.

## Can be done on the 3080 (10 GB)

1. **taeh3 for H3Video** (user asked for it as a fast preview decoder, mainly for fastest4steps/fastest2steps): the
   full multi-frame TAEHV decode. Reference: `comfy/taesd/taehv.py` (apply_model_with_memblocks, non-parallel: frames
   run depth first, a MemBlock's past is the same block's input of the previous frame, TGrow splits a frame into
   stride frames) and its H3 chunking in `TAEHV.decode` (pad to chunks of 5 latents = 20 frames, drop the first 3 frames
   of every chunk, then the last 12). Taeh3Decoder.Decoded is the one-latent special case (Ran, Remembered, Grown,
   Shuffled, the tiny fixture tinytaeh3.safetensors comes from ComfyUI's own TAEHV blocks). H3Video's decode path:
   H3Vae.Decoded / H3Decoding; an option like H3Image's `--vae taeh3.safetensors`.
2. **Moss on the 3080 for the website**: 10 image batches per preset with the new default decoder and with
   `--vae taef2.safetensors` (website wants both where the difference is over 10 %), plus MossWeb 480p with each decoder
   (Release/Web, C:/StrictDumps/vae/webrun.mjs now takes a "decoder" per run).
3. P-H3 and taeh3 for H3Image on a 10 GB card: does P-H3 (3.1 GB int8) still fit next to the streamed H3 Image at
   1024? Time decode only.

## Needs the 4090 (24 GB), do on the AiServer

1. **H3Image benchmark at 1, 4, 8 and 16 MP** (1024², 2048², 3840x2160 or 2880², 4096²) warm, new prompt per run,
   against ComfyUI's H3 Image template on the same files (headless ComfyUI with PYTHONUTF8=1; call
   `comfy.model_management.unload_all_models()` after conditioning and after sampling or it runs out of VRAM). The user
   expects the gap to grow at 8/16 MP. Also time each decoder (image VAE, P-H3, taeh3) per size.
2. Moss 4090 chart with the Flux 2 VAE default (the website's 4090 numbers were measured with taef2: Nov 12 re-run
   with taef2: fastest2steps 0.46, fastest 0.54, faster 0.75, fast 0.75, best 1.04 s per image).
3. Update `strict-lang.org/website/pages/en/index.js`: H3Image 8/16 MP and ComfyUI rows, decoder choices, Moss with
   the new default (and taef2 rows where it is over 10 % faster). H3Video numbers there are from Nov 12 (best 74.6 s
   since best skips no blocks, 56.9 s with --skip 5:45).

Scripts on the AiServer (gitignored): `artifacts/bench/bench.sh` (H3Image cases), `stamp.sh` (timestamped log lines),
`site.sh` (Moss + H3Video website runs), `psnr.py`; the ComfyUI decoder scripts are in the session scratchpad
(roundtrip.py, gendecode.py, tinytaehv.py) and easy to rewrite from comfy.sd.VAE.
