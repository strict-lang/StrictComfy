# Packages

Every folder with `.strict` files is a package. Types are found across all folders of the project, tests and native tests run from this folder (fixture paths like `tinymodels/tinyh3.safetensors` stay as they are).

Each package may only use the packages listed before it, never the ones after it, and never the programs in the main folder:

| Folder | Purpose | May use |
|---|---|---|
| `Shared` | ComfyUI workflow graphs, samplers and solvers, model file lookup, resident backends, prompt batches, the Flux 2 and TAE decoders, local HTTP helpers, tensor helpers | Strict base packages |
| `Moss` | Moss image model: transformer, text encoder blocks, LoRAs, sampling, spectrum, options, resident backend | `Shared` |
| `H3` | H3 model shared by video and image: transformer blocks, layout, rope, spectrum and spans, video and audio VAEs, frames, extend clips | `Shared`, `Moss` |
| `Video` | H3 video pipeline: options, files, text encoder, batches, commands, estimates, resident backend, client and server | `Shared`, `Moss`, `H3` |
| `Image` | H3 Image: image model and layout, image VAE, options, job, resident backend and server | `Shared`, `Moss`, `H3`, `Video` |
| main folder | programs with a Release build (`MossImage`, `H3Video`, `H3Image`, `StrictGen`, `MossWeb`), the StrictGen page and its parts, diagnostics | all |

The programs stay in the main folder: `strict build` puts `Release/` beside the program file, the ComfyUI nodes (`generate.py`) start `Release/Windows/*.exe` and StrictGen starts the backends beside itself. No folder is named like a program, `strict build H3Video` would build that folder instead.

## Files

- `Shared` (25): Backend, BetaDistribution, CommandWords, ComfyGraph, ComfyInputs, ComfySize, ErSde, Euler, FluxAttention, FluxDecoder, FluxImage, FluxResnet, GenerationCommand, Lanczos, MathExpression, ModelFiles, PromptBatch, Solver, StepTimes, StrictGenHttp, StrictGenRemote, Taef2Block, Taef2Decoder, Taeh3Decoder, Tensors (new)
- `Moss` (22): LoraFiles (was H3Lora), Moss, MossBatch, MossBlock, MossDecoding, MossEstimate, MossFiles, MossGguf, MossLora, MossOptions, MossQuality, MossResident, MossRun, MossSampling, MossServed, MossServer, MossSpan, MossSpectrum, MossText, MossTiming, MossTuning, MossWorkflow
- `H3` (28): H3, H3Ahead, H3Audio, H3AudioBlock, H3AudioEncoder, H3Block, H3Clip, H3ClipEncoder, H3Conv, H3Decoder, H3DecoderBlock, H3Final, H3Format, H3Frames, H3Job, H3Layout, H3Picture, H3Previewer, H3Snake, H3Span, H3Spectrum, H3Tiles, H3Vae, H3VaeEncoder, Rope, StrictGenClip, StrictGenExtend, StrictGenJoin
- `Video` (22): H3Artifacts, H3Batch, H3Client, H3Commands, H3Decoding, H3Encoder, H3Encoding, H3Estimate, H3Files, H3Inputs, H3Options, H3Plan, H3Quality, H3Qwen, H3Refiner, H3Resident, H3Sampling (new), H3Served, H3Server, H3StepLog, H3Workflow, RtxVideo
- `Image` (8): H3ImageJob (new), H3ImageLayout, H3ImageModel, H3ImageOptions, H3ImageResident, H3ImageServed, H3ImageServer, H3ImageVae
- main folder (16): H3Image, H3Video, MossImage, MossSpectrumImage, MossWeb, StrictGen, StrictGenConsole, StrictGenFiles, StrictGenForm, StrictGenHost, StrictGenImage, StrictGenJob, StrictGenNet, StrictGenQueue, StrictGenStats, StrictGenTitle

## Code moves that keep the dependencies one way

- `Tensors.RelativeError` (from MossImage), `Tensors.Transpose` (from Moss), `Tensors.Combine` (from H3Span), used by every package.
- `ModelFiles.Weights` from `H3.Weights` (Moss loads with it).
- `Backend.EncoderGrowth` from `H3Resident` (Moss, Video and Image servers).
- `MossResident.Spelled` gets its own tiny tokenizer instead of `H3Encoder.Spelled`.
- `H3Lora` becomes `Moss/LoraFiles`, the LoRA file reader of both models; its H3 fixtures (Block, Input, Layered, Tuned, Outputs, Forward, Refined) move to the H3 types whose tests use them.
- `MossSpectrum.Configured` from `MossSpectrumImage.Configure` (MossSampling uses it), the diagnostic program MossSpectrumImage moves to the main folder.
- Estimates: StrictGen sends `mode=image` requests to `MossEstimate.Told` itself, `Mebibyte`/`Gibibyte` go to Backend, `Lasting` to StepTimes, `Optional` to StrictGenHttp.
- `StrictGenHttp.Inputs`/`Outputs` from StrictGen (the extend clips and remote files use them).
- `H3Sampling` (Video) takes Load, Loading, Decoding, Generate, Layout, Compare, Layer, Sample and the tiny fixtures out of the H3Video program, the model names Base, Turbo, TurboV4, TaoMate go to H3Files.
- `H3ImageJob` (Image) takes everything but Run, Help and Unknown out of the H3Image program.
- `GenerationCommand` and `ComfyGraph.Tiny` tests read their fixtures without MossImage and H3Workflow.
