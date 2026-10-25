# Moss performance

## Runtime-LoRA turbo 8 on tensor cores (2026-10-25 01:40, RTX 3080)

Same inputs as the "Runtime turbo 8" row below. Resident warm passes: sampling 11.464 -> 4.79-4.86 s, decode 1.079 s -> 430-450 ms, generation 12.563 -> 5.28-5.33 s, load 5.34 -> 4.2 s.

Changes: bf16 GPU-resident LoRA through Products tensor cores, fused QKV down projection, bf16 epilogues, plus the elementwise, VAE bf16 and layer-41 text-skip work. PSNR 33.28 dB vs the old control; visually checked.

Details, profile and next steps: Handoff.md. Normal clocks; this is not a matched ComfyUI comparison.

## RTX 3080: complete five-adapter stack, merge versus runtime (2026-10-25)

User's approximately 15 s/image report used DMD plus four style adapters. Old premerged/no-runtime-adapter Spectrum results were a different workload. Latest concurrent native MossLora rewrite already improves the exact floating stack. Same pinned executable/PTX, identical conditioning/noise, 720x1280, eight steps, CFG1, JPEG85, three resident generations per variant:

| Path | First generation | Next two complete generations |
|---|---:|---:|
| Base + complete floating runtime stack | 9.638 s | 7.232 / 7.183 s |
| Complete stochastic INT8 merge | 4.418 s | 3.255 / 3.252 s |
| Same merge + spectrum8_w3p3 | 3.424 s | 2.297 / 2.327 s |

Complete generation includes sampling, decode, save and inter-image cleanup, excludes process/model/text-encoder startup. Normal user Afterburner settings; no clock/power changes. Sequential resident runs, not locked-clock alternating repetitions or a new matched ComfyUI measurement. Exact ordered five-adapter identities/strengths are recorded locally in D:/StrictDumps/moss-merge-study/stack.json. Accumulate all deltas before requantization; CPU stochastic rounding seed 0, split fused qkv. One-time merge took 11.58 s isolated / 15.35 s via launcher. Source checkpoint unchanged; each cached stack occupies a full model.

Real MossImage CLI count3 with that stack reused the cache and enabled three forecasts: sampling 1.723/1.744 s, complete warm generations 2.289/2.307 s. Direct cached-model count2 avoided duplicate DMD and gave 2.291 s warm. Cold CUDA-cache/backend loading took 31.60-41.66 s, but a repeat invocation loaded in 3.059 s and took 10.63 s total for two images. First CLI off run included uncached text encoding and cold loading: 69.44 s total, generations 6.187/4.231/3.462 s. Thus 2.3 s is resident performance, not process wall time. Concurrent agent rebuilt native executable between controlled and CLI tests; pinned comparisons use the same earlier snapshot.

| Seed | Merge vs runtime PSNR | Merge+Spectrum vs runtime PSNR | Spectrum vs merged PSNR |
|---|---:|---:|---:|
| 42 | 21.29 dB | 20.51 dB | 25.57 dB |
| 1234 | 24.50 dB | 22.64 dB | 25.47 dB |
| 7 | 24.99 dB | 23.37 dB | 27.37 dB |

Prompt: portrait of a woman wearing a green wool sweater, holding a ceramic mug in a cafe. All three merged and Spectrum outputs were inspected: coherent images with visible pose/skin/finger/background changes. This is an approximate quality tradeoff, not proof of fidelity for all adapters/prompts. User review gallery: D:/StrictDumps/moss-merge-study/index.html; raw paired and CLI logs plus quality.json alongside it.

CLI defaults to cached --lora_mode merged, offers floating --lora_mode runtime, and explicitly enables prediction with --spectrum spectrum8_w3p3 (off by default). Previous CLI hardcoded off, explaining forecasts 0. --merge_only prepares/prints a reusable marked checkpoint without encoder/VAE/prompt work; baked DMD metadata prevents automatic reapplication. Cache identity includes model, ordered stack, strengths and merger version; old uncertified caches are unused. Nineteen CPU-only CLI/merge checks pass, StrictComfy folder check 0 issues. No compiler/native production changes in this commit. See Generation.md for commands.

## Quality correction: original floating turbo adapter (2026-10-25)

The old premerged DMD default is not a validated turbo baseline: sampled q/FFN-down INT8 code arrays are unchanged from the base checkpoint, with only tiny row-scale differences. Earlier 45.07 s/ten-image timing measured that artifact, not the corrected runtime-LoRA workload. CLI now loads unchanged base INT8 weights and automatically applies original DMD at eight steps. It packs weighted adapter stacks without quantizing their deltas into base weights. New comparison manifests use the same runtime approach; existing manifests are not rewritten.

Same prompt "Portrait photograph of a woman, natural daylight.", seed 42, 720x1280, no Spectrum. Base50/base25 use CFG3, no LoRA; turbo8 uses CFG1 and original DMD. Direct native backend times (unlocked clocks, not a calibrated ComfyUI comparison):

| Case | Sampling | Decode/save/full generation | Process including load |
|---|---:|---:|---:|
| Base 50 | 37.315 s | 37.861 s | 71.91 s (33.672 s first-use load/JIT) |
| Base 25 | 19.316 s | 21.087 s | 24.52 s |
| Runtime turbo 8 | 11.464 s | 12.563 s | 18.39 s |

MossLora.Scaled now preserves original packed rows when scale is 1; prior runtime turbo exceeded 180 s and reached only step 2. The normal CLI check took 26.04 s and reproduced the native turbo JPEG exactly; it created no JSON by default. Sixteen command tests pass, including floating stack preservation of updates smaller than base quantization steps. Images/logs are under D:/StrictDumps/moss-quality-review/{base50,base25,turbo8-runtime}; the user should judge these three samples. No claim that all prompts are fixed or that runtime turbo now beats ComfyUI. An additional run became contended with a separate GPU benchmark and was stopped; its timing is invalid. Next performance target is the floating LoRA product path and resident adapter/weight memory behavior.

## Resident randomized CLI batches (2026-10-24, RTX 3080)

--count now reuses one native model and VAE across different prompts/noise, rather than restarting inference for every image. Same 720x1280 merged DMD, eight steps, two PEFT adapters at 0.5/0.31, normal Afterburner settings. --seed 42 fixes a ten-image batch with ten distinct seeds and three randomized prompt variants. The first batch took 78.34 s including one encoder load for two uncached unique prompts, one stack merge and one native load. A fully cached repeat took **45.07 s for ten images**, with one 3.052 s native load. Completed-image native durations: 5298 / 4507 / 4189 / 3563 / 3537 / 3477 / 3473 / 3482 / 4469 / 4001 ms; median after image one **3563 ms**, including VAE/JPEG and between-image cleanup. No clocks/power settings changed; this is not a locked-clock ComfyUI comparison.

All ten output hashes differ within a batch and match their corresponding seeded repeat exactly. Image one also reproduces the previous standalone seed-42 JPEG. Per-image metadata and dimensions were checked; output numbering advanced 0003..0022 without changing earlier files. Prompts are encoded once per unique cache miss before native startup, freeing encoder VRAM; the native loop reads distinct per-job safetensors. Logs and images: D:/StrictDumps/moss-cli/count10{,-cached}.log. Thirteen command tests and native builds/checks pass. A separate default-only single-image run with --jpg_quality 85 validated the newly optional model/VAE/encoder defaults (10.24 s total).

## User command timing (2026-10-24, RTX 3080)

MossImage.exe now accepts prompt files, repeated weighted LoRAs and numbered output paths; usage is at the top of Generation.md. Full command validation used 720x1280, the existing merged eight-step DMD checkpoint, and two PEFT style adapters at 0.5/0.31 (246 matched modules each). A four-step invocation checks CLI plumbing only; shortening an eight-step checkpoint is not a four-step quality recommendation.

A cached default eight-step invocation from another working directory took 10.81 s end to end: fresh native load 3.076 s, sampling 4.132 s, decode 1.505 s, JPEG save 19 ms, full generation including cleanup 5.658 s. Prompt conditioning, merged stack and driver compilation were cached. This is a new native process per invocation, not the resident 3.47 s measurement below. First use is slower: one successful new-stack/first-driver-use run took 51.47 s with prompt conditioning already cached. Cold decode also varies (1.5-4.1 s in these plumbing checks). No comparison against ComfyUI's warm pipeline is claimed from these timings.

Actual outputs MossImage0001.jpg / MossImage0002.jpg, per-image settings, logs and caches: D:/StrictDumps/moss-cli. Default prompt/steps/output, spaces in output paths, and cache hits all passed full-size runs. Nine lightweight command tests plus VM/native launcher tests and folder checks pass. A resident worker and coordinated ComfyUI text-encoder/Strict VRAM handoff remain needed before a sampler node avoids cold model loading.

## JPEG output and merged decoder check (2026-10-24, RTX 3080)

JPEG quality 85 is now MossGenerate/compare_moss.py's Windows default. Native ImageCodec.SaveJpeg uses the built-in Windows codec on flat RGB values; --format png retains lossless stored PNG, --format png --compressed retains compressed PNG. Same portrait pixels: 95,022-byte JPEG versus 3,688,028-byte stored PNG (97.4% smaller), JPEG PSNR 42.22 dB and mean absolute byte difference 1.45/255. Actual Strict save costs 23-30 ms versus 451-473 ms for stored PNG in the fresh normal-clock pair, including foreign argument conversion. This is lossy output compression, not a model-precision change.

Full public-harness pair, same 720x1280 turbo8 inputs/checkpoint, Spectrum off, model/decoder reused three times per process:

| Format | Sampling, passes 0/1/2 | VAE decode, passes 0/1/2 | Save, passes 0/1/2 | Full generation, passes 0/1/2 |
|---|---|---|---|---|
| JPEG 85 | 4445 / 2875 / 2796 ms | 1845 / 995 / 649 ms | 25 / 27 / 23 ms | 6347 / 3938 / 3473 ms |
| Stored PNG | 4473 / 2935 / 2808 ms | 1934 / 901 / 574 ms | 473 / 464 / 451 ms | 6946 / 4380 / 3836 ms |

Normal Afterburner settings, no nvidia-smi locks or clock/power changes this session. Do not compare these directly with the earlier locked-clock ComfyUI table. Decode still varies with allocator warmup; the save stage is the clear measured improvement. Earlier normal-clock JPEG pass reached 3424 ms total. Diagnostics: D:/StrictDumps/vae-regression/{jpg,png}-final{,-stages}.log.

The merged H3 copy-stream/trim guard is present (Strict 2354fc5c). Keep it: trimming while rows stream repeatedly drains/frees workspace that subsequent layers need. Native cache checks confirm opportunistic trim is suppressed with spilled rows, while explicit GpuMemory.Released still frees unused large blocks between resident Moss generations. Fixed a test merge break: wrong DLL/export, two-element view of three counters, and leftover blocks from the streaming guard test. No allocator policy change was made in this session.

The checked-in 4090 report calls the c1829a08 regression **Moss VAE**, not H3 VAE. It is not reproduced as a 300 ms regression on this 3080. Diagnostic ablations only, current host allocator retained:

- Same fixed latent, standalone decoder, current versus pre-change linear batching: warm 433 / 431 ms, PSNR 61.01 dB in both. No meaningful regression.
- Including Attention in an isolated decoder selects sm_80 instead of sm_75 (a c1829a08 build change): warm decode 405 ms. No slowdown from that target change here.
- Full MossImage after layer comparisons/sampling, current versus all pre-c1829a08 GPU/compiler paths: decode current 480 / 475 ms, ablation 470 / 473 ms, order current/ablation/ablation/current. Same model/noise/conditioning, but old attention math changes the sampled latent, so this is a performance ablation, not an exact-output comparison. No production compiler rollback is justified by this result. First-use driver compilation excluded from decode timings.
- Current resident nsys trace: VAE kernel work about 383 ms each pass, roughly 308 ms in convolutions and 31 ms in Products; host/transfer/allocation gaps vary with warmup. This remains slower than ComfyUI's earlier locked-clock decoder, but is separate from establishing the claimed regression.

Exact H3/4090 failing command and old/new run logs remain needed to reproduce the reported 0.3 s increase. Do not mark that issue fixed. Diagnostic compiler worktree StrictVaeCheck is unmerged; model executables restored to current production artifacts. Evidence: D:/StrictDumps/vae-regression, including profile.sqlite, fixed-latent controls and moss-*-pair*.log. JPEG codec check, ImageProcessing VM tests, native JPEG reader/ImageSaver tests, allocator/cache checks and both changed-folder checks pass.

## H3 first upload, resident rows and ComfyUI's attention kernels (2026-10-25 evening, 4090, quiet GPU)

Alternating plain runs of `H3Video.exe C:/StrictDumps/h3fox` (`C:/StrictDumps/ab/pairs.sh`, step 0 uploads all 21.8 GB of weights):

| Strict runtime | Step 0 | Steps 1-7 mean |
|---|---:|---:|
| master 0d18343f | 14.4-14.9 s | 3.70-3.77 s |
| rows tables up to 256 MB through the pinned staging buffer (taken, ../Strict) | 10.1-10.6 s | 3.68-3.70 s |
| + rows with one scale uploaded with that scale alone, uploads taking over spilled rows until 1.25 GB are free | 8.7-9.4 s | 3.82-3.84 s |
| first try: compact rows with an own reserve halved whenever a trimmed class came back | 13.8-14.5 s | 3.88-4.23 s |
| first try + bf16 `Modulated`, `Activated` and `Rope.Rotated` results | 12.6 s | 4.44-4.71 s |

The first step took 68 ms per 165 MB table: ~45 ms gathering into a fresh `malloc` (a page fault per 4 KB) and ~21 ms of synchronous copy through the driver's own staging; with the pinned buffer most of what is left is the gather on one thread. nsys of steps 3-4 (`C:/StrictDumps/prof/{master,resident}.sqlite`): master keeps 37 of 50 blocks resident and streams the rest by taking over spilled rows, GPU busy 3431 of 3663 ms; the compact rows keep 44 blocks resident (16.0 GB) next to 5.7 GB of pool blocks, but a 1 GB activation block and the re-uploaded weights then pushed each other out at every block (`cuMemFree` of 1 GB, ~20 ms and a synchronization each, GPU busy 75 %). The power of two pool classes hold 7 blocks of 512 MB for 319 MB lists (14,844 tokens x 5376 x 4 bytes) and 1 GB for the ~600 MB feed forward chunk, ~1.9 GB unused, about what is missing to fit everything. ComfyUI does not fit either: 47 GB of uploads in one 8 step run (`comfyh3.sqlite`), it overlaps them with the kernels.

ComfyUI's own faster attention kernels, `dump.py h3 --runs 2` (warm second run, final latents against the default run's):

| ComfyUI attention | Per call | Sampling | Video, audio error |
|---|---:|---:|---|
| default (cuDNN flash) | 3.22-3.24 s | 26.3 s | |
| SageAttention 1.0.6 (`--use-sage-attention`) | 2.35-2.36 s | 19.5 s | 0.279, 0.141 |
| comfy_kitchen INT8 (`--attention int8`: Hadamard rotated int8 Q and K, int8 V, uint8 shares) | 2.06-2.08 s | 17.1 s | 0.256, 0.164 |
| comfy_kitchen Sol-Attn (`--attention sol`, tau 1: 64 token blocks attend routed key blocks exactly, the others pooled) | 1.56-1.60 s | 13.1 s | 0.594, 0.346 |

`H3Video.exe --count 2` end to end (cached prompt, warm CUDA cache, `C:/StrictDumps/h3cli/e2e.log`): 108.3 instead of 141.2 s. The ComfyUI decode worker now starts with the backend and imports ComfyUI and loads both VAEs (~23 s) while the backend loads and samples, waiting for it before sampling cost those 23 s up front instead; launcher 4.1 s, load 11.9 s, first video 36.2 s (pinned staging), second 29.5 s, both decodes 23 s.

Strict against ComfyUI's default is 0.305, 0.092, so INT8 and Sage stay as close as Strict does; Sol-Attn changes the composition but not the quality (frames in `C:/StrictDumps/h3attn-*`). Attention is now the biggest gap: ComfyUI's INT8 kernel saves 1.16 s of its 1.96 s per step, Strict's f16 attention takes 1.70 s.

`H3Block.Heads` sets `Attention.quantized` (../Strict ec0c55e4): the keys become 8 bit numbers with a scale per position and head once per launch, the kernel quantizes each query row in registers and computes the scores with m16n8k32 8 bit products. Alternating runs of one binary (steps 2-7, h3fox): f16 3.59 / 3.84 s, 8 bit keys 3.24 / 3.41 s; one attention launch under ncu 40.8 -> 31.1 ms (tensor pipe 53 % busy, 35 of it the f16 products of shares and values); final errors against ComfyUI video 0.236, audio 0.084 (f16 0.305, 0.092). Quantizing the keys inside the kernel was slower (48 ms per launch): each of the 156 query blocks of a head quantized all its keys again, with shuffles and divisions on the path of every tile.

`--routing 1` (`Attention.routing`, ../Strict e77b332f, the H3 blocks hand `H3Layout.routing` to their attention): sparse attention like Sol-Attn, training free. Per 64 positions the keys, values and queries are pooled first, each block of 128 queries attends exactly the key blocks whose centered pooled score clears routing standard deviations (and its neighbors), every other key block counts once with its pooled score plus the logarithm of its length and its mean value. Alternating runs of one binary (steps 2-7, h3fox): dense 8 bit keys 3.51 / 3.41 s, routing 1 2.67 / 2.53 s. Final errors against ComfyUI video 0.590, audio 0.286, like ComfyUI's own Sol-Attn against its default (0.594, 0.346), the frames clean (`C:/StrictDumps/h3/routed1-frame-*.png`). The two halves of a 128 query block attend about 30 % of the key blocks exactly (a probe of comfy_kitchen's routing in ComfyUI: 23.5 % per block of 64, 30 % for the union of two). Off by default like `--spectrum`, it changes the result.

## H3 step against ComfyUI by category, and what did not help (2026-10-25, 4090, nsys windows of plain runs)

Per 8-step turbo INT8 step (14,844 tokens), Strict `C:/StrictDumps/prof/now/fox.sqlite` (steps 1-4), ComfyUI `comfy.sqlite` (dump.py h3, 8 warm steps of run 0, 3.22-3.24 s each under nsys):

| | Strict | ComfyUI |
|---|---:|---:|
| Attention | 1704 ms (`strict_device_attentions`) | 1962 ms (cuDNN flash fprop) |
| int8 GEMM | 961 ms (`strict_device_wholes_fast`) | 896 ms (CUTLASS `Kernel2`) |
| Quantize + Hadamard | 113 ms | 110 ms |
| Elementwise | 467 ms: Rope.Rotated 105, Gated 84, Qwen3Block.Scales 73, Modulated 73, Activated 65, Safetensors.Part 31, Batch.Columns 22 | ~206 ms: elementwise 126, rope 46, layer norm 34 |
| Kernels | 3250 ms | ~3170 ms |
| Idle | 460-650 ms | ~60 ms |

Both do the same work; Strict's attention is 258 ms faster, its elementwise kernels 260 ms slower and it idles 0.5 s per step. The idle of one step (50 gaps over 0.2 ms, 468 ms): host waits in small downloads (99 ms), `cuMemFree` (44 ms) and `cuMemAlloc` (15 ms, one of 34.5 ms at the step start), synchronizations around the streamed blocks (1-34 ms), uploads the next GEMM waits for (86 ms), the FedForward chunk loop (about 110 ms between its 4 chunks per block) and the step boundary (about 60 ms). 4.57 GB of weights go up per step (the device rows keep a float scale for every 32 int8 numbers: 436 instead of 387 MB per block, 21.8 GB in all).

Tried on alternating h3fox runs (steps 1-7, base 3.58-3.92 s):
- Strided attention values like Moss (combined qkv as values, no `Batch.Columns` copy): 3.64-4.32 s, slower, the combined list stays alive through the attention (+640 MB) and more weights stream.
- Inlining `normed` and `expanded` so they die earlier: 3.69-3.87 s, no change (the runtime frees them at their last use already); FedForward chunks of 2048 instead of 4096 tokens: 3.70-4.03 s, no change.
- `Qwen3Block.HeadScale` inside Modulated, Normalized and Rope.Rotated (Moss's fused form): 6.1-7.8 s, its reduction over H3's 5376-wide rows is not shared across the row.
- The pool's own-allocation reserve at 256 MB instead of 1 GB (../Strict NativeRuntime.mlir): 50 blocks unchanged (rows are spilled, the trim is skipped then); a test with 40 of the 50 blocks (everything resident) took 9.5-12 s per step with 1 GB (440 driver allocations and frees, 200 synchronizations per step) and 2.8 s with 256 MB, 0.27 s of that idle. Recorded in ../Strict/improvements.md 5 (c, d): the trim policy and a compact layout for rows with one scale come first, fitting alone would save about 0.1 s per step.

End to end after the latent and decoder fixes (`H3Video.exe --count 2`, cached prompt, timestamped, 141.2 s instead of 163.3 s): launcher 3.9 s, model load 12.6 s, the first video 41.6 s (with the 12 s first upload of the weights), the second 30.1 s (29.8 s sampling, the binary latent costs milliseconds instead of 5.4 s of text), then 12 s until the ComfyUI decode worker prints, 18.5 s of its start and VAE loading, 20.7 s decoding both (the audio VAE is loaded once instead of per video, libx264 takes 0.93 instead of libx265's 4.96 s per mp4 at the same size). A warm video costs 29.8 + about 10.3 s = 40 s against ComfyUI's warm 26.7 + 7.4 + 2.8 s + the mp4 = about 38 s; with spectrum8_w1p2 about 29.6 s. The decode worker's 30 s start is the biggest fixed cost left for single videos.

## H3 against today's ComfyUI, the H3Video command and bf16 results (2026-10-25, 4090, quiet GPU)

ComfyUI re-measured with `dump.py h3 --runs 2` (the second, warm run): turbo INT8 8 steps 26.7 s (3.26-3.29 s per call), base FL2VA INT8 + TaoMate 3-step LoRA (208 patches) 10.7 s (3.35-3.40 s per call), video VAE 7.4 s, audio VAE 2.8 s. The older 38.5 s (4.6-5.0 s per call) predates the GPU repaste and is withdrawn; ComfyUI's outputs are unchanged (Strict's per step errors against both dumps are identical). ComfyUI keeps the whole model resident ("19995MB Staged"), Strict streams part of the weights every step.

| Strict run | Steps (ms) | Sampling | Final error against ComfyUI |
|---|---|---:|---|
| Turbo INT8 8 steps, reference folder (step 0 partly warm) | 4763 3529 3639 3580 3643 3494 3608 3493 | 29.7 s | video 0.305, audio 0.092 |
| Same, second video of `H3Video.exe --count 2` (warm) | 3621 3541 3689 3533 3559 3525 3553 3539 | 28.6 s | |
| Turbo INT8, `spectrum8_w1p2` | 4832 1 3511 126 3605 33 3587 3579 | 19.3 s | video 0.415, audio 0.123 |
| Base + TaoMate, cached merge.py model, 3 steps | 5528 3682 3593 | 12.8 s | video 0.352, audio 0.106 |
| Base + TaoMate merged at load (runtime), 3 steps | 4153 4988 3888 | 13.0 s | video 0.385, audio 0.115 |

So warm Strict steps are about 9 % slower than ComfyUI's on the 4090 (3.57 against 3.27 s), Spectrum makes the 8 steps 28 % faster than ComfyUI. With TaoMate the refined text is as close to ComfyUI as without LoRA (0.0076 against 0.0069), the larger block errors come from both sides requantizing the merged int8 rows with their own stochastic rounding.

`H3Video.exe` end to end (cached prompt, turbo INT8, 8 steps): one video 120.9 s, two videos 163.3 s (load 10.5 s, the first generation 49.8 s with the 12 s first upload of the weights, the second 34.0 s, then one ComfyUI worker decodes both). Each generation spends about 5.4 s writing its 1.4M number latent as text (and the decode worker about as long reading it back): a binary latent needs a float writer in Strict. A fresh CUDA cache (the launcher's `artifacts/h3/cuda`) compiles the kernels during the first step (67 s once).

Rejected, Moss's bf16 elementwise results for H3 (`H3Block.Modulated` and `Activated` returning `BFloat16s` like MossBlock): alternating runs base 3.49-3.78 s per step, bf16 3.65-3.85 s in the first round and 4.4-4.9 s in the second, audio error 0.140 instead of 0.092 (video 0.261 instead of 0.305). The large bf16 results are GPU-only lists, which can't be evicted while H3 streams its weights (../Strict/improvements.md 13), the same effect that slowed the feature anchors of Spectrum.

## H3 LoRAs, step counts and Spectrum (2026-10-25, 4090 unlocked, quiet machine)

Plain runs of `H3Video.exe`, t2va 480x832, 124 frames. "Reference" is `C:\StrictDumps\h3` (its per-layer comparison before sampling uploads the weights, so step 0 is warm), "fresh" a folder without references where step 0 includes the first upload of all 21 GB (about 12 s).

| Run | Load | Steps (ms) | Sampling | Result |
|---|---:|---|---:|---|
| Turbo INT8, 8 steps, reference | 18.5 s | 4757 3498 3603 3509 3630 3523 3658 3554 | 29.7 s | video 0.3053642975063925, audio 0.0918839879386419: identical to before |
| Turbo INT8, 8 steps, Spectrum period 2, reference | 16.2 s | 3963 0 3796 51 3686 53 3724 3699 | 19.0 s | video 0.415, audio 0.123 against ComfyUI's 8 steps; PSNR 18.05 instead of 21.74 dB |
| Base FL2VA, 3 steps, fresh | 20.3 s | 15961 3753 3575 | 23.3 s | no LoRA |
| Base + TaoMate 3-step LoRA merged, 3 steps, fresh | 41.7 s | 18154 4046 3495 | 25.7 s | |
| Base + turbo 8-step LoRA (rank 384/128) merged, 8 steps, fresh | 99.6 s | 17916 4139 4373 3591 3636 3562 3684 3583 | 44.5 s | same composition as the pre-merged turbo INT8 model |

Rejected: applying LoRAs per step as an extra low rank product (`MossLora.Applied`, what Moss does) took 17.1 / 13.2 / 9.2 s per TaoMate step and 27-45 s per turbo LoRA step: the low rank products are cheap, but every linear writes a full size f32 delta (qkv 14865 x 21504 = 1.28 GB) and its sum next to the base output, which pushes the streamed weights out (eviction rounds, synchronizations, idle host gaps of about 250 ms). ComfyUI also merges into the int8 weights (requantized with stochastic rounding). Merging costs load time instead: `TensorFile_Tuned` adds rank 384 to a 21504 x 5376 matrix in about 450 ms on 32 pool threads (11.7 instead of 4.6 multiply-adds per ns per thread with tiles of 8 rows x 128 columns), plus 230 ms to join its 10M factor numbers in Strict.

Spectrum on H3 forecasting the hidden residual of the last block (like upstream) saved little or got slower: 26.6 s with a flat history (its concatenation of 80M-number anchors has no GPU kernel: about 1 GB down and up per remembered step), and with anchors kept as separate bf16 lists the remembering steps alone took 5-7.9 s instead of 3.6 s (GPU-only lists can't be spilled, ../Strict/improvements.md 13). Forecasting the output (1.4M numbers) keeps every actual step at 3.7 s and costs 0-50 ms per forecast, with the same error level as the feature forecast (0.415 against 0.42).
## RTX 3080 end-to-end timing correction (2026-10-24)

Compiler/runtime: Strict 30781d21. The resident runner uses its explicit workspace release and optional stored PNG output.

The user's 6-6.5 s ComfyUI measurement is 720x1280 with the model already loaded. The previous 9.70 s Strict number restarted the process for each image, so it was not a fair resident-model comparison. The old sampling-only win also did not establish an end-to-end win.

Implemented resident generations with the same loaded Moss model and VAE, stage timing, and optional stored-DEFLATE PNGs. Four warm production passes (two processes, each generating three times; first pass excluded) take 4.098 / 4.179 / 4.098 / 4.139 s. Full inference executes every time with the saved conditioning/noise; it does not reuse a generated image. Same int8 DMD model, 720x1280, eight Euler steps, CFG 1, Spectrum off. Core locked at 1695 MHz, actual CUDA memory 9506 MHz. Conditioning is precomputed in both implementations; UI/display and text encoding are excluded from the matched comparison.

| Warm generation stage | Strict | Installed ComfyUI, headless |
|---|---:|---:|
| Sampling | 3.155-3.171 s | 3.159-3.164 s |
| VAE decode and readback | 0.628-0.634 s | 0.267-0.268 s |
| Pixel conversion | about 0.026 s | 0.005-0.007 s |
| PNG saving | 0.271-0.315 s | 0.078-0.079 s |
| Complete generation, including boundary cleanup when another follows | **4.098-4.179 s** | **3.511-3.516 s** |

Strict remains roughly 0.6 s behind the matched headless pipeline; the remaining gap is mainly VAE decoding and PNG encoding. Do not describe the 4.1 s result as proof of beating ComfyUI's backend. A fresh Strict process still takes 9.057 s in the single-image check (2.988 s loading, 5.605 s generation, plus startup/shutdown), so cold loading remains a separate issue. The user's loaded-model comparison is the relevant warm target.

Two avoidable costs were identified and fixed:

- PNG compression cost about 0.92-0.95 s. Existing Deflate.Stored reduces it to about 0.28 s without changing decoded pixels; the portrait PNG grows from 2994029 to 3688028 bytes. ImageSaver's shared default remains compressed. MossGenerate defaults to the faster stored PNG; --compressed selects the old mode.
- After VAE decoding, retained free workspace left Moss streaming weights on subsequent evaluations. nsys showed mean HtoD of 1.991 GB per evaluation for warm steps 1-6, despite the model fitting before decoding. Explicit GpuMemory.Released between resident generations collects dead temporaries and returns unused large pool blocks without evicting live copies. Afterward, steps 1-6 upload only 2048 bytes; traced windows fall 476.91 -> 380.66 ms. A first-evaluation reload remains after decoding, so average upload over steps 0-6 is still about 370 MB/evaluation. Automatic H3 allocation/eviction policy is unchanged.

Global pool-first eviction and a spilled-row free-space guard were tested only in generated-host diagnostics; neither gave a robust improvement and neither is shipped. The explicit phase boundary did improve the full pipeline and is used by the resident runner. It increases warm VAE decode from about 0.55 to 0.63 s because workspace must be allocated again; total generation still improves.

Validation: resident-repeat check failed before implementation and now passes; PNG option's regression failed before the branch and passes in VM/native tests; native cache test checks collection/drain counts and retained live cache entries; compiler native mapping tests and stage 2/3 fixpoint pass. Full-size stored/compressed/resident images decode to identical pixels. The merged cache test's duplicate allocator mocks and missing pool DLL exports were also repaired so all budget, pool and lifetime checks run together.

Evidence: D:/StrictDumps/moss-end-to-end. Production logs/JSON: harness/portrait-42/turbo8/run-{0,1}; fresh-process case: single/portrait-42/turbo8/run-0; ComfyUI: comfy-end-to-end.json and comfy.py; before/after traces: profile-before.sqlite / profile.sqlite and phase-comparison.json. Clock limits restored after timing. Reproduce with compare_moss.py --resident-runs 3; see [Generation.md](Generation.md).

## RTX 3080 sampling tradeoffs: 25, 8 and 4 steps (2026-10-24)

36 images completed: lighthouse, portrait with hands and cafe menu; seeds 42 and 1234; 720x1280. Shared conditioning and CPU noise within each case. Strict 7e61d34e (includes H3's fixed-budget memory accounting) and the MossGenerate/MossSampling changes. Locked core 1695 MHz / actual CUDA memory 9506 MHz throughout all 476 high-load telemetry samples; maximum sampled GPU temperature 64.56 C, no thermal slowdown flags. Limits reset after timing.

| Variant | Sampling median | Range | Whole process median |
|---|---:|---:|---:|
| base25 | 20.422 s | 20.370-20.674 s | 25.75 s |
| spectrum25 | 13.448 s | 13.363-13.744 s | 18.37 s |
| turbo8 | 4.413 s | 4.380-4.462 s | 9.70 s |
| spectrum8 | 4.146 s | 4.084-4.215 s | 9.58 s |
| akhaliq4 | 2.892 s | 2.859-2.900 s | 8.18 s |
| rapid4 | 2.922 s | 2.900-2.936 s | 8.29 s |

Sampling includes text projection/setup, initial weight uploads and final latent synchronization; loading and decode/save are separate. Whole-process times include those costs. The writable CUDA JIT cache is warmed. An initial uncached setup took about 28 seconds; it is excluded from this matrix. The executable is restarted for each image, so these are not resident-model steady-step-only timings.

Spectrum reduces 25-step sampling time by 34.2% (1.52x faster) with 14 actual evaluations plus 11 forecasts per CFG branch, versus 25 actual evaluations per branch. On the eight-step turbo it saves 6.0% with only one forecast. Four-step merged adapters save about 34% sampling time against the existing eight-step turbo, rather than a full 2x because setup/initial uploads remain. They are comparison options, not selected final-quality defaults.

The 25-step preset is Euler/shift 5, ordinary CFG 3 with zero conditioning, no negative block skip. It is not the upstream release's negative-block-22-skip configuration or ComfyUI's ER-SDE quality preset. Eight-step uses the current merged DMD/Euler setup, CFG 1. akhaliq's pinned current root LoRA is Euler-native; darask0 uses its required custom data-space loop, Q, step codes and replacement f32 time weights/biases. Both speed versions are merged/requantized into int8 Convrot with the existing merge.py; this may lose small LoRA deltas.

Quality review: all 36 images were retained rather than rejected for differences. The four-step examples are coherent and, on these prompts, generally sharper and render the cafe text better than the existing eight-step model. They change composition and appearance substantially. Spectrum stays visually close on the inspected comparisons, but this small set does not establish a universal quality preset. Gallery: D:/StrictDumps/moss-speed-comparison/index.html; contact sheets lighthouse-comparison.jpg, portrait-comparison.jpg and menu-comparison.jpg. Full-resolution PNGs and commands/timings are in each case folder.

Unmerged tests on lighthouse-42: akhaliq completed, sampling 48.004 s, process 106.49 s (load 35.46 s, decode/save 22.16 s). darask0's unmerged path timed out after 240 s. Those paths use Strict's existing 16-bit low-rank loading, not upstream's fp32-LoRA configuration. Int8 merging took about 10.6 / 10.8 s once and matched all 252 projection modules for each adapter. It produced coherent images and is the speed comparison's default.

The new generation runner omits per-layer reference comparisons. It uses independent explicitly configured Spectrum histories for CFG branches; partial construction without explicit history initially crashed on the first real anchor, while the existing Configured helper resolves it. This is not presented as a generic compiler lifetime fix. No shared compiler change was needed for the sampler work.

H3 fixed-budget allocator check on 3080, two alternating pairs against the prior trim build: budget 395.33 / 395.67 ms steady evaluations, previous 403.17 / 396.67 ms. Mean 395.50 versus 399.92 ms (about 1.1%); variation is too large to claim a substantial gain. Exact errors and PNG stayed unchanged. The H3-sized ~20% improvement does not transfer to this Moss fixture. Diagnostics: D:/StrictDumps/bf16-attention/{budget1,basebudget1,basebudget2,budget2}.json. Copy-stream prefetch is still the 4090 agent's separate work; no such change was available at the final fetch.

Reproduction, adapter provenance, timing definitions and limitations: [Generation.md](Generation.md). Upstream Spectrum coverage: [Spectrum.md](Spectrum.md); adaptive scheduling and ER-SDE integration remain missing.

## RTX 3080 precision results (2026-10-24)

Same 720x1280 merged-int8-Convrot Moss fixture, eight Euler evaluations, steps 1-6 averaged. Locked core 1695 MHz, actual CUDA memory 9506 MHz; no thermal slowdown. Compiler c1829a08, model parent 502da91 plus the precision changes described in Handoff.md. F32 control uses the same compiler/runtime as bf16.

| Alternating pair | F32 residual/QK (ms) | BF16 residual/QK (ms) |
|---|---:|---:|
| 1 | 405.50 | 394.67 |
| 2 (reverse order) | 406.00 | 395.00 |
| 3 | 406.17 | 396.00 |
| Mean | 405.89 | 395.22 |

Saving 10.67 ms/evaluation (2.63%). Fresh warmed ComfyUI at identical clocks: 402.43 / 402.56 ms, mean 402.50 ms, so Strict is 7.28 ms (1.81%) faster per steady evaluation. ComfyUI reference latent error is zero. Historical unlocked 363-366 ms numbers are not comparable.

| Accuracy / process metric | F32 | BF16 |
|---|---:|---:|
| Step 0 error | 0.007891071269124554 | 0.00909505812896113 |
| Final latent error | 0.2691597633635198 | 0.4578933680652566 |
| PNG PSNR against ComfyUI | 28.74 dB | 23.30 dB |
| Full diagnostic process | 16.0-16.5 s | 24.3-25.0 s |
| Peak GPU memory | 8658-8788 MiB | 9604-9606 MiB |

The quality tradeoff is intentional under the user's bf16 preference; equivalence is not claimed. F32 PNG SHA256: `50ab7c7f2c0798e57ac40097061c9d778a6401dfe7b101d82f7dea841eda4913`. BF16: `4aaf91a74757676d7f0a6dafabad43a58e94ffff4c13cc500fea78f44055d972`, repeated exactly. QK-only bf16 measured 27.08 dB. The entire debug harness remains slower: CPU comparisons/phase transitions need investigation. Do not report an end-to-end process speedup or lower peak VRAM.

nsys steady windows (profiler overhead included): 410.37 -> 399.19 ms; kernel sums 402.95 -> 391.40 ms; busy.py target idle 7.37 -> 7.75 ms. Copies: 2048 bytes HtoD and 1843200 bytes DtoH per steady evaluation (latent comparison).

| Kernel category, ms/evaluation | F32 | BF16 |
|---|---:|---:|
| GEMM | 195.30 | 194.04 |
| Attention | 127.86 | 125.74 |
| Quant/Hadamard | 24.28 | 24.13 |
| Rotated | 14.19 | 14.03 |
| Gated | 14.17 | 8.27 |
| Modulated | 12.95 | 10.89 |
| Activated | 11.38 | 11.52 |

Native bf16 attention uses a noinline constant-format specialization: otherwise both predicated QK formats occupy tensor issue slots (42.1M -> 63.1M tensor instructions; math-pipe throttle 42.5 -> 56.6%). Specialized isolated attention: 3.009 ms bf16 against 3.038 ms f32. GEMM emitter unchanged. Flat four-wide loops accept arbitrary independent linear reads; grouped Modulated/Rotated vectorization is still pending.

Unused pool blocks are trimmed before large activation allocations under memory pressure. Global reserve, power-of-two classes and weight-cache policy remain unchanged. Eager lending, finer classes, global 1 GiB reserve and Readable comparison materialization were rejected. Inlining Rotated's half-width expression exposed a separate full-size miscompile; original arithmetic is retained and the bug is recorded in ../Strict/improvements.md.

All scripts, raw counters, paired JSONs and rejected patches: `D:/StrictDumps/bf16-attention`; final paired binaries `model-trim` / `model-trim-base`, traces `profile-final-*`. Full validation and next steps: [Handoff.md](Handoff.md).

## RTX 3080 GEMM analysis (2026-10-24)

[Full counter/SASS report](GemmStalls3080.md). Strict `4d1cafc9`, installed comfy-kitchen 0.2.39, identical prequantized data, checked 50-launch graphs and six alternating rounds. Final lock 1695 MHz core / 9506 MHz actual CUDA memory; 156 high-load samples held these clocks, no thermal slowdown. Final clean medians, Strict / CUTLASS: qkv 1198.8/1125.2 us, o 443.2/392.0, up 2143.8/2040.8, down 1068.2/936.6. Equivalent gap 15.09 ms per 42-layer step; telemetry repeats 17.40/15.47 ms. These are isolated-kernel estimates, not full-model improvements.

Fixed installed CUTLASS controls isolate roughly 5-6 ms/step from its smaller o tile and Stream-K down distribution; approximately 8-10 ms remains in the ordinary-tile comparison. Source samples locate the qkv/up excess in strided scale/bias setup and the larger epilogue, with essentially equal main-loop totals. The historical 14.6/13.2 warp-cycle gap does not by itself identify a missing pipeline optimization: current Fast executes fewer instructions. No production change was made. Await user review before emitter changes; validate A on 3080 when the 4090 agent commits it. The 4090 results below retain their own fixture and clock conditions.

## RTX 3080: A confirmed, B1 rejected, C inconclusive (2026-10-24, locked 1695/9756)

Reference for every run: PNG sha256 `50ab7c7f...`, latent error 0.2691597633635198, PSNR 28.74 dB.

| | Result |
|---|---|
| A, nsys idle per steady step | 12.3 -> 3.4 ms (traced step window -9.6 ms) |
| A, quiet paired full runs, steps 1-6 | baseline 415.5 / 416.3 ms, A 410.2 / 406.3 ms (5-10 ms saved) |
| ComfyUI, same locks, warm evaluations | 403.4-403.5 ms; Strict with A is 3-7 ms behind |
| B1 (Activated fused into Hadamard/quantize) | bytes identical, but fused 632.8 vs separate 587.5 us per layer (+1.9 ms/step), reversed run confirms: rejected |
| C small tile (128 rows, 4 stages, wave-aware) | isolated down -4 %, o mixed, qkv/up slower, in-model down unchanged, full pairs +3 / -4 ms: not merged (branch `codex/gemm-small-tile-3080`, 35e8cc0b) |
| C scale loads, hoisted result-format branch, dense scale metadata | slower / flat / no broad win |
| C separate BF16-output entry ("brain"), same run, alternating, medians us, baseline / brain / CUTLASS | qkv 1126.7 / 1117.8 / 1067.2, o 417.4 / 421.9 / 382.4, up 2043.8 / 2045.9 / 1930.2, down 1011.3 / 1026.6 / 896.1: within noise (-0.8 to +1.5 %), rejected (`brain-timings.log`) |

Scripts and logs: `D:/StrictDumps/gemm-c-implementation`.

# H3 on RTX 4090

## Kernels and idle per step (2026-10-24, unlocked, nsys `--cuda-graph-trace=node`, one step)

| Build | Kernels | Idle | Steps under nsys |
|---|---:|---:|---:|
| Strict fc3817b3 | 3504 ms | 520-570 ms | 4.01-4.08 s |
| + norm weights at 32 bits | 3394 ms | 506-579 ms | 3.89-3.96 s |
| + fused fc2 quantize, H3Final loops on the GPU | 3270 ms | 613-805 ms | 3.88-4.08 s |

- `Rope.Rotated` 192 -> 80 ms and `H3Block.Modulated` 87 -> 73 ms per step: with the 16 bit norm weights next to 32 bit lists no format copy fit and the generic kernel ran.
- fc2's 14336 wide input: `strict_device_hadamards` + `strict_device_quantize_tokens` (163 ms) -> `strict_device_hadamard_tokens` with two groups of 256 per warp (63 ms).
- Errors: the norms and the fused quantize are exact (final video error 0.3227169, audio 0.1445423). `H3Final.Unpatched`/`Unpacked` on the GPU keep the final layer's output there, so the Euler steps run in 32 instead of 64 bit (the host round trip made it a host list of doubles): step 0 video error 0.0024212779 -> 0.0024211848, final 0.3054 and 0.0919 (8 steps amplify small differences).
- Idle: 11 of the 50 blocks (4.8 GB) go up from their pinned copies in order on the one stream (0.21 s, the GPU waits at the quantize before each of their GEMMs), a few eviction rounds (gaps of 36-113 ms with driver frees, they move when the memory pattern changes) and the step boundary.
- ComfyUI (2026-10-23 profile, the text encoder share included): attention 2.61 s, GEMM ~1.25 s, convrot int8 quantize 0.17 s, other elementwise 0.14 s, RoPE 0.07 s per step.
- GPU memory counted against a fixed budget (Strict, `strict_gpu_budget`): plain runs used to be 0.8 s per step slower than under `nsys -t cuda`, because `cuMemGetInfo` lagged the host and the eviction rounds drained the rows uploads take over (76 driver allocations of 18.7 GB and 27 rounds per step). Alternating plain runs, 8 steps:

| Build | Steps 0-7 (ms) | Total |
|---|---|---:|
| Budget | 4865 3669 3630 3607 3788 3622 3622 3646 | 30.5 s |
| Before | 4600 3990 4760 4579 4536 4593 4973 4518 | 36.6 s |
| Budget | 3872 3819 3831 3815 3841 3865 3830 3826 | 30.7 s |
| Before | 5049 3902 5004 5059 4742 4891 5257 5227 | 39.1 s |

  Identical errors; 30 driver allocations of 2.6 GB and 4 rounds per step. ComfyUI took 4.15-4.38 s per step (2026-10-23).
- Spilled rows go up ahead on a copy stream (Strict 2354fc5c, `strict_gpu_prefetch`): each spill remembers the spill that went up after it, the next one goes up after the launches so far while they run, the launches wait at its first use. Under nsys 150-170 of the 280 ms of uploads per step now overlap kernels (none before, about half of the 6.5 GB per step still goes up in order where the chain breaks), GPU idle 370-640 -> 225-335 ms. The same commit stops `strict_gpu_obtain`'s trim of unused pool blocks while rows are spilled: with the 3080 session's merge (c1829a08) 3 of 9 runs took 10-15 s for steps 0-2 (285 and 245 trims with a GPU wait each, freeing 130 and 96 GB of blocks the next layers allocated again). Alternating plain runs, master 7e61d34e against 2354fc5c:

| Build | Steps 0-7 (ms) | Total |
|---|---|---:|
| Copy stream | 4744 3563 3568 3555 3536 3517 3585 3478 | 29.5 s |
| Master | 10032 9872 4405 3875 3770 3842 3924 3662 | 43.4 s |
| Master | 3939 4474 3854 3950 3689 3783 3715 3928 | 31.3 s |
| Copy stream | 4768 3569 3531 3523 3656 3522 3501 3518 | 29.6 s |

  Steady steps 3.82 -> 3.54 s, 15-19 % faster than ComfyUI's 4.15-4.38 s, identical errors (final video 0.3053642975063925, audio 0.0918839879386419). A first version that forked each upload from the start of its graph overlapped 53 of 277 ms (the GEMM that needs the rows comes early in the graph) and gained nothing.

### 3080 against ComfyUI per category (nsys, ms per evaluation, locked 1695/9756)

Strict `profile-c-baseline` (9 evaluations incl. cold), ComfyUI `comfy-prof/comfy.sqlite` (16 evaluations, same fixture, `comfy_locked.py` copy):

| | Strict | ComfyUI |
|---|---:|---:|
| int8 GEMM | 195.5 | 190.4 |
| Attention | 128.9 | 153.4 |
| Quantize/Hadamard (+ Activated) | 24.2 (+11.5) | 37.0 (Activated fused) |
| Rotated vs `rope_kernel` (q and k in one launch) | 14.3 | 9.2 |
| Gated + Modulated vs `elementwise_kernel` + `adaln_kernel` | 27.1 | 16.4 |

Kernel sums are about equal (Strict ~402, ComfyUI ~408); Strict loses on idle (3-6 ms) and on elementwise. ComfyUI runs the whole residual stream, gate/modulation and q/k in bf16 (checked at runtime: `gated_residual` gets bf16, bf16, bf16), Strict keeps the residual and rotated q/k in f32.

Fusions on the 3080 (`benchmark_fusion.py`, bit-identical, per layer pair): B1 separate 582.9 us, fused 600.5 us (-0.74 ms/step, rejected); B3 separate 322.6 us, fused 297.4 us (+1.06 ms/step for one pair per layer, about 2 ms with both, not worth a day of work).

Why the elementwise kernels are slow, measured with ncu on the 3080: Modulated (bf16 in) is L1/TEX bound at 88.7 % with 58 % DRAM, about 97 executed instructions per element (bounds checks, failure stores, index math per number). Gated moves 115 MB at 685 GB/s with the f32 residual, which is its floor.

bf16 residual experiment (Gated returns `BFloat16s`, batching relaxed to any number of linear reads, `width` as a Gated parameter; patches `gated-bf16-*.patch`): Gated 167 -> 110 us per call (-4.8 ms/step), Modulated 155 -> 128 us (bf16 input), but PSNR against ComfyUI 28.74 -> 25.43 dB (latent error 0.269 -> 0.350) and full steps 1-6 only 401 / 411 / 401 against 410 / 410 / 410 ms (alternating, the candidate is bimodal 401 or 411, the baseline always 410). Not committed: an earlier agent rejected the same precision change, needs a decision. Also tried and dropped: bf16 branch in `strict_device_squares` (Rotated 173 -> 164 us), lowering the bf16 list threshold from 2^20 to 2048 (VAE convolutions 11 -> 127 ms), Modulated/norm weights as bf16 lists.

# Moss on RTX 4090

## bf16 residual, q/k and native bf16 attention from the 3080 session (2026-10-24, 4090, locked 2400/10251)

Strict 7e61d34e (c1829a08 merged) with StrictComfy 7712c60 against Strict 793f4b79 with cb1f417, alternating, first run of each new build left out (cold):

| Build | Steps 1-6 | Step 0 error | Latent error | PSNR | PNG |
|---|---:|---:|---:|---:|---|
| f32 residual and q/k | 161.2 / 161.8 / 162.5 ms | 0.006325 | 0.2229 | 32.36 dB | a439d0b78f51c944 |
| bf16 residual and q/k, native bf16 attention | 151.0 / 154.7 / 152.3 ms | 0.006656 | 0.2184 | 32.04 dB | 81ca899073054026 |

9.5 ms per step faster (5.9 %) for 0.3 dB (the 3080 fixture lost 5.4 dB). The whole diagnostic run takes 22 -> 38 s: the step 0 comparison after every layer reads the bf16 outputs on the CPU, 8.4 -> 21.6 s for the 42 layers (the new compiler with the f32 MossBlock: 8.9 s); the VAE decode takes 0.93 -> 1.2-1.6 s since c1829a08 (not its trim: 1.3 s without it). The copy stream commit (2354fc5c) gives the same PNG, steps 150.8-152.8 ms.

## bf16 residual like ComfyUI: faster, but further from ComfyUI (2026-10-24, 4090, locked 2400/10251)

ComfyUI keeps the residual stream in bf16 (`gated_residual` is `x.addcmul_(y, gate)` on the bf16 `x`, `rms_adaln` reads it). Strict keeps it in f32 (`MossBlock.Gated` returns `Numbers`), which is most of its extra elementwise bytes (Gated 111 against 67 MB, Modulated 67 against 45 MB per call). Measured by returning `BFloat16s` from `Gated` (residual) and also from `Rotated` (q/k), Strict master e7752f3f, one run each in order res, both, master, res:

| Variant | Steps 1-6 | Step 0 error | Latent error | PSNR |
|---|---:|---:|---:|---:|
| Master (f32 residual and q/k) | 163.8 ms | 0.006325 | 0.2229 | 32.36 dB |
| bf16 residual | 156.5 / 156.8 ms | 0.006809 | 0.2922 | 27.47 dB |
| bf16 residual and q/k | 159.2 ms | 0.006659 | 0.2750 | 28.90 dB |

The bf16 residual saves 7 ms per step (4.3 %) but moves 31 % further from ComfyUI's own result: both round the residual to bf16 in different places, so their errors add up instead of cancelling. bf16 q/k make attention slower (the bf16 key path converts through f32 to f16). Not taken; the f32 residual stays.

## A: step-boundary round trips removed (2026-10-24, 4090, locked 2400/10251)

Paired alternating runs, steps 1-6 mean: baseline 187.0 / 182.7 / 187.8 ms, candidate 169.8 / 161.7 / 163.5 ms; idle per step 23.2 -> 5.0 ms (only the harness's latent download is left), no uploads or graph re-instantiation per step, identical errors and PNG. Details, the rule behind the round trips and the B1 estimate: [Handoff.md](Handoff.md), section "A done on the 4090".

## Locked-clock baseline (2026-10-24, Strict master e1cb907b + native-flags build fix 23e61750)

Same fixture, model and settings as below. After the repaste (no thermal throttling since, see the GPU section of Handoff.md) every timing run here used `nvidia-smi -lgc 2400,2400` and `-lmc 10251,10251` (10451 MHz effective with the +200 MHz offset). The lock is for A/B timing only, production runs stay unlocked. 2400 is the highest lock tested that holds steadily: locked at 2550 the clock reads 2550 but bf16 GEMM throughput swings between 139 and 160 TFLOPS, against 151-153 at 2400 and 163-166 unlocked. Memory is not slowed by the lock: under CUDA the driver uses the P2 point (10451 MHz effective with +200) whether locked or not (`-lmc 10501` changes nothing); 10701 MHz for CUDA would need NVIDIA Profile Inspector's "CUDA - Force P2 State" off. Both held in every one-second sample, no throttle flags, hotspot at most 61 °C in Moss runs (84 °C after 60 s of bf16 GEMM stress at 370-376 W). ComfyUI used the same conditioning, tokens, noise and sigmas, bit-identical to the fixture, and reproduced the fixture's final latent bit for bit.

Run to run, plain runs without a profiler, steps 1-6 mean:

| | Runs | Steps 1-6 mean | Spread |
|---|---|---:|---:|
| Strict | 3 | 178.7 / 176.8 / 176.8 ms (single steps 165-185, step 0 196-216) | 1.9 ms (1.1 %) |
| ComfyUI (`dump.py`, clean warm runs 1 and 2) | 2 | 224.2 / 264.7 ms (step 0 351-392) | 40 ms (18 %) |

Strict's errors were identical in every run: step 0 0.006325429288679978, latent 0.2228712467784174, PSNR 32.36 dB.

Per step (nsys `-t cuda --cuda-graph-trace=node`, Strict steps 1-7, ComfyUI clean run 1, ms):

| | Strict | ComfyUI |
|---|---:|---:|
| int8 GEMM | 68.9 (168 calls, q/k/v fused) | 68.3 (259 calls, 6 per layer, CUTLASS `Kernel2`) |
| Attention | 49.2 | 55.3 (cuDNN flash) |
| Elementwise | 29.1: Gated 8.2, Activated 7.2, Rotated 7.0, Modulated 5.2, other 1.5 | 11.1: elementwise 5.3, rope 3.6, adaln 1.4, other 0.8 |
| Quantize / Hadamard | 9.1 | 14.5 (situ included) |
| Kernel sum | 156.4 | 149.2 |
| Idle (plain wall − kernels) | ~21 (nsys: 23.2) | ~75-116 |
| Wall (plain runs) | 177 | 224-265 |

Strict is 21-33 % faster than ComfyUI on the 4090. ComfyUI is host bound and uploads only 0.5 MB per step, so it doesn't stream weights. Strict's kernels take 7 ms more: elementwise +18, GEMM +0.6, attention −6.1, quantize −5.4.

Strict's idle is all at step boundaries; inside the 42 layers the gaps sum to 0.1-0.2 ms per step. Per step (`idle.py`):

| Cause | Idle |
|---|---:|
| Sigma sync: `Euler.Stepped` runs the 9-element `Sigmas` loop on the GPU and reads 2 values back | 6.5 ms |
| Rotation table: `MossBlock.Rotations` recomputed every step, 7.44 MB downloaded and 7.44 + 3.72 MB uploaded again | 10.5 ms |
| Time embedding: `Moss_Sinusoid` result read back | 2.0 ms |
| The harness's latent error check (`MossImage.Sampled`, 1.84 MB) | 2.3 ms |
| Graph re-instantiation after a sync (3-11 `cuGraphInstantiateWithFlags` per step) | in the gaps above |

Per kernel, one layer (layer 20 of the check pass, direct launches `STRICT_GRAPHS=0`, single-pass ncu with `--cache-control none --clock-control none`, DRAM peak 1003 GB/s, L2 72 MB; durations match nsys within 10 %):

| Kernel | µs | DRAM read / write MB | GB/s | % peak | L2 hit | Working set |
|---|---:|---:|---:|---:|---:|---|
| Activated | 174 | 110 / 55 | 945 | 94 % | 39 % | 178 MB, does not fit L2 |
| Gated (×2) | 82-95 | 56-59 / 16-19 | 915 | 91 % | 49-53 % | 111 MB, does not fit |
| Modulated (after Gated) | 71-74 | 34 / 27 | 818 | 82 % | 76 % | 67 MB, only 24 % of its f32 input still in L2 |
| Modulated (layer start) | 67 | 45 / 5 | 726 | 72 % | 59 % | input from DRAM |
| Rotated (q, k) | 81-87 | 24-26 / 42 | 785-805 | 78-80 % | 67-69 % | 70 MB, writes f32 |
| Hadamard after Modulated (×2) | 34 | 0-2 / 0 | — | — | 98-100 % | 22 MB bf16 input entirely in L2 |
| Hadamard after attention | 38 | 21 / 0.4 | 571 | 57 % | 63 % | |
| Hadamard after Activated | 106 | 49 / 19 | 642 | 64 % | 49 % | |
| GEMM qkv / o / up+gate / down | 449 / 160 / 746 / 376 | | 130-215 | 13-21 % | 97-98 % | compute bound: 73 / 68 / 78 / 77 % of int8 peak (629 TOPS at 2400 MHz) |

Write bytes are counted when they reach DRAM during the kernel; the rest leaves L2 later. Multi-metric ncu runs need 7 replay passes, which restore memory between passes and give colder caches and 5-30 % longer durations; only single-pass numbers are in this table.

Isolated GEMMs (3631 tokens, `C:/StrictDumps/prof/gemmbench.py`, the GEMM half of `benchmark.py`, bit-exact against CUTLASS), µs:

| Rows × width | Strict | CUTLASS |
|---|---:|---:|
| 9216 × 3072 | 448.3 | 433.1 |
| 3072 × 3072 | 154.3 | 144.0 |
| 16384 × 3072 | 750.1 | 717.7 |
| 3072 × 8192 | 368.2 | 355.3 |

That is 1721 against 1650 µs per layer, at most 3.0 ms per step.

Conclusions from the RTX 3080 that do not hold on the 4090:
- **Elementwise is not instruction bound here:** the kernels run at 72-94 % of DRAM peak, so only fewer bytes help. Fast division plus position reduction was flat at full clocks and gave Rotated −5 % only at 1350 MHz (3080 like).
- **Gated is not L2 bound:** its working set is 111 MB and it runs at 91 % of DRAM peak.
- **GEMM is not 20-30 % slower than ComfyUI:** in the model it is equal; per shape CUTLASS is 3.5-7 % faster.
- **Idle is not 3-5 %:** it is 13 % of a step (the host costs stay, the GPU part is shorter).

Ranked by expected ms per step on the 4090 (noise: about 2 ms, the spread of the run means):

| | Candidate | Expected | Confidence | Work |
|---|---|---:|---|---|
| A | Remove the step-boundary host round trips: host sigma schedule, no time embedding readback, rotation table once per run and GPU resident, runtime rule against GPU-only lists round-tripping, fewer graph re-instantiations. A whole-step CUDA graph is not needed, the 256-launch segments already keep the layers gapless; the host reads are what end them. | 18-21 ms | high | small to medium |
| B1 | Activated (situ) into Hadamard + int8 quantization: removes 104 MB DRAM per layer (4.4 GB per step) at 94 % of peak | 4.8-5.9 ms | medium-high | large: a read hook from a generated element function into the runtime Hadamard kernel |
| B3 | Gated + next Modulated in one grouped kernel: saves Modulated's 34-45 MB f32 residual read | 3.2-4.2 ms both pairs, 1.6-2.1 ms in-layer only | medium | medium in-layer, large across the layer loop |
| B2 | Gated into the o/down GEMM epilogue | 2.1-2.9 ms by bytes, up to 6.9 ms if hidden under the GEMM | low-medium | large; competes with B3 for Gated |
| C | GEMM stall gap against CUTLASS | ≤ 3.0 ms (more on the 3080: 8-17 %) | low | analysis first |

Decision (2026-10-24): A, then B1, here; C goes to the RTX 3080 agent.

Scripts in `C:/StrictDumps/prof`: `sensors.py` (GEMM load or `watch`, NVAPI hotspot, clocks, throttle reasons), `phase1.sh` (locked plain runs), `nsys.sh`, `cats.py` (categories, busy, idle per step window), `idle.py` (idle by host cause), `ncu1.sh` (single-pass ncu of real kernels, `--kill on`), `gemmbench.py`. Data: `C:/StrictDumps/phase1`. Under ncu, graph launches are very slow: profile with `STRICT_GRAPHS=0`.

## 2026-10-22 (unlocked clocks)

2026-10-22, 720×1280, 8 steps, `moss-dmd.safetensors`, fixture `artifacts/moss4090` (3600 image + 31 text tokens), `STRICT_GPU=all`, NVIDIA driver 616.92, clocks not locked (steps get 5-10 % slower as a run warms the GPU, so compare back to back).

### Step times

| Version | Warmed steps | GPU busy per step |
|---|---:|---:|
| ComfyUI (`dump.py`, second sampling run) | 223–261 ms | 147 ms |
| Strict a0e51c03 | 213–232 ms | 197–217 ms |
| Strict master b78fe8b2 (rows on 128 byte lines 453d73e8, exp2 c2577213) | 187–194 ms | 171–181 ms |

ComfyUI is host bound on the 4090 (GPU busy 60 % of a step), Strict is GPU bound (93 %). Master: step 0 error 0.006406, latent 0.2621, PSNR 29.34 dB (a0e51c03: 0.006357, 0.2547, 29.75 dB, the exp2 change). One of four master runs gave other numbers (step 0 0.006606, latent 0.2551, 29.65 dB); a0e51c03 repeated exactly in three runs.

### Kernel time per step (nsys, step 1, ms)

| Work | Strict master | ComfyUI |
|---|---:|---:|
| int8 GEMM | 65.9 (168 calls, q/k/v fused) | 68.3 (252 calls, CUTLASS) |
| Attention | 46.3 | 53.0 (cuDNN flash) |
| Hadamard + int8 quantization | 13.1 | 14.3 (`quantize_int8_rowwise_convrot64`) |
| Elementwise | 44.9: Activated 12.1, Rotated 9.1, Gated 8.2, Modulated 6.7, Scales 5.9, Columns 2.9 | about 11: elementwise 5.2, rope 4.0, adaln 1.3 |

Elementwise work is the whole remaining gap (34 ms). Strict's loops already move 0.7–1.4 TB/s (32 bit lists, L2 hits), so only fewer bytes help. ComfyUI keeps activations in bfloat16, fuses RMS norm and modulation (`rms_adaln`), q/k norm and RoPE (`rms_rope_split_half`), applies situ inside the ffn_down quantization and adds the gated residual in place.

### GEMM

Nsight Compute counters work for the Benjamin account (the earlier `ERR_NVGPUCTRPERM` came from the `CodexSandboxOffline` account). On a0e51c03, `strict_device_wholes` used the same 128×256×64 tiles and occupancy as CUTLASS but was L2 bound: 88.6 % L2 throughput, 2.14 GB read from L2 against CUTLASS's 1.27 GB (the ideal for these tiles), tensor pipe 59 % against 88 %. Its weight rows started 16 bytes into a sector, so every 64 byte row piece touched 3 sectors instead of 2; 453d73e8 starts them on a 128 byte line. `benchmark.py` with the aligned layout, µs per call:

| Rows × width | Strict | CUTLASS |
|---|---:|---:|
| 9216 × 3072 | 421 | 407 |
| 3072 × 3072 | 140 | 133 |
| 16384 × 3072 | 763 | 784 |
| 3072 × 8192 | 432 | 388 |

2.6 % in sum, about 2 ms per step: a CUTLASS bridge is not worth a native C++ dependency.

RoPE in isolation (`MossBlock_Rotated`, bfloat16): 107 µs generated, 94 µs without the sign checks of its divisions, 69 µs with the launch values specialized as well (the range analysis in ../Strict/improvements.md).

## Reproduce

```powershell
..\Strict\strict.exe build MossImage.strict
$env:STRICT_GPU = "all"
.\MossImage.exe artifacts/moss4090 C:/ComfyUI_windows_portable/ComfyUI/models/diffusion_models/moss-dmd.safetensors '""' C:/ComfyUI_windows_portable/ComfyUI/models/vae/flux2-vae.safetensors
C:\ComfyUI_windows_portable\python_embeded\python.exe benchmark.py
```

Kernel times: `nsys profile -t cuda --cuda-graph-trace=node` (Strict launches CUDA graphs, without node tracing nsys records no kernels). ComfyUI: `dump.py moss --dit moss-dmd.safetensors --lora --width 720 --height 1280 --runs 2` with `COMFYUI=C:\ComfyUI_windows_portable\ComfyUI`.
