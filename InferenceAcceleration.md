# Inference acceleration assessment, 2026-10-23

## Measured Moss test after driver and overclock change

RTX 3080 10 GB, driver 621.01. User settings: core +82 MHz, memory +255 MHz, voltage slider +8%, power 106% (392.2 W), temperature target 86 C. No settings changed by this test. Both runs used the exact previously validated `production-gemm-fast/MossImage.exe`, merged turbo model, fixed conditioning/noise, 720x1280 and eight Euler steps. This isolates executable changes, but driver and overclock changed together; their individual effects cannot be separated.

| Run | Steps 2-6 mean | Logged step subtotal |
|---|---:|---:|
| Earlier driver/settings 1 | 380.6 ms | 2777 ms |
| Earlier driver/settings 2 | 382.2 ms | 2802 ms |
| Driver 621.01/new settings 1 | 379.6 ms | 2762 ms |
| Driver 621.01/new settings 2 | 379.0 ms | 2761 ms |

Mean steady improvement is about 0.55%, within observed run variation. Every logged numerical error and PNG SHA-256 matches the earlier baseline: `986b307e47af99d345eb09f44d6dce168d6fb1cec5e13d85dd61172067a3502e`. Latent relative error 0.3276526817290422, reported PSNR 27 dB. These are comparisons to the existing Comfy dumps, not new quality scores.

Telemetry: 92 samples at roughly one-second intervals, five at >=90% utilization. Maximum core 66.12 C, maximum sampled board power 389.47 W, busy core clocks 1905-1980 MHz. No sampled SW/HW thermal slowdown. This bursty inference check is not a sustained overclock stability certification; hotspot/memory temperatures remain unavailable. Evidence: `D:/StrictDumps/moss-driver621-oc/{summary.json,telemetry.csv,run1.log,run2.log}`. Earlier comparison: `D:/StrictDumps/codex-elementwise/gemm-scheduling/full-validation.json`.

## What Spectrum changes

Spectrum fits time-varying denoiser features with regularized Chebyshev polynomials and forecasts selected evaluations. It reduces full-network evaluations, rather than making each GEMM faster. The paper studies image/video diffusion; its headline speedups are not measurements for our models or hardware. [Paper](https://arxiv.org/html/2603.01623v2), [official implementation](https://github.com/hanjq17/Spectrum).

The inspected MOSS node stores real final-block image features, solves a small CPU system to obtain anchor weights, blends spectral/linear predictions, and reruns the current-time output head. Defaults include eight warmup evaluations, degree four and eight history points. Adaptive scheduling uses sampled-token p95/p99 error at real refreshes. It cannot measure true forecast error on a skipped evaluation. [MOSS implementation](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS/blob/main/nodes.py).

Its ER-SDE adapter additionally substitutes solver-space predictions based on two real guided-output anchors, preserving native noise/RNG handling. Therefore ER-SDE results do not isolate hidden-feature forecasting alone. Our current Euler path does not need that bridge. [Solver adapter](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS/blob/main/er_sde_bridge.py).

The node reports approximately 2-2.5x with its default setup; this is an upstream claim, not our result. [README](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS). The paper implementation is MIT; this node is GPL-3.0-only. A future Strict implementation should derive its math from the paper and use the official reference as appropriate, preserving source attribution for any reused code.

## Proposed diffusion experiment

This is a design, not an implemented or benchmarked approximation. Exact inference stays the default. Start with Moss/Euler, then H3; share only the fitted predictor and history/scheduling arithmetic when both adapters establish the same contract.

1. Capture actual final-block image features immediately after `Moss.Transformed`. Split `Moss.Output` so its timestep embedding and final normalization/modulation/projection can run with either actual or predicted image features. Preserve `MossImage.Sampled`'s sigma schedule and Euler update. Never replace the latent directly with a forecast hidden state.
2. First run in shadow mode: predict at each eligible evaluation but still execute the network. Compare predicted features and outputs to that real evaluation before adding its feature to history. Store only real evaluations as anchors. This measures local fitting error; a second, live-skip test is still necessary because shadow mode does not measure accumulated trajectory drift.
3. Compare hold-last, two-anchor linear prediction and regularized spectral prediction. Require spectral fitting to earn its memory/CPU overhead against these cheap baselines. Use FP32 accumulation and original feature precision first. Anchor compression is a separate approximation to measure later.
4. Start with one isolated skipped middle evaluation, followed by a real refresh. Use protected initial/final bands and a hard forecast horizon. Force real evaluation on nonfinite fits, missing history, memory budget failure or changed conditioning/model/LoRA/shape/schedule. Separate CFG branches. A refresh can limit future drift; it does not undo an already altered latent trajectory.
5. The 25/50-step non-turbo case is the primary target. Obtain a matching full model and baseline for those schedules before drawing conclusions from the turbo checkpoint. For eight-step turbo, use a deliberately shorter warmup/lower-order candidate only after shadow results; degree four already needs at least five anchors. Defaults with eight warmup steps provide zero skips here. Four-step turbo has even less room; do not compound its existing quality problems without measurements.
6. H3 needs its own adapter around the complete block stack and before `H3.final.Output`. Validate joint audio/video behavior and keyed/text conditioning together. Budget histories per token count; video is much larger than an image. SDXL/UNet models need another adapter with correctly defined denoiser features/output. Do not assume all model heads or schedules are interchangeable.

For this Moss shape, 3600 image tokens x 3072 channels x eight anchors costs 337.5 MiB at FP32, or 168.75 MiB at BF16, excluding pool rounding and scratch. Estimate remaining VRAM before enabling it. H3 must choose history size from a byte budget. Avoid copying full anchors to CPU each step. CPU should handle only the tiny basis/fit; a generated GPU weighted sum can read resident anchors once and accumulate FP32. Keep the normal compiler/emitter path, without handwritten model-specific CUDA.

Use actual completed per-evaluation cost. Correction: the historical eighth-step log of 11-12 ms did not wait for GPU completion because no next-step reference existed. Its remaining work was paid during the final latent comparison. Historical sums in the table above are logged subtotals, not complete sampling times. A completion-aware runner measures the eighth evaluation near the middle-step cost. Let T_full be full evaluation cost, T_predict forecast cost, N_full/N_forecast their counts, and T_fixed loading/encoding/VAE cost. Compare `T_fixed + N_full*T_full + N_forecast*T_predict + cache/validation overhead` with the same complete baseline. Record steady sampling and end-to-end latency separately.

Acceptance matrix: multiple prompts/seeds; text rendering, small objects, hands/faces, difficult compositions; fixed sampler/LoRA/seed comparisons. Record real evaluation counts, latency, peak memory, sampled-feature tail errors, final latent error and image quality. PSNR alone does not establish perceptual quality; include side-by-side review and a perceptual metric when available. For video also assess temporal consistency and audio synchronization. Report median and bad-tail cases, not only the best image. Expose a bounded skip/error budget and measured presets; do not promise a universal percentage quality slider.

## What transfers to autoregressive models

Forecasting the next token's final hidden state assumes continuity that the token stream does not provide. A changed token becomes future context and may change subsequent expert routes and KV/recurrent states. Thus a small hidden-vector error is not a bound on answer correctness. This is the main reason a simple layer/feature-skip rule can fail abruptly.

| Approach | Applicable path | Output/quality contract | Priority here |
|---|---|---|---|
| Verified draft/MTP blocks | Qwen/Flash with usable drafter | Exact greedy tokens if verification/rollback correct; sampled decoding needs acceptance/residual-sampling algorithm | First |
| Expert grouping, residency and CPU/GPU placement | Existing MoE models | Preserve selected experts and weights; placement can change numeric reduction order | First |
| Calibrated low-bit weights | Dense and MoE projections | Approximate; measure calibration/task errors and actual kernel support | Next |
| KV precision/eviction | Attention layers with KV state | Approximate; long-context retrieval may degrade abruptly | Workload-specific |
| Unverified layer/expert skipping | Existing dense/MoE models | Unbounded semantic risk without training/calibration | Experimental |
| Learned sparse/early-exit models | Compatible trained checkpoints | Architecture/training-dependent | Separate model support |

Verified speculation moves the tradeoff to acceptance rate and overhead rather than output quality. The original algorithm preserves the target sampling distribution with the required acceptance/correction rule. Strict's `Speculation`/`FlashSpeculation` currently use greedy verification; their equality checks do not establish sampled-decoding correctness. [Speculative decoding](https://arxiv.org/abs/2211.17192).

Concrete local work: `FlashSpeculation.Advanced` already verifies token blocks and truncates rejected suffixes. `FlashNext.Truncated` must restore both attention caches and Delta/recurrent/PLE state. Measure draft cost, accepted tokens, verification cost and state repair before adapting draft length. Continue the chat-prompt and 2/3-draft comparison already listed in Strict `improvements.md`. An adaptive horizon should maximize verified tokens per wall second, not acceptance alone.

Current local notes report Flash MTP 325 vs 391 ms/token on the same greedy text after batched experts and GGUF metadata reuse; these are historical measurements, not new driver results. They also report a failed LFU GPU expert cache at 600-650 ms/token: insertion and small CPU miss batches outweighed hits. Do not reintroduce that cache unchanged. Profile expert reuse across verified blocks and overlap justified transfers, keep miss work batched, and include warmup/insertion costs. Fiddler is a useful reference for computing cold experts where their weights reside instead of automatically uploading them. [Fiddler](https://github.com/efeslab/fiddler).

MoE's active-expert count is part of its trained architecture. Lowering that count on an existing checkpoint is a new approximation, not a free MoE optimization. Likewise PowerInfer's learned activation sparsity is not directly available in every SwiGLU Qwen checkpoint. [PowerInfer](https://arxiv.org/abs/2312.12456). LayerSkip trains useful early exits; attaching an LM head to arbitrary intermediate Qwen layers is not equivalent. [LayerSkip](https://arxiv.org/abs/2404.16710).

For Qwen3 and other dense autoregressive models, prioritize compatible drafters, prompt-prefix reuse and quantized resident weights. For Qwen3.8/Flash hybrid models, preserve their additional recurrent state during every speculative rollback. If MPT means MosaicML MPT, it belongs in the dense autoregressive group; MTP means multi-token prediction and is the existing draft-head mechanism discussed here.

For quantization, calibrate sensitive channels rather than uniformly discarding more bits; use real task evaluation and compatible fused kernels. [AWQ](https://arxiv.org/abs/2306.00978). Retaining important/recent KV entries is another quality-memory tradeoff, but published results on older architectures are not a guarantee for Flash hybrid attention/recurrent state. [H2O](https://arxiv.org/abs/2306.14048). Separate attention KV compression from recurrent-state precision.

## Next-Concept Modeling

NCM learns a multiscale residual VQ-VAE representation and a generator that predicts a level's codes in parallel, proceeding coarse to fine. Its reported 10-31x comparisons are to diffusion-language baselines on specified tasks, not Qwen3.8 or our token decoder. Main generators are around 100M parameters, with scaling to 725M; the paper notes fixed hierarchy costs and transition artifacts. [NCM paper](https://arxiv.org/html/2610.14162v1), [code](https://github.com/HazyResearch/ncm).

It requires a different trained tokenizer/generator/decoder; ordinary Qwen weights cannot acquire this behavior through a compiler flag. Supporting a released NCM checkpoint would be a separate NeuralNetworks frontend. Useful shared backend work includes quantized GEMM, attention, codebook lookup, multi-resolution tensor layouts and fast parallel output heads. Judge complete-text latency, first usable output latency, reconstruction/exact-string fidelity and task quality separately. It is a promising architectural direction, not evidence that unverified forecasting of current autoregressive hidden states is safe.

## Implementation order

1. Preserve exact compiler/GEMM/attention improvements across all models and keep current inference reproducible.
2. Moss shadow-only capture and hold/linear/spectral comparison; establish 25/50-step baselines and bounded memory before live skipping.
3. Validate a conservative live-skip Moss preset; then implement H3-specific joint audio/video validation.
4. In parallel as independent work, tune existing verified Flash MTP and expert-block execution using current profiling evidence.
5. Consider lossy weight/KV/model architecture changes independently, with task-specific quality tests. Do not combine several approximations before measuring each one.
