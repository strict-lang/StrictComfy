# Experimental MOSS feature prediction

`MossSpectrumImage` adds opt-in Chebyshev/ridge feature forecasting to the existing Euler path. Generic fitting and bounded real-anchor history live in Strict's `NeuralNetworks/Spectral{Fit,Solve,History}`. The MOSS adapter captures final-block image features and reruns the current timestep's normalization/modulation/projection head. Forecasts never become anchors. Each invocation constructs fresh history.

Modes:

- `off`: every transformer evaluation runs normally; default state setting.
- `shadow`: every transformer evaluation runs, with prediction errors logged once history is ready.
- `on`: scheduled middle evaluations use predictions. Initial warmup, insufficient history and the final evaluation force actual computation.

CLI arguments after the executable are `dumps model lora vae mode steps warmup period degree`. Example, from this folder:

```powershell
.\MossSpectrumImage.exe D:/StrictDumps/mix720 D:/ComfyUI_windows_portable/ComfyUI/models/diffusion_models/moss-image-int8-convrot-dmd-lenovo-finger.safetensors "" D:/ComfyUI_windows_portable/ComfyUI/models/vae/flux2-vae.safetensors off 8 8 2 4
```

Use `shadow 8 5 2 4` to measure fitting on the existing turbo fixture, or `on 8 5 2 4` for one forecast at zero-based step 5. Degree four needs five actual anchors. Default eight-step warmup deliberately skips nothing in an eight-step run. This adapter implements bounded periodic scheduling; the upstream adaptive sampled-token controller and ER-SDE bridge are not implemented.

History retains at most eight real features and disables forecasting if its estimated raw FP32 capacity exceeds 512 MiB. For 3600 x 3072 features, eight FP32 anchors cost 337.5 MiB before pooling/scratch. This is an estimate, not a measured peak-VRAM guarantee. Ridge is 0.1; fitted and last-two linear weights blend equally. Coordinates are normalized evaluation indices. Changing conditioning, model, LoRA, shape or schedule requires a fresh state.

## Upstream audit, 2026-10-24

Checked current Rkkss source against the checked-in Strict implementation. We implement the predictor core, not the complete ComfyUI node:

| Capability | Strict status |
|---|---|
| Capture final-block image features; rerun current-time head | Implemented |
| Degree-four Chebyshev fit, ridge 0.1, equal linear blend | Implemented |
| Eight real anchors; forecasts excluded from history | Implemented |
| Normalized step coordinates, warmup, actual final step | Implemented |
| Skip full transformer on forecast steps | Implemented |
| Manual scheduling | Fixed period, rather than upstream's expanding window |
| Adaptive local p95/p99 drift controller, detail guard, repair/cooldown | Missing |
| ER-SDE solver bridge and synchronized branch decisions | Missing; Euler only |
| Cache-placement choices, control/patch signature lifecycle | Missing; fresh invocation required |
| Forecast finite-value replacement/clamping | Missing; no claim of collapse protection |

The prior eight-step validation skipped only one evaluation. It does not establish upstream's advertised 2-2.5x gain, nor a validated 25-step quality preset. Default warmup eight skips nothing at eight steps. New comparison work uses separately labeled 25-step CFG and eight-step turbo cases; changing their outputs is expected. Image quality and generation time, not bit identity across sampling methods, determine useful presets.

Upstream sources: [node implementation](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS/blob/main/nodes.py), [ER-SDE bridge](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS/blob/main/er_sde_bridge.py). The local mathematical implementation remains independent; upstream GPL node code is not vendored.

## RTX 3080 validation, 2026-10-23

Driver 621.01, existing user clock settings, same saved noise/conditioning and merged eight-step turbo model. Tests and source checks pass. The generated predictor uses the ordinary GPU loop/reduction emitter.

| Mode | Completed logged sampling sum | Forecasts | Final latent relative error to Comfy reference | PNG result |
|---|---:|---:|---:|---|
| Off | 3020 ms | 0 | 0.3276526817290422 | Exact historical Strict PNG |
| Shadow | 3258 ms | 0 | 0.3276526817290422 | Exact off PNG |
| On, warmup 5 | 2767 ms | 1 | 0.34812940160525724 | Changed; 36.20 dB PSNR against off PNG |

The forecast evaluation took 17 ms versus roughly 380 ms for an actual evaluation. Total improvement in this single sequential comparison is 8.4%; clocks varied, so this is not a preset benchmark. Live PNG mean absolute channel difference is 1.92/255, maximum 146/255. Visual comparison is close on this prompt but does not establish general quality. Decoder PSNR against Comfy's float reference changes from about 27 to 26.84 dB. Shadow feature errors at steps 5/6/7 are 0.699/0.491/0.825: hidden-feature error alone does not predict image quality. Full non-turbo 25/50-step schedules, multiple prompts/seeds, CFG and H3 remain unvalidated.

The initial full-size runs used a diagnostic monolithic host link because split/ThinLTO builds exhausted VRAM entering the VAE. This is resolved by Strict's memory-pressure handling (e1cb907b and later): prepared kernel arguments are protected across nested dispatch, and cold dedicated host-backed GPU cache copies can be evicted after queued work finishes. Heap roots and GPU-only activations remain valid; evicted copies can be uploaded again. No model-specific noinline rule or compiler-link workaround is required.

On 2026-10-24 both compiler executables were rebuilt and the ordinary `strict build` path completed full-size off/on runs. Off reproduces the exact historical PNG; on reproduces the previously validated live-prediction PNG (SHA-256 031781db011b2613a310688f9e79992ad6dd1387ae031c9acc081095c8bd8d25). The forecast took 9 ms in this run. VAE decoding still costs 1.4-4.9 seconds in these runners versus 0.46 seconds standalone, so reducing transition overhead remains open. This change bounds cached GPU-copy lifetimes under memory pressure; it does not add precise heap-root maps or promise prompt collection of all dead model objects.

Current normal-build evidence: `D:/StrictDumps/spectrum-moss/{verified-off.json,verified-on.json,verified-off.log,verified-on.log}`. Initial experiment evidence: `D:/StrictDumps/spectrum-moss/{validation.json,monolithic.log,shadow.log,on.log,off.png,shadow.png,on.png}`; dispatch profile `off-profile.sqlite`; generated-host relinking reproductions `build_{dispatch,monolithic,noinline}.py`. Diagnostic runtime/source changes are outside the shared branches.

Math/background: [Spectrum paper](https://arxiv.org/abs/2603.01623), [official implementation](https://github.com/hanjq17/Spectrum), [MOSS integration examined](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS). This implementation derives the fit from the mathematics; it does not copy the GPL node implementation.

## H3 (RTX 4090, 2026-10-25)

`H3Spectrum` forecasts H3's output (the velocities of both streams, 1.4M numbers) from the last `degree + 1` actual outputs with `SpectralFit` (Chebyshev degree 1 by default, ridge 0.1, blended half with the line through the last two), coordinates are the unshifted flow time of the step's sigma, one anchor is held. Schedule: `warmup` actual steps, then every `period`-th step actual, the last step always actual (8 steps, period 2: A F A F A F A A like the H3 node's Euler cadence). Arguments are in [Generation.md](Generation.md).

This differs from the upstream node, which forecasts the residual of the last block (target rows only) and reruns the current timestep's head. That needs anchors of 14865 x 5376 numbers kept across steps; on the 4090 they come out of the memory the 50 streamed blocks rotate through, and actual steps slowed from 3.6 to 5-7 s (Performance.md "H3 LoRAs, step counts and Spectrum"). The output forecast measured the same error against ComfyUI's 8 steps (final video 0.415 against 0.42 for the feature forecast, 0.305 without skipping) at 19.0 instead of 29.7 s for the 8 steps. Judge the frames, not these numbers: they only compare against ComfyUI's unskipped output.
