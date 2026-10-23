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

## RTX 3080 validation, 2026-10-23

Driver 621.01, existing user clock settings, same saved noise/conditioning and merged eight-step turbo model. Tests and source checks pass. The generated predictor uses the ordinary GPU loop/reduction emitter.

| Mode | Completed logged sampling sum | Forecasts | Final latent relative error to Comfy reference | PNG result |
|---|---:|---:|---:|---|
| Off | 3020 ms | 0 | 0.3276526817290422 | Exact historical Strict PNG |
| Shadow | 3258 ms | 0 | 0.3276526817290422 | Exact off PNG |
| On, warmup 5 | 2767 ms | 1 | 0.34812940160525724 | Changed; 36.20 dB PSNR against off PNG |

The forecast evaluation took 17 ms versus roughly 380 ms for an actual evaluation. Total improvement in this single sequential comparison is 8.4%; clocks varied, so this is not a preset benchmark. Live PNG mean absolute channel difference is 1.92/255, maximum 146/255. Visual comparison is close on this prompt but does not establish general quality. Decoder PSNR against Comfy's float reference changes from about 27 to 26.84 dB. Shadow feature errors at steps 5/6/7 are 0.699/0.491/0.825: hidden-feature error alone does not predict image quality. Full non-turbo 25/50-step schedules, multiple prompts/seeds, CFG and H3 remain unvalidated.

These full-size runs used a diagnostic monolithic host link. Normal split/ThinLTO host builds exposed a VAE transition regression: a late operation downloads about 450 MiB and continues on CPU. Same MLIR/PTX linked monolithically completes VAE in 429 ms with the exact baseline image; preventing inlining of `MossSpectrumImage.Generated` also completes the split build. This implicates host code generation/lifetime behavior, but does not prove an LLVM bug. Do not promote the adapter until its normal build passes the complete run.

Local evidence: `D:/StrictDumps/spectrum-moss/{validation.json,monolithic.log,shadow.log,on.log,off.png,shadow.png,on.png}`; dispatch profile `off-profile.sqlite`; generated-host relinking reproductions `build_{dispatch,monolithic,noinline}.py`. Diagnostic runtime/source changes are outside the shared branches.

Math/background: [Spectrum paper](https://arxiv.org/abs/2603.01623), [official implementation](https://github.com/hanjq17/Spectrum), [MOSS integration examined](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS). This implementation derives the fit from the mathematics; it does not copy the GPL node implementation.
