# Moss on RTX 4090

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
