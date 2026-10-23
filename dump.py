"""Reference dumps from ComfyUI's own code for the Strict ports. Run with ComfyUI's python_embeded:
  python_embeded\\python.exe dump.py moss [--out D:\\StrictDumps\\moss] [--width 576 --height 1024 ...], prompt from prompt.txt
  python_embeded\\python.exe dump.py h3 [--out C:\\StrictDumps\\h3] [--image first.png --length 124 ...], t2va or i2v at 480x832
  python_embeded\\python.exe dump.py h3decode [--out C:\\StrictDumps\\h3]   (H3Video.strict's strict-latent.txt decoded with ComfyUI's VAEs, both as H.265 MP4)
  python_embeded\\python.exe dump.py tinymoss   (tinymoss.safetensors here, for MossBlock.strict and the other Moss tests)
  python_embeded\\python.exe dump.py tinylora   (tinylora.safetensors here, rank-8 LoRA for tinymoss + outputs with it, for MossLora.strict)
  python_embeded\\python.exe dump.py tinyint8   (..\\Strict\\NeuralNetworks\\tinyint8.safetensors, generic int8 rows)
  python_embeded\\python.exe dump.py tinyh3     (tinyh3.safetensors here, for H3Block.strict and the other H3 tests)
  python_embeded\\python.exe dump.py tinyh3vae  (tinyh3vae.safetensors here, for the H3 video VAE decoder)
"""
import argparse
import json
import os
import sys
import time

COMFY = os.environ.get("COMFYUI", r"D:\ComfyUI_windows_portable\ComfyUI")
sys.path.insert(0, COMFY)
sys.argv = sys.argv[:1] + [a for a in sys.argv[1:]]
arguments = argparse.ArgumentParser()
arguments.add_argument("model", choices=["moss", "h3", "h3decode", "tinymoss", "tinylora", "tinyint8", "tinyvae", "tinyh3", "tinyh3vae"])
arguments.add_argument("--out", default=None)
arguments.add_argument("--prompt", default=None, help="default: prompt.txt next to this script")
arguments.add_argument("--width", type=int, default=None)
arguments.add_argument("--height", type=int, default=None)
arguments.add_argument("--length", type=int, default=124, help="h3 frames at 24 fps, snapped to the 17k+5 grid")
arguments.add_argument("--image", default=None, help="h3 first frame (i2v), t2va without")
arguments.add_argument("--steps", type=int, default=8)
arguments.add_argument("--scheduler", default="simple")
arguments.add_argument("--seed", type=int, default=42)
arguments.add_argument("--runs", type=int, default=2, help="sampling runs, the last one is timed and dumped")
arguments.add_argument("--dit", default=None)
arguments.add_argument("--te", default=None)
arguments.add_argument("--lora", nargs="*", default=None, help="LoRA names in the loras folder, each name[:strength]")
arguments.add_argument("--lora-strength", type=float, default=1.0)
arguments.add_argument("--vae", default=None)
arguments.add_argument("--audio-vae", default="minimax_h3_audio_vae_fp32.safetensors")
options = arguments.parse_args()
DEFAULTS = {"moss": {"width": 576, "height": 1024, "dit": "moss-image-sft-dit-int8-convrot.safetensors", "te": "qwen3.5_9b_int8_convrot.safetensors",
		"lora": [r"moss\dmd_turbo_lora.safetensors"], "vae": "flux2-vae.safetensors"},
	"h3": {"width": 480, "height": 832, "dit": "Minimax_H3_FL2VA_PRUNED_Turbo_8step_v1.0_768p_comfyui_INT8_.safetensors",
		"te": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "lora": [], "vae": "minimax_h3_video_vae_int8_convrot.safetensors"}}
for key, value in DEFAULTS["h3" if options.model.startswith("h3") else "moss"].items():
	if getattr(options, key) is None:
		setattr(options, key, value)
prompt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt.txt")
if options.prompt is None:
	options.prompt = open(prompt_file, encoding="utf-8").read().strip() if os.path.exists(prompt_file) else ""
	options.prompt = options.prompt or "A weathered lighthouse on a rocky cliff at sunset, waves crashing below, warm golden light, photo"
sys.argv = sys.argv[:1]

import logging  # noqa: E402
logging.basicConfig(level=logging.INFO, format="%(message)s")
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
from comfy.cli_args import args, enables_dynamic_vram  # noqa: E402
import cuda_malloc  # noqa: E402, F401 (sets the allocator before torch loads, like main.py)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import comfy_aimdo.control  # noqa: E402
if enables_dynamic_vram():
	comfy_aimdo.control.init(simple_vram_headroom=None if args.reserve_vram is None else int(args.reserve_vram * 1024 ** 3), nvml_pressure=not args.disable_nvml_pressure)
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
import folder_paths  # noqa: E402
import comfy.memory_management  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.model_management  # noqa: E402
# DynamicVRAM like main.py: without it the legacy ModelPatcher misjudges free VRAM on Windows and spills into shared memory
if enables_dynamic_vram() and comfy.model_management.is_nvidia():
	try:
		dynamic = comfy_aimdo.control.init_devices((d.index, int(args.vram_headroom * 1024 ** 3)) for d in comfy.model_management.get_all_torch_devices())
	except TypeError:
		dynamic = comfy_aimdo.control.init_devices(d.index for d in comfy.model_management.get_all_torch_devices())
	if dynamic:
		comfy_aimdo.control.set_log_warning()
		comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
		comfy.memory_management.aimdo_enabled = True
	logging.info(f"DynamicVRAM {'enabled' if dynamic else 'not available'}")


def saved(tensors, path, metadata=None):
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	save_file({name: value.detach().to("cpu").contiguous() for name, value in tensors.items()}, path, metadata=metadata)
	print("wrote", path, f"{os.path.getsize(path) / 1e6:.1f} MB")


def moss():
	out = options.out or r"D:\StrictDumps\moss"
	os.makedirs(out, exist_ok=True)
	started = time.perf_counter()
	clip = comfy.sd.load_clip([folder_paths.get_full_path_or_raise("text_encoders", options.te)],
		embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=comfy.sd.CLIPType.MOSS_IMAGE)
	tokens = clip.tokenize(options.prompt)
	conditioning = clip.encode_from_tokens_scheduled(tokens)
	encoded = time.perf_counter()
	peaks = {"encode": torch.cuda.max_memory_allocated() / 2 ** 30}
	context = conditioning[0][0]
	token_ids = [token[0] if isinstance(token, tuple) else token for token in next(iter(tokens.values()))[0]]
	saved({"context": context, "tokens": torch.tensor([int(t) for t in token_ids], dtype=torch.int32)},
		os.path.join(out, "conditioning.safetensors"), {"prompt": options.prompt})
	del clip
	comfy.model_management.unload_all_models()
	model = comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("diffusion_models", options.dit))
	for entry in [entry for entry in options.lora if entry]:
		name, strength = (entry.rsplit(":", 1)[0], float(entry.rsplit(":", 1)[1])) if ":" in entry else (entry, options.lora_strength)
		lora = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", name), safe_load=True)
		model, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)
	loaded = time.perf_counter()
	diffusion = model.model.diffusion_model
	latent = torch.zeros([1, 128, options.height // 16, options.width // 16])
	noise = comfy.sample.prepare_noise(latent, options.seed)
	sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), options.scheduler, options.steps)
	calls, block_outputs, hooks = [], {}, []

	def model_hook(module, args, kwargs, output):
		calls.append({"x": kwargs.get("x", args[0] if args else None).clone(), "timestep": kwargs.get("timestep", args[1] if len(args) > 1 else None).clone(), "output": output.clone()})

	def keep(name):
		def hook(module, args, output):
			if len(calls) == 0:
				block_outputs[name] = (output[0] if isinstance(output, tuple) else output).clone()
		return hook

	step_times = []

	def callback(step, x0, x, total):
		step_times.append(time.perf_counter())
		denoised.append(x0.clone())

	for run in range(options.runs):
		last = run == options.runs - 1
		if last:
			hooks.append(diffusion.register_forward_hook(model_hook, with_kwargs=True))
			for name in ("time_embedding", "shared_block_modulation", "video_head_modulation", "text_embedder", "video_embedder", "final_linear_video"):
				hooks.append(getattr(diffusion, name).register_forward_hook(keep(name)))
			for index, layer in enumerate(diffusion.layers):
				hooks.append(layer.register_forward_hook(keep(f"layers.{index}")))
		denoised, step_times = [], []
		comfy.model_management.soft_empty_cache()
		torch.cuda.synchronize()
		torch.cuda.reset_peak_memory_stats()
		sampling = time.perf_counter()
		step_times.append(sampling)
		samples = comfy.sample.sample(model, noise, options.steps, 1.0, "euler", options.scheduler, conditioning, conditioning, latent, seed=options.seed, callback=callback, disable_pbar=True)
		torch.cuda.synchronize()
		sampled = time.perf_counter()
		peaks["sampling"] = torch.cuda.max_memory_allocated() / 2 ** 30
		print(f"run {run}: sampling {sampled - sampling:.3f} s, steps", ", ".join(f"{(b - a) * 1000:.0f}" for a, b in zip(step_times, step_times[1:])), "ms")
	for hook in hooks:
		hook.remove()
	step0 = {"x": calls[0]["x"], "timestep": calls[0]["timestep"], "context": context, "noise": noise, "output": calls[0]["output"]}
	step0.update(block_outputs)
	saved(step0, os.path.join(out, "step0.safetensors"))
	steps = {"noise": noise, "sigmas": sigmas, "latent": samples}
	for index, call in enumerate(calls):
		steps[f"x.{index}"], steps[f"timestep.{index}"], steps[f"output.{index}"] = call["x"], call["timestep"], call["output"]
	for index, value in enumerate(denoised):
		steps[f"denoised.{index}"] = value
	vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", options.vae)))
	torch.cuda.reset_peak_memory_stats()
	decoding = time.perf_counter()
	image = vae.decode(samples)
	decoded = time.perf_counter()
	peaks["decode"] = torch.cuda.max_memory_allocated() / 2 ** 30
	steps["image"] = image
	saved(steps, os.path.join(out, "steps.safetensors"))
	from PIL import Image
	picture = Image.fromarray((image[0].clamp(0, 1) * 255).round().byte().cpu().numpy())
	picture.save(os.path.join(out, "image.png"))
	picture.save(os.path.join(out, "image.jpg"), quality=80)
	times = {"gpu": torch.cuda.get_device_name(), "width": options.width, "height": options.height, "steps": options.steps,
		"scheduler": options.scheduler, "seed": options.seed, "lora": options.lora, "encode_s": encoded - started, "load_s": loaded - encoded,
		"step_ms": [(b - a) * 1000 for a, b in zip(step_times, step_times[1:])], "sampling_s": sampled - sampling, "decode_s": decoded - decoding,
		"peak_vram_gb": peaks}
	with open(os.path.join(out, "times.json"), "w") as file:
		json.dump(times, file, indent=1)
	print(json.dumps(times))


def h3():
	"""t2va (i2v with --image) through ComfyUI's own H3 nodes like the UI workflow (MiniMaxH3ImageToVideo, euler, the simple
	scheduler, BasicGuider): conditioning.safetensors (Qwen3-VL states, token tags, keyframe latents, their frame indices and
	the noise augmented cond rows the model packs after the text), step0.safetensors (the
	refined text, block 0's input, the curve rows and the outputs of a few blocks in the first call), steps.safetensors (per call
	the streams and velocities of _forward, each stream on its own schedule, and the sampler's packed ones as sampler.*), the
	decoded frames (first, middle and last as PNG) and audio (audio.wav), times.json."""
	import numpy
	from PIL import Image
	from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
	from comfy_extras.nodes_custom_sampler import BasicScheduler, KSamplerSelect, BasicGuider, RandomNoise, SamplerCustomAdvanced
	out = options.out or r"C:\StrictDumps\h3"
	os.makedirs(out, exist_ok=True)
	started = time.perf_counter()
	clip = comfy.sd.load_clip([folder_paths.get_full_path_or_raise("text_encoders", options.te)], clip_type=comfy.sd.CLIPType.MINIMAX)
	vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", options.vae)))
	image = torch.from_numpy(numpy.array(Image.open(options.image).convert("RGB"))).float()[None] / 255 if options.image else None
	conditioning, latent = MiniMaxH3ImageToVideo.execute(clip, vae, options.prompt, options.width, options.height, options.length, first_frame=image).result
	encoded = time.perf_counter()
	values = conditioning[0][1]
	tags = values.get("minimax_token_tags")
	tags = torch.ones(conditioning[0][0].shape[1]) if tags is None else tags
	condition = {"context": conditioning[0][0][0], "tags": tags.to(torch.float32).flatten()}
	keyframes = values.get("minimax_keyframes") or []
	for index, keyframe in enumerate(keyframes):
		condition[f"keyframe.{index}"] = keyframe["latent"][0]
	if keyframes:
		condition["keyframes"] = torch.tensor([float(keyframe["resolved_frame_index"]) for keyframe in keyframes])
	saved(condition, os.path.join(out, "conditioning.safetensors"), {"prompt": options.prompt})
	del clip
	comfy.model_management.unload_all_models()
	model = comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("diffusion_models", options.dit))
	loaded = time.perf_counter()
	diffusion = model.model.diffusion_model
	sigmas = BasicScheduler.execute(model, options.scheduler, options.steps, 1.0).result[0]
	guider = BasicGuider.execute(model, conditioning).result[0]
	sampler = KSamplerSelect.execute("euler").result[0]
	noise = RandomNoise.execute(options.seed).result[0]
	calls, outer, first, times, recording = [], [], {}, [], []
	original = diffusion._forward

	def recorded(x, timestep, context, transformer_options={}, **kwargs):
		torch.cuda.synchronize()
		begun = time.perf_counter()
		output = original(x, timestep, context, transformer_options, **kwargs)
		torch.cuda.synchronize()
		times.append(time.perf_counter() - begun)
		if recording:
			if not calls:
				first["text"] = context[0].clone()
			calls.append({"video": x[0][0].clone(), "audio": x[1][0].clone(), "timestep": timestep.clone(),
				"video_output": output[0][0].clone(), "audio_output": output[1][0].clone()})
		return output
	diffusion._forward = recorded
	conditioned = diffusion._cond_video_rows

	def cond_rows(payload, device):
		rows = conditioned(payload, device)
		if rows is not None and "rows" not in condition:
			condition["rows"] = rows.float().clone()
		return rows
	diffusion._cond_video_rows = cond_rows

	def packed(module, args, kwargs, output):
		if recording:
			outer.append({"video": args[0][0][0].clone(), "audio": args[0][1][0].clone(), "video_output": output[0][0].clone(), "audio_output": output[1][0].clone()})

	def keep(name, entering=False):
		def hook(module, args, output=None):
			if recording and not calls:
				value = args[0] if entering else (output[0] if isinstance(output, tuple) else output)
				first[name] = value.clone() if entering else value.to(torch.bfloat16).clone()
		return hook
	hooks = [diffusion.register_forward_hook(packed, with_kwargs=True), diffusion.blocks[0].register_forward_pre_hook(keep("blocks.0.input", True)),
		diffusion.blocks[0].adaln_proj.register_forward_pre_hook(keep("t_emb", True))]
	for index in sorted({0, 1, 2, 3, len(diffusion.blocks) // 2, len(diffusion.blocks) - 1}):
		hooks.append(diffusion.blocks[index].register_forward_hook(keep(f"blocks.{index}")))
	for run in range(options.runs):
		if run == options.runs - 1:
			recording.append(True)
		times.clear()
		torch.cuda.synchronize()
		torch.cuda.reset_peak_memory_stats()
		sampling = time.perf_counter()
		samples = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent).result[0]["samples"]
		torch.cuda.synchronize()
		sampled = time.perf_counter()
		print(f"run {run}: sampling {sampled - sampling:.3f} s, model calls", ", ".join(f"{(t * 1000):.0f}" for t in times), "ms")
	peak = torch.cuda.max_memory_allocated() / 2 ** 30
	for hook in hooks:
		hook.remove()
	diffusion._forward = original
	diffusion._cond_video_rows = conditioned
	if "rows" in condition:
		saved(condition, os.path.join(out, "conditioning.safetensors"), {"prompt": options.prompt})
	saved(first, os.path.join(out, "step0.safetensors"))
	video, audio = samples.unbind()
	noises = noise.generate_noise(latent).unbind()
	steps = {"sigmas": sigmas, "noise.video": noises[0][0], "noise.audio": noises[1][0], f"video.{len(calls)}": video[0], f"audio.{len(calls)}": audio[0]}
	for prefix, records in (("", calls), ("sampler.", outer)):
		for index, record in enumerate(records):
			steps.update({f"{prefix}{name}.{index}": value for name, value in record.items()})
	del model, diffusion, guider
	comfy.model_management.unload_all_models()
	frames, decode_s, audio_s = decoded_av(vae, video, audio, out, "")
	steps["frames"] = frames
	saved(steps, os.path.join(out, "steps.safetensors"))
	times = {"gpu": torch.cuda.get_device_name(), "width": options.width, "height": options.height, "length": options.length, "steps": options.steps,
		"seed": options.seed, "dit": options.dit, "image": options.image, "encode_s": encoded - started, "load_s": loaded - encoded,
		"call_ms": [t * 1000 for t in times], "sampling_s": sampled - sampling, "decode_s": decode_s, "audio_s": audio_s,
		"peak_vram_gb": peak, "frames": list(frames.shape), "tokens": calls[0]["video"].numel() // 96 + calls[0]["audio"].numel() // 32}
	with open(os.path.join(out, "times.json"), "w") as file:
		json.dump(times, file, indent=1)
	print(json.dumps(times))


def decoded_av(vae, video, audio, out, prefix):
	"""Decodes an H3 video latent [1, 24, T, H, W] and audio latent [1, 32, 2, T] with ComfyUI's VAEs: the frames as uint8
	(the first, middle and last also as prefix + frame-*.png), prefix + audio.wav, the seconds both decodes took."""
	import wave
	from PIL import Image
	from comfy_extras.nodes_audio import vae_decode_audio
	decoding = time.perf_counter()
	frames = vae.decode(video)
	decoded = time.perf_counter()
	frames = (frames.reshape(-1, *frames.shape[-3:]).clamp(0, 1) * 255).round().byte().cpu()
	for name, frame in (("first", 0), ("middle", len(frames) // 2), ("last", len(frames) - 1)):
		Image.fromarray(frames[frame].numpy()).save(os.path.join(out, f"{prefix}frame-{name}.png"))
	audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", options.audio_vae)))
	heard = time.perf_counter()
	sound = vae_decode_audio(audio_vae, {"samples": audio})
	waveform = (sound["waveform"][0].clamp(-1, 1) * 32767).round().short().T.contiguous().cpu().numpy()
	with wave.open(os.path.join(out, prefix + "audio.wav"), "wb") as file:
		file.setnchannels(waveform.shape[1])
		file.setsampwidth(2)
		file.setframerate(sound["sample_rate"])
		file.writeframes(waveform.tobytes())
	audio_s = time.perf_counter() - heard
	encoded(frames, os.path.join(out, prefix + "audio.wav"), os.path.join(out, prefix + "video.mp4"))
	return frames, decoded - decoding, audio_s


def encoded(frames, audio, path):
	"""H.265 MP4 of uint8 frames [T, H, W, 3] at 24 fps with the audio WAV, through the installed ffmpeg (libx265)."""
	import shutil
	import subprocess
	command = [shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{frames.shape[2]}x{frames.shape[1]}",
		"-r", "24", "-i", "-", "-i", audio, "-c:v", "libx265", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-tag:v", "hvc1",
		"-x265-params", "log-level=error", "-c:a", "aac", "-b:a", "192k", "-shortest", path]
	subprocess.run(command, input=frames.contiguous().numpy().tobytes(), check=True)
	print("wrote", path, f"{os.path.getsize(path) / 1e6:.1f} MB")


def h3decode():
	"""Decodes the latent H3Video.strict wrote into strict-latent.txt in --out (the video then the audio numbers, one per line,
	shaped like the last step of steps.safetensors) into strict-frame-*.png and strict-audio.wav, PSNR against ComfyUI's frames."""
	import numpy
	from safetensors.torch import load_file
	out = options.out or r"C:\StrictDumps\h3"
	steps = load_file(os.path.join(out, "steps.safetensors"))
	count = len(steps["sigmas"]) - 1
	shapes = steps[f"video.{count}"].shape, steps[f"audio.{count}"].shape
	numbers = torch.from_numpy(numpy.loadtxt(os.path.join(out, "strict-latent.txt"), dtype=numpy.float32))
	video = numbers[:shapes[0].numel()].reshape(shapes[0])[None]
	audio = numbers[shapes[0].numel():].reshape(shapes[1])[None]
	vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", options.vae)))
	frames, decode_s, audio_s = decoded_av(vae, video, audio, out, "strict-")
	if not os.path.exists(os.path.join(out, "video.mp4")):
		encoded(steps["frames"], os.path.join(out, "audio.wav"), os.path.join(out, "video.mp4"))
	error = (frames.float() - steps["frames"].float()).pow(2).mean().item()
	print(f"decoded in {decode_s:.1f} s, audio in {audio_s:.1f} s, PSNR {10 * numpy.log10(255 ** 2 / error):.2f} dB against ComfyUI's frames")


class TinyWeights(dict):
	"""Random weights named like a real file: BF16 unless given, int8 convrot linears quantized in groups of 256."""
	quant = list(json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode())

	def brain(self, name, shape, scale, dtype=None):
		self[name] = (torch.randn(shape) * scale).to(dtype or torch.bfloat16)

	def linear(self, name, outputs, inputs, bias=True, dtype=None):
		self.brain(name + ".weight", (outputs, inputs), inputs ** -0.5, dtype)
		if bias:
			self.brain(name + ".bias", (outputs,), 0.05, dtype)

	def norm(self, name, size):
		self[name + ".weight"] = (1 + 0.2 * torch.randn(size)).to(torch.bfloat16)

	def convrot(self, name, outputs, inputs):
		from comfy_kitchen.backends.eager.quantization import quantize_int8_convrot_weight
		self[name + ".weight"], self[name + ".weight_scale"] = quantize_int8_convrot_weight(torch.randn(outputs, inputs) * inputs ** -0.5, 256)
		self[name + ".comfy_quant"] = torch.tensor(self.quant, dtype=torch.uint8)


def tinymoss():
	"""2 layers at width 256 (the convrot group), int8 convrot layers as in the real file, the rest BF16, plus the
	reference.* tensors of one forward through ComfyUI's model: 4x4 latent tokens, 3 text tokens."""
	out = options.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinymoss.safetensors")
	dim, ffn, layers, text_dim, channels = 256, 256, 2, 64, 16
	torch.manual_seed(0)
	weights = TinyWeights()
	linear, norm, convrot = weights.linear, weights.norm, weights.convrot
	linear("video_embedder", dim, channels)
	linear("text_embedder.proj", dim, text_dim)
	for block in range(2):
		prefix = f"text_embedder.blocks.{block}."
		for name in ("norm1", "norm2"):
			norm(prefix + name, dim)
		for name in ("q_norm", "k_norm"):
			norm(prefix + name, 128)
		linear(prefix + "qkv", 3 * dim, dim)
		linear(prefix + "proj", dim, dim)
		linear(prefix + "ffn_gate", ffn, dim)
		linear(prefix + "ffn_up", ffn, dim)
		linear(prefix + "ffn_down", dim, ffn)
	linear("time_embedding.0", dim, 256)
	linear("time_embedding.2", dim, dim)
	linear("shared_block_modulation", 6 * dim, dim)
	for layer in range(layers):
		prefix = f"layers.{layer}."
		for name in ("norm1", "norm2"):
			norm(prefix + name, dim)
		for name in ("q_norm", "k_norm"):
			norm(prefix + name, 128)
		for name in ("q", "k", "v", "o"):
			convrot(prefix + "proj." + name, dim, dim)
		convrot(prefix + "proj.ffn_up_gate", 2 * ffn, dim)
		convrot(prefix + "proj.ffn_down", dim, ffn)
	norm("final_norm_video", dim)
	linear("final_linear_video", channels, dim, bias=False)
	linear("video_head_modulation", 2 * dim, dim)
	references = tiny_references(dict(weights))
	saved({**weights, **{name: value.float() for name, value in references.items()}}, out)


def tiny_references(weights, inputs=None, lora=None):
	"""One forward of the tiny Moss weights through ComfyUI's model (optionally with a LoRA applied like ComfyUI does),
	reference.* outputs per module; inputs (x, timestep, context) are random 4x4 latent and 3 text tokens if not given."""
	import comfy.model_detection
	text_dim, channels = weights["text_embedder.proj.weight"].shape[1], weights["video_embedder.weight"].shape[1]
	detect = comfy.model_detection.detect_unet_config

	def detected(state_dict, key_prefix, metadata=None):
		config = detect(state_dict, key_prefix, metadata=metadata)
		if config and config.get("image_model") == "moss_image":
			config.update(text_dim=text_dim, latent_channels=channels)
		return config
	comfy.model_detection.detect_unet_config = detected
	model = comfy.sd.load_diffusion_model_state_dict(dict(weights))
	if lora is not None:
		model, _ = comfy.sd.load_lora_for_models(model, None, lora, 1.0, 0)
	comfy.model_management.load_models_gpu([model])
	diffusion = model.model.diffusion_model
	device, dtype = comfy.model_management.get_torch_device(), model.model.get_dtype()
	x, timestep, context = inputs or (torch.randn(1, channels, 4, 4), torch.tensor([0.75]), torch.randn(1, 3, text_dim))
	references, hooks = {}, []

	def keep(name):
		def hook(module, args, output):
			references["reference." + name] = (output[0] if isinstance(output, tuple) else output).clone()
		return hook
	for name in ("time_embedding", "shared_block_modulation", "video_head_modulation", "text_embedder", "video_embedder"):
		hooks.append(getattr(diffusion, name).register_forward_hook(keep(name)))
	for index, layer in enumerate(diffusion.layers):
		hooks.append(layer.register_forward_hook(keep(f"layers.{index}")))
	with torch.no_grad():
		output = diffusion(x.to(device, dtype), timestep.to(device, dtype), context.to(device, dtype))
	references.update({"reference.x": x, "reference.timestep": timestep, "reference.context": context, "reference.output": output})
	for hook in hooks:
		hook.remove()
	return references


def tinylora():
	"""A rank-8 LoRA (alpha 16, scale 2) for tinymoss.safetensors with the key names of moss\\dmd_turbo_lora.safetensors, plus
	the reference.* outputs of the tiny model with it applied by ComfyUI and reference.q, the exact float32 layers.0 q delta
	scale * up(down(x)) for x = video_embedder + text_embedder tokens."""
	from safetensors.torch import load_file
	here = os.path.dirname(os.path.abspath(__file__))
	out = options.out or os.path.join(here, "tinylora.safetensors")
	tiny = load_file(os.path.join(here, "tinymoss.safetensors"))
	weights = {name: value for name, value in tiny.items() if not name.startswith("reference.")}
	rank, alpha = 8, 16.0
	torch.manual_seed(1)
	lora = {}
	for layer in range(2):
		for name in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down"):
			key = f"diffusion_model.layers.{layer}.proj.{name}"
			outputs, inputs = weights[f"layers.{layer}.proj.{name}.weight"].shape
			lora[key + ".lora_down.weight"] = torch.randn(rank, inputs) * inputs ** -0.5
			lora[key + ".lora_up.weight"] = torch.randn(outputs, rank) * inputs ** -0.5
			lora[key + ".alpha"] = torch.tensor(alpha)
	inputs = (tiny["reference.x"], tiny["reference.timestep"], tiny["reference.context"])
	references = tiny_references(weights, inputs, dict(lora))
	hidden = torch.cat([tiny["reference.video_embedder"], tiny["reference.text_embedder"]], 1).reshape(-1, 256)
	down, up = lora["diffusion_model.layers.0.proj.q.lora_down.weight"], lora["diffusion_model.layers.0.proj.q.lora_up.weight"]
	references["reference.q"] = alpha / rank * (hidden @ down.T) @ up.T
	saved({**lora, **{name: value.float() for name, value in references.items()}}, out, {"rank": str(rank), "alpha": str(alpha)})


def tinyvae():
	"""A tiny Flux2 VAE decoder with ComfyUI's classes (ch 8, ch_mult [1, 2], 1 res block + 1, z_channels 4, GroupNorm
	with 4 groups instead of 32), random weights named like flux2-vae.safetensors, a 16x3x2 latent and reference.*
	intermediates of AutoencoderKL.decode in float32 on the CPU, channel last (height, width, channels)."""
	import comfy.ldm.modules.diffusionmodules.model as model
	from comfy.ldm.models.autoencoder import AutoencoderKL
	out = options.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinyvae.safetensors")
	model.Normalize.__defaults__ = (4,)
	torch.manual_seed(0)
	config = {"double_z": True, "z_channels": 4, "resolution": 256, "in_channels": 3, "out_ch": 3, "ch": 8, "ch_mult": [1, 2],
		"num_res_blocks": 1, "attn_resolutions": [], "dropout": 0.0, "batch_norm_latent": True}
	vae = AutoencoderKL(ddconfig=config, embed_dim=4).eval()
	weights = {}
	for name, value in vae.state_dict().items():
		if name.startswith("encoder") or name.startswith("quant_conv") or "num_batches" in name:
			continue
		if name.endswith("running_var"):
			value = 0.5 + torch.rand(value.shape)
		elif name.endswith("running_mean"):
			value = 0.5 * torch.randn(value.shape)
		elif "norm" in name and name.endswith("weight"):
			value = 1 + 0.2 * torch.randn(value.shape)
		elif name.endswith("weight"):
			value = torch.randn(value.shape) * (value[0].numel() ** -0.5)
		else:
			value = 0.1 * torch.randn(value.shape)
		weights[name] = value.float()
	vae.load_state_dict(weights, strict=False)
	weights = {("decoder." + name if name.startswith("post_quant") else name): value for name, value in weights.items()}
	latent = torch.randn(1, 16, 3, 2)
	references, hooks = {}, []

	def keep(name):
		def hook(module, args, output):
			references["reference." + name] = output[0].permute(1, 2, 0).clone()
		return hook
	decoder = vae.decoder
	for name in ("conv_in", "mid.block_1.norm1", "mid.block_1", "mid.attn_1", "mid.block_2", "up.1.block.0", "up.1.upsample",
		"up.0.block.0", "norm_out"):
		hooks.append(decoder.get_submodule(name).register_forward_hook(keep(name)))
	hooks.append(vae.post_quant_conv.register_forward_hook(keep("post_quant_conv")))
	hooks.append(vae.post_quant_conv.register_forward_pre_hook(lambda module, args: references.update({"reference.z": args[0][0].permute(1, 2, 0).clone()})))
	with torch.no_grad():
		decoded = vae.decode(latent)
	references["reference.output"] = decoded[0].permute(1, 2, 0)
	references["reference.image"] = ((decoded + 1) / 2).clamp(0, 1)[0].permute(1, 2, 0)
	saved({**weights, "latent": latent, **references}, out)


def tinyh3():
	"""2 blocks and 2 refiner blocks at hidden 256 with 4 heads of 128 (inner 512, like H3's 7168 against 5376), int8
	convrot block linears and curve-form adaln (a 9 x 8 table) as in the real file, plus the reference.* tensors of one
	t2va forward through ComfyUI's model at sigma 0.75: a 2x4x4 video latent (8 tokens), 3 audio frames (6 rows), 3 text
	tokens; reference.t_emb holds the curve rows of the timesteps the blocks modulate with, reference.blocks.0.* the inputs
	and outputs of block 0's attention (qkv_proj output, merged heads before out_proj) and MLP; reference.i2v.* the same
	forward with a keyframe latent at frame 0 (4 cond rows after the text, noise augmented as reference.i2v.rows) and
	text tags (1, 0, 1), the middle token carrying the video modality like a vision pad."""
	import comfy.ldm.minimax.model as minimax
	out = options.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinyh3.safetensors")
	hidden, inner, ffn, text_dim, grid, curve = 256, 512, 256, 64, 9, 8
	torch.manual_seed(0)
	weights = TinyWeights()
	weights.linear("video_patch_proj", hidden, 96, dtype=torch.float32)
	weights.linear("audio_patch_proj", hidden, 32, dtype=torch.float32)
	weights.linear("condition_proj", hidden, text_dim)
	weights.brain("adaln_t_table", (grid, curve), 0.3, torch.float32)
	weights["rope.inv_freq"] = 10000.0 ** (-torch.arange(16, dtype=torch.float32) / 16)
	for prefix in ("token_refiner.blocks.0.", "token_refiner.blocks.1.", "blocks.0.", "blocks.1."):
		weights.norm(prefix + "norm1", hidden)
		weights.norm(prefix + "norm2", hidden)
		weights.norm(prefix + "attn.q_norm", 128)
		weights.norm(prefix + "attn.k_norm", 128)
		dense = prefix.startswith("token_refiner.")
		project = (lambda name, outputs, inputs: weights.linear(name, outputs, inputs, False)) if dense else weights.convrot
		project(prefix + "attn.qkv_proj", 3 * inner, hidden)
		project(prefix + "attn.out_proj", hidden, inner)
		project(prefix + "mlp.fc1", 2 * ffn, hidden)
		project(prefix + "mlp.fc2", hidden, ffn)
		if not dense:
			weights.linear(prefix + "adaln_proj.linear", 6 * hidden * 3, curve, dtype=torch.float16)
	weights.norm("token_refiner.final_norm", hidden)
	weights.norm("final_layer.norm", hidden)
	weights.linear("final_layer.adaln_proj.linear", 2 * hidden, curve, dtype=torch.float16)
	weights.linear("final_layer.video_out", 96, hidden, dtype=torch.float32)
	weights.linear("final_layer.audio_out", 32, hidden, dtype=torch.float32)
	model = comfy.sd.load_diffusion_model_state_dict(dict(weights))
	comfy.model_management.load_models_gpu([model])
	diffusion = model.model.diffusion_model
	device, dtype = comfy.model_management.get_torch_device(), model.model.get_dtype()
	video, audio, context, sigma = torch.randn(1, 24, 2, 4, 4), torch.randn(1, 32, 2, 3), torch.randn(1, 3, text_dim), 0.75
	payload = {"layout": minimax.PackedLayout(3, 2, 4, 4, 3), "seed": 0, "audio_scale": 1.0}
	references, hooks = {}, []

	def keep(name):
		def hook(module, args, output):
			references["reference." + name] = (output[0] if isinstance(output, tuple) else output).clone()
		return hook

	def entered(name):
		def hook(module, args):
			references["reference." + name] = args[0].clone()
		return hook
	for name in ("token_refiner", "video_patch_proj", "audio_patch_proj"):
		hooks.append(getattr(diffusion, name).register_forward_hook(keep(name)))
	attention, feed = diffusion.blocks[0].attn, diffusion.blocks[0].mlp
	for name, module in (("blocks.0.attn.qkv_proj", attention.qkv_proj), ("blocks.0.attn", attention), ("blocks.0.mlp", feed)):
		hooks.append(module.register_forward_hook(keep(name)))
	entries = (("blocks.0.input", diffusion.blocks[0]), ("blocks.0.attn.input", attention), ("blocks.0.attn.merged", attention.out_proj),
		("blocks.0.mlp.input", feed))
	for name, module in entries:
		hooks.append(module.register_forward_pre_hook(entered(name)))
	hooks.append(diffusion.blocks[0].adaln_proj.register_forward_pre_hook(entered("t_emb")))
	for index, block in enumerate(diffusion.blocks):
		hooks.append(block.register_forward_hook(keep(f"blocks.{index}")))
	streams, timestep = [video.to(device, dtype), audio.to(device, dtype)], torch.tensor([sigma * 1000], device=device)
	with torch.no_grad():
		output = diffusion(streams, timestep, context.to(device, dtype), {}, minimax_payload=payload)
	for hook in hooks:
		hook.remove()
	references.update({"reference.video": video[0], "reference.audio": audio[0], "reference.context": context, "reference.sigma": torch.tensor([sigma]),
		"reference.video_output": output[0][0], "reference.audio_output": output[1][0]})
	keyframe, tags = torch.randn(1, 24, 1, 4, 4), torch.tensor([1, 0, 1])
	keyframes = [{"resolved_frame_index": 0, "latent": keyframe}]
	payload = {"layout": minimax.PackedLayout(3, 2, 4, 4, 3, keyframes=keyframes), "keyframes": keyframes, "cond_video_latents": [keyframe],
		"seed": 0, "audio_scale": 1.0, "text_token_tags": tags}
	conditioned = diffusion._cond_video_rows

	def cond_rows(payload, device):
		rows = conditioned(payload, device)
		references["reference.i2v.rows"] = rows.clone()
		return rows
	diffusion._cond_video_rows = cond_rows
	hooks = [block.register_forward_hook(keep(f"i2v.blocks.{index}")) for index, block in enumerate(diffusion.blocks)]
	hooks += [diffusion.blocks[0].register_forward_pre_hook(entered("i2v.blocks.0.input")),
		diffusion.blocks[0].adaln_proj.register_forward_pre_hook(entered("i2v.t_emb"))]
	with torch.no_grad():
		output = diffusion(streams, timestep, context.to(device, dtype), {}, minimax_payload=payload)
	for hook in hooks:
		hook.remove()
	diffusion._cond_video_rows = conditioned
	references.update({"reference.i2v.keyframe": keyframe[0], "reference.i2v.tags": tags, "reference.i2v.video_output": output[0][0],
		"reference.i2v.audio_output": output[1][0]})
	saved({**weights, **{name: value.float() for name, value in references.items()}}, out)


def tinyh3vae():
	"""H3's video VAE decoder with ComfyUI's own decode (MiniMaxH3VideoVAE: latent normalization, post_quant_conv, temporal
	windows of 5 latent frames plus 2 overlapping, spatial tiles, blending, pixel finalization) around a tiny ViT (2 blocks, 4
	heads of 64, float weights named like minimax_h3_video_vae_int8_convrot.safetensors) with tiles of 32 pixels overlapping
	by 16, a 24x12x3x4 latent decoded on the CPU in float32: reference.latent, reference.pixels [3, frames, 48, 64] and
	reference.tile, the decoder output (_decode_pixels) of the first 2x2 tile of the first window."""
	import comfy.ldm.minimax.vae as minimax
	out = options.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinyh3vae.safetensors")
	dim, inner = 256, 1024
	torch.manual_seed(0)
	weights = TinyWeights()
	weights.brain("post_quant_conv.weight", (24, 24, 1, 1, 1), 24 ** -0.5, torch.float32)
	weights.brain("post_quant_conv.bias", (24,), 0.05, torch.float32)
	weights.linear("decoder.x_embedder", dim, 24, dtype=torch.float32)
	weights.brain("decoder.register_tokens", (1, 4, dim), 1.0, torch.float32)
	weights["decoder.mask_token"] = torch.zeros(1, 1, dim)
	for block in range(2):
		prefix = f"decoder.transformer_blocks.{block}."
		weights.linear(prefix + "attn.to_qkv", 3 * dim, dim, dtype=torch.float32)
		weights.linear(prefix + "attn.to_out", dim, dim, dtype=torch.float32)
		weights.linear(prefix + "ff.w1", 2 * inner, dim, dtype=torch.float32)
		weights.linear(prefix + "ff.w2", dim, inner, dtype=torch.float32)
		for name in ("norm1.weight", "norm2.weight"):
			weights[prefix + name] = 1 + 0.2 * torch.randn(dim)
		for name in ("scale1", "scale2"):
			weights.brain(prefix + name, (dim,), 0.5, torch.float32)
	weights["decoder.norm_out.weight"] = 1 + 0.2 * torch.randn(dim)
	weights.brain("decoder.norm_out.bias", (dim,), 0.05, torch.float32)
	weights.linear("decoder.proj_out", 3 * 4 * 16 * 16, dim, dtype=torch.float32)
	weights["latents_mean"], weights["latents_std"] = torch.tensor(minimax.LATENTS_MEAN), torch.tensor(minimax.LATENTS_STD)
	vae = minimax.MiniMaxH3VideoVAE(encoder=False, num_layers=2, tile_size=32, tile_overlap_min=16)
	vae.decoder = minimax.ViT3DDecoder(patch_size=16, patch_size_t=4, in_channels=24, out_channels=3, num_layers=2, heads=4, dim_head=64)
	vae.load_state_dict(dict(weights), strict=True)
	latent = torch.randn(1, 24, 12, 3, 4)
	with torch.no_grad():
		pixels = vae.decode(latent)
		denormalized = latent * vae.latents_std.view(1, -1, 1, 1, 1) + vae.latents_mean.view(1, -1, 1, 1, 1)
		tile = vae._decode_pixels(denormalized[:, :, 0:7, 0:2, 0:2])
	saved({**weights, "reference.latent": latent[0], "reference.pixels": pixels[0], "reference.tile": tile[0]}, out)


def tinyint8():
	"""An int8 matrix with an F32 scale per row and its comfy_quant bytes, for the generic tests in Strict/NeuralNetworks."""
	out = options.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Strict", "NeuralNetworks", "tinyint8.safetensors")
	quant = torch.tensor(list(json.dumps({"format": "int8_tensorwise"}).encode()), dtype=torch.uint8)
	weight = (torch.arange(4 * 64) % 255 - 127).to(torch.int8).reshape(4, 64)
	saved({"a.weight": weight, "a.weight_scale": torch.tensor([[0.5], [0.25], [2.0], [1.0]]), "a.comfy_quant": quant}, out)


if __name__ == "__main__":
	{"moss": moss, "h3": h3, "h3decode": h3decode, "tinymoss": tinymoss, "tinylora": tinylora, "tinyint8": tinyint8, "tinyvae": tinyvae, "tinyh3": tinyh3, "tinyh3vae": tinyh3vae}[options.model]()
