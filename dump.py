"""Reference dumps from ComfyUI's own code for the Strict ports. Run with ComfyUI's python_embeded:
  python_embeded\\python.exe dump.py moss [--out D:\\StrictDumps\\moss] [--width 720 --height 1280 ...], prompt from prompt.txt
  python_embeded\\python.exe dump.py tinymoss   (tinymoss.safetensors here, for MossBlock.strict and the other Moss tests)
  python_embeded\\python.exe dump.py tinyint8   (..\\Strict\\NeuralNetworks\\tinyint8.safetensors, generic int8 rows)
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
arguments.add_argument("model", choices=["moss", "tinymoss", "tinyint8"])
arguments.add_argument("--out", default=None)
arguments.add_argument("--prompt", default=None, help="default: prompt.txt next to this script")
arguments.add_argument("--width", type=int, default=720)
arguments.add_argument("--height", type=int, default=1280)
arguments.add_argument("--steps", type=int, default=8)
arguments.add_argument("--scheduler", default="simple")
arguments.add_argument("--seed", type=int, default=42)
arguments.add_argument("--runs", type=int, default=2, help="sampling runs, the last one is timed and dumped")
arguments.add_argument("--dit", default="moss-image-sft-dit-int8-convrot.safetensors")
arguments.add_argument("--te", default="qwen3.5_9b_int8_convrot.safetensors")
arguments.add_argument("--lora", default=r"moss\dmd_turbo_lora.safetensors")
arguments.add_argument("--vae", default="flux2-vae.safetensors")
options = arguments.parse_args()
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
	if options.lora:
		lora = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", options.lora), safe_load=True)
		model, _ = comfy.sd.load_lora_for_models(model, None, lora, 1.0, 0)
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


def tinymoss():
	"""2 layers at width 256 (the convrot group), int8 convrot layers as in the real file, the rest BF16, plus the
	reference.* tensors of one forward through ComfyUI's model: 4x4 latent tokens, 3 text tokens."""
	import comfy.model_detection
	from comfy_kitchen.backends.eager.quantization import quantize_int8_convrot_weight
	out = options.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinymoss.safetensors")
	dim, ffn, layers, text_dim, channels = 256, 256, 2, 64, 16
	torch.manual_seed(0)
	weights = {}
	quant = torch.tensor(list(json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode()), dtype=torch.uint8)

	def brain(name, shape, scale):
		weights[name] = (torch.randn(shape) * scale).to(torch.bfloat16)

	def linear(name, outputs, inputs, bias=True):
		brain(name + ".weight", (outputs, inputs), inputs ** -0.5)
		if bias:
			brain(name + ".bias", (outputs,), 0.05)

	def norm(name, size):
		weights[name + ".weight"] = (1 + 0.2 * torch.randn(size)).to(torch.bfloat16)

	def convrot(name, outputs, inputs):
		weights[name + ".weight"], weights[name + ".weight_scale"] = quantize_int8_convrot_weight(torch.randn(outputs, inputs) * inputs ** -0.5, 256)
		weights[name + ".comfy_quant"] = quant.clone()
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
	detect = comfy.model_detection.detect_unet_config

	def detected(state_dict, key_prefix, metadata=None):
		config = detect(state_dict, key_prefix, metadata=metadata)
		if config and config.get("image_model") == "moss_image":
			config.update(text_dim=text_dim, latent_channels=channels)
		return config
	comfy.model_detection.detect_unet_config = detected
	model = comfy.sd.load_diffusion_model_state_dict(dict(weights))
	comfy.model_management.load_models_gpu([model])
	diffusion = model.model.diffusion_model
	device, dtype = comfy.model_management.get_torch_device(), model.model.get_dtype()
	x = torch.randn(1, channels, 4, 4)
	timestep = torch.tensor([0.75])
	context = torch.randn(1, 3, text_dim)
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
	saved({**weights, **{name: value.float() for name, value in references.items()}}, out)


def tinyint8():
	"""An int8 matrix with an F32 scale per row and its comfy_quant bytes, for the generic tests in Strict/NeuralNetworks."""
	out = options.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Strict", "NeuralNetworks", "tinyint8.safetensors")
	quant = torch.tensor(list(json.dumps({"format": "int8_tensorwise"}).encode()), dtype=torch.uint8)
	weight = (torch.arange(4 * 64) % 255 - 127).to(torch.int8).reshape(4, 64)
	saved({"a.weight": weight, "a.weight_scale": torch.tensor([[0.5], [0.25], [2.0], [1.0]]), "a.comfy_quant": quant}, out)


if __name__ == "__main__":
	{"moss": moss, "tinymoss": tinymoss, "tinyint8": tinyint8}[options.model]()
