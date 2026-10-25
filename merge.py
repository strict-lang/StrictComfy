"""Merges LoRAs into an int8 convrot model once, so Strict (and ComfyUI) load the result like any model. Run with ComfyUI's python:
  python_embeded\\python.exe merge.py <model.safetensors> <out.safetensors> <lora.safetensors>[:strength] ...
Each convrot linear: W·Hᵀ (int8 · row scale) + strength · alpha / rank · (up · down)·Hᵀ, requantized to int8 with one scale per row
(stochastic rounding like ComfyUI: nearest rounding drops LoRA deltas below half a step). Moss checkpoints in other layouts
(bf16, fp16 or fp8 linears, fused qkv, ComfyUI's model prefix) are converted to the unfused int8 convrot one Strict loads. Matching floating-point weights receive their deltas without quantization. LoRAs trained on a fused proj.qkv
match no weight of this unfused model and are skipped like ComfyUI does (strengths tuned in ComfyUI expect that);
--split-qkv applies their q/k/v thirds instead.
"""
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from generate import lora_spec
from safetensors import safe_open
from safetensors.torch import save_file

split = "--split-qkv" in sys.argv
paths = [argument for argument in sys.argv[1:] if argument != "--split-qkv"]
model, out, loras = paths[0], paths[1], paths[2:]
if Path(out).resolve() in [Path(model).resolve(), *[Path(path).resolve() for path, strength in map(lora_spec, loras)]]:
	raise ValueError('Merged output must differ from every input file')
device = "cuda" if torch.cuda.is_available() else "cpu"
h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
hadamard = h4
for _ in range(3):
	hadamard = torch.kron(hadamard, h4)
hadamard = (hadamard / 16).to(device)
generator = torch.Generator(device=device).manual_seed(0)
QUANT = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256})


def rotated(rows, group):
	return (rows.reshape(rows.shape[0], -1, group) @ hadamard).reshape(rows.shape)


def delta(files, name):
	total = None
	for index, (file, strength) in enumerate(files):
		part = found(file, name)
		parent, _, own = name.rpartition(".")
		if part is None and own in ("q", "k", "v") and split:
			fused = found(file, parent + ".qkv")
			if fused is not None:
				third = fused.shape[0] // 3
				part = fused["qkv".index(own) * third:("qkv".index(own) + 1) * third]
		if part is not None:
			matched[index] += 1
			total = strength * part if total is None else total + strength * part
	return total


def found(file, module):
	"""The product of the first LoRA module named like ComfyUI matches it: diffusion_model.<name>, <name> or kohya's lora_unet_<name>."""
	bare = module.removeprefix("model.").removeprefix("diffusion_model.")
	prefixes = ("diffusion_model." + bare + ".", bare + ".", "lora_unet_" + bare.replace(".", "_") + ".")
	return next((part for part in (product(file, prefix) for prefix in prefixes) if part is not None), None)


def product(file, prefix):
	"""alpha / rank · up · down of one LoRA module in kohya (lora_down/lora_up) or PEFT (lora_A/lora_B) naming, alpha missing = rank."""
	for down_name, up_name in (("lora_down", "lora_up"), ("lora_A", "lora_B")):
		if prefix + down_name + ".weight" in file.keys():
			down = file.get_tensor(prefix + down_name + ".weight").float().to(device)
			up = file.get_tensor(prefix + up_name + ".weight").float().to(device)
			alpha = float(file.get_tensor(prefix + "alpha")) if prefix + "alpha" in file.keys() else down.shape[0]
			return alpha / down.shape[0] * (up @ down)
	return None


source = safe_open(model, "pt")
specs = [(path, strength) for path, strength in map(lora_spec, loras) if strength != 0]
files = [(safe_open(path, "pt"), strength) for path, strength in specs]
matched = [0] * len(files)
tensors, merged = {}, 0
moss = any(key.removeprefix("model.diffusion_model.").startswith("layers.") for key in source.keys())
for key in source.keys():
	tensors[key.removeprefix("model.diffusion_model.") if moss else key] = source.get_tensor(key)
for key in [key for key in tensors if ".proj.qkv." in key]:
	fused = tensors.pop(key)
	parts = (fused,) * 3 if key.endswith((".comfy_quant", ".scale_weight")) else fused.chunk(3)
	for own, part in zip("qkv", parts):
		tensors[key.replace(".proj.qkv.", ".proj." + own + ".")] = part.clone()
for key in list(tensors):
	name = key.removesuffix(".weight")
	if re.fullmatch(r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)", name) and name + ".comfy_quant" not in tensors:
		tensors[name + ".comfy_quant"] = torch.tensor(list(QUANT.encode()), dtype=torch.uint8)
for key in list(tensors):
	if not key.endswith(".comfy_quant"):
		continue
	name = key[:-len(".comfy_quant")]
	quant = json.loads(bytes(tensors[key].tolist()))
	assert quant.get("convrot") and quant.get("convrot_groupsize") == 256 and quant.get("format") == "int8_tensorwise", quant
	change = delta(files, name)
	weight = tensors[name + ".weight"]
	if change is None and weight.dtype == torch.int8:
		continue
	if weight.dtype == torch.int8:
		weight = weight.float().to(device) * tensors[name + ".weight_scale"].float().to(device)
	else:
		weight = rotated(weight.float().to(device) * tensors.pop(name + ".scale_weight", torch.ones(())).float().to(device), 256)
	if change is not None:
		if change.shape != weight.shape:
			raise ValueError(f"LoRA shape {tuple(change.shape)} does not match {name}: {tuple(weight.shape)}")
		weight = weight + rotated(change, quant["convrot_groupsize"])
		merged += 1
	scale = (weight.abs().amax(dim=1, keepdim=True) / 127).clamp(min=1e-30)
	noise = torch.rand(weight.shape, generator=generator, device=device) if change is not None else 0.5
	tensors[name + ".weight"] = (weight / scale + noise).floor().clamp(-128, 127).to(torch.int8).cpu()
	tensors[name + ".weight_scale"] = scale.float().cpu()
quantized = {key[:-len('.comfy_quant')] for key in tensors if key.endswith('.comfy_quant')}
for key in list(tensors):
	if not key.endswith('.weight') or key[:-len('.weight')] in quantized:
		continue
	change = delta(files, key[:-len('.weight')])
	if change is None:
		continue
	weight = tensors[key]
	if change.shape != weight.shape or not weight.is_floating_point():
		raise ValueError('Unsupported LoRA target: ' + key)
	tensors[key] = (weight.float().to(device) + change).to(dtype=weight.dtype, device='cpu')
	merged += 1
for key in tensors:
	if moss and tensors[key].is_floating_point() and not key.endswith(".weight_scale"):
		tensors[key] = tensors[key].to(torch.bfloat16)
for (path, strength), count in zip(specs, matched):
	if count == 0:
		raise ValueError('LoRA matched no supported model modules: ' + path)
	print('matched', count, 'modules:', path, 'strength', strength)
print("merged", merged, "linears")
inherited = json.loads((source.metadata() or {}).get('strict_lora_merge', '{}'))
marker = {'algorithm': 'convrot-int8-stochastic-v1', 'seed': 0, 'device': device,
	'base_model': inherited.get('base_model', str(Path(model).resolve())),
	'loras': inherited.get('loras', []) + [(str(Path(path).resolve()), strength) for path, strength in specs]}
save_file(tensors, out, metadata={"merged_loras": json.dumps(loras), 'strict_lora_merge': json.dumps(marker)})
print("wrote", out)
