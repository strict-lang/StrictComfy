"""Merges LoRAs into an int8 convrot model once, so Strict (and ComfyUI) load the result like any model. Run with ComfyUI's python:
  python_embeded\\python.exe merge.py <model.safetensors> <out.safetensors> <lora.safetensors>[:strength] ...
Each convrot linear: W·Hᵀ (int8 · row scale) + strength · alpha / rank · (up · down)·Hᵀ, requantized to int8 with one scale per row
(stochastic rounding like ComfyUI: nearest rounding drops LoRA deltas below half a step). Matching floating-point weights receive their deltas without quantization. LoRAs trained on a fused proj.qkv
match no weight of this unfused model and are skipped like ComfyUI does (strengths tuned in ComfyUI expect that);
--split-qkv applies their q/k/v thirds instead.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from moss import lora_spec
from safetensors import safe_open
from safetensors.torch import save_file

split = "--split-qkv" in sys.argv
paths = [argument for argument in sys.argv[1:] if argument != "--split-qkv"]
model, out, loras = paths[0], paths[1], paths[2:]
device = "cuda" if torch.cuda.is_available() else "cpu"
h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
hadamard = h4
for _ in range(3):
	hadamard = torch.kron(hadamard, h4)
hadamard = (hadamard / 16).to(device)
generator = torch.Generator(device=device).manual_seed(0)


def rotated(rows, group):
	return (rows.reshape(rows.shape[0], -1, group) @ hadamard).reshape(rows.shape)


def delta(files, name):
	total = None
	for index, (file, strength) in enumerate(files):
		part = product(file, "diffusion_model." + name + ".")
		parent, _, own = name.rpartition(".")
		if part is None and own in ("q", "k", "v") and split:
			fused = product(file, "diffusion_model." + parent + ".qkv.")
			if fused is not None:
				third = fused.shape[0] // 3
				part = fused["qkv".index(own) * third:("qkv".index(own) + 1) * third]
		if part is not None:
			matched[index] += 1
			total = strength * part if total is None else total + strength * part
	return total


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
for key in source.keys():
	tensors[key] = source.get_tensor(key)
for key in list(tensors):
	if not key.endswith(".comfy_quant"):
		continue
	name = key[:-len(".comfy_quant")]
	quant = json.loads(bytes(tensors[key].tolist()))
	change = delta(files, name)
	if change is None:
		continue
	assert quant.get("convrot") and quant.get("convrot_groupsize") == 256 and quant.get("format") == "int8_tensorwise", quant
	weight = tensors[name + ".weight"].float().to(device) * tensors[name + ".weight_scale"].float().to(device)
	if change.shape != weight.shape:
		raise ValueError(f"LoRA shape {tuple(change.shape)} does not match {name}: {tuple(weight.shape)}")
	weight = weight + rotated(change, quant["convrot_groupsize"])
	scale = (weight.abs().amax(dim=1, keepdim=True) / 127).clamp(min=1e-30)
	noise = torch.rand(weight.shape, generator=generator, device=device)
	tensors[name + ".weight"] = (weight / scale + noise).floor().clamp(-128, 127).to(torch.int8).cpu()
	tensors[name + ".weight_scale"] = scale.to(tensors[name + ".weight_scale"].dtype).cpu()
	merged += 1
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
for (path, strength), count in zip(specs, matched):
	if count == 0:
		raise ValueError('LoRA matched no supported model modules: ' + path)
	print('matched', count, 'modules:', path, 'strength', strength)
print("merged", merged, "linears")
save_file(tensors, out, metadata={"merged_loras": json.dumps(loras)})
print("wrote", out)
