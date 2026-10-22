"""Merges LoRAs into an int8 convrot model once, so Strict (and ComfyUI) load the result like any model. Run with ComfyUI's python:
  python_embeded\\python.exe merge.py <model.safetensors> <out.safetensors> <lora.safetensors>[:strength] ...
Each convrot linear: W·Hᵀ (int8 · row scale) + strength · alpha / rank · (up · down)·Hᵀ, requantized to int8 with one scale per row
(stochastic rounding like ComfyUI: nearest rounding drops LoRA deltas below half a step). Everything else is copied unchanged. LoRAs trained on a fused proj.qkv
match no weight of this unfused model and are skipped like ComfyUI does (strengths tuned in ComfyUI expect that);
--split-qkv applies their q/k/v thirds instead.
"""
import json
import sys
import torch
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
	for file, strength in files:
		part = product(file, "diffusion_model." + name + ".")
		parent, own = name.rsplit(".", 1)
		if part is None and own in ("q", "k", "v") and split:
			fused = product(file, "diffusion_model." + parent + ".qkv.")
			if fused is not None:
				third = fused.shape[0] // 3
				part = fused["qkv".index(own) * third:("qkv".index(own) + 1) * third]
		if part is not None:
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
files = [(safe_open(path.rsplit(":", 1)[0] if path.count(":") > 1 else path, "pt"), float(path.rsplit(":", 1)[1]) if path.count(":") > 1 else 1.0)
	for path in loras]
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
	weight = weight + rotated(change, quant["convrot_groupsize"])
	scale = (weight.abs().amax(dim=1, keepdim=True) / 127).clamp(min=1e-30)
	noise = torch.rand(weight.shape, generator=generator, device=device)
	tensors[name + ".weight"] = (weight / scale + noise).floor().clamp(-128, 127).to(torch.int8).cpu()
	tensors[name + ".weight_scale"] = scale.to(tensors[name + ".weight_scale"].dtype).cpu()
	merged += 1
print("merged", merged, "linears")
save_file(tensors, out, metadata={"merged_loras": json.dumps(loras)})
print("wrote", out)
