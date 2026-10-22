"""Merges LoRAs into an int8 convrot model once, so Strict (and ComfyUI) load the result like any model. Run with ComfyUI's python:
  python_embeded\\python.exe merge.py <model.safetensors> <out.safetensors> <lora.safetensors>[:strength] ...
Each convrot linear: W·Hᵀ (int8 · row scale) + strength · alpha / rank · (up · down)·Hᵀ, requantized to int8 with one scale per row
(nearest rounding, ComfyUI rounds stochastically). Everything else is copied unchanged.
"""
import json
import sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

model, out, loras = sys.argv[1], sys.argv[2], sys.argv[3:]
device = "cuda" if torch.cuda.is_available() else "cpu"
h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
hadamard = h4
for _ in range(3):
	hadamard = torch.kron(hadamard, h4)
hadamard = (hadamard / 16).to(device)


def rotated(rows, group):
	return (rows.reshape(rows.shape[0], -1, group) @ hadamard).reshape(rows.shape)


def delta(files, name):
	total = None
	for file, strength in files:
		prefix = "diffusion_model." + name + "."
		if prefix + "lora_down.weight" not in file.keys():
			continue
		down = file.get_tensor(prefix + "lora_down.weight").float().to(device)
		up = file.get_tensor(prefix + "lora_up.weight").float().to(device)
		alpha = float(file.get_tensor(prefix + "alpha")) if prefix + "alpha" in file.keys() else down.shape[0]
		part = strength * alpha / down.shape[0] * (up @ down)
		total = part if total is None else total + part
	return total


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
	tensors[name + ".weight"] = (weight / scale).round().clamp(-128, 127).to(torch.int8).cpu()
	tensors[name + ".weight_scale"] = scale.to(tensors[name + ".weight_scale"].dtype).cpu()
	merged += 1
print("merged", merged, "linears")
save_file(tensors, out, metadata={"merged_loras": json.dumps(loras)})
print("wrote", out)
