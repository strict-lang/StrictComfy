"""Converts a Moss int8 convrot checkpoint (ComfyUI's int8_tensorwise convrot, also with a fused qkv and the
model.diffusion_model. prefix) or a bf16/fp16 one into the int4 convrot_w4a4 layout of moss_image_sft_4step_int4_convrot:
unfused q/k/v, int4 weights packed two per byte with one scale per row, quantized by ComfyUI's own comfy_kitchen after
clipping each row's outliers where the int4 error is smallest.
Run with ComfyUI's python:
  python_embeded\\python.exe quantize4.py <model.safetensors> <out.safetensors> [<reference int8 model> to compare] [--keep <regex>]
--keep leaves the linears whose name (layers.<n>.proj.<q|k|v|o|ffn_up_gate|ffn_down>) matches in int8 (mixed precision),
the source must then be an unfused int8 convrot model like merge.py writes.
"""
import json
import re
import sys
from pathlib import Path
import torch
from comfy_kitchen.tensor.convrot_w4a4 import quantize_convrot_w4a4_weight, dequantize_convrot_w4a4_weight
from safetensors import safe_open
from safetensors.torch import save_file

arguments = sys.argv[1:]
keep = ""
if "--keep" in arguments:
	at = arguments.index("--keep")
	keep = arguments[at + 1]
	del arguments[at:at + 2]
source, out = arguments[0], arguments[1]
reference = arguments[2] if len(arguments) > 2 else ""
if Path(out).resolve() == Path(source).resolve():
	raise ValueError("Output must differ from the input file")
device = "cuda" if torch.cuda.is_available() else "cpu"
h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
hadamard = h4
for _ in range(3):
	hadamard = torch.kron(hadamard, h4)
hadamard = (hadamard / 16).to(device)
LINEARS = ("proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate", "proj.ffn_down")
LAYER = {"format": "convrot_w4a4", "convrot_groupsize": 256, "quant_group_size": 64}
PREFIX = "model.diffusion_model."


def rotated(rows, group=256):
	# the regular Hadamard is symmetric and orthonormal, so the same product rotates and rotates back
	return (rows.reshape(rows.shape[0], -1, group) @ hadamard).reshape(rows.shape)


def clipped(value):
	"""The weight with each row's rotated values clipped where int4 rounding (-7..7) errs least: comfy_kitchen scales a
	row by its largest value / 7, which leaves most weights on the middle 7 codes (15.5 % weight error); clipping the
	outliers first uses all codes (11.3 %, like the hosted MossInt4 files)."""
	turned = rotated(value.float())
	largest = turned.abs().amax(1, keepdim=True).clamp_min(1e-12)
	best, chosen = None, None
	for ratio in torch.linspace(0.3, 1.0, 36).tolist():
		step = largest * ratio / 7
		error = ((torch.clamp(torch.round(turned / step), -7, 7) * step - turned) ** 2).sum(1, keepdim=True)
		better = error if best is None else error < best
		best = error if best is None else torch.where(better, error, best)
		chosen = torch.full_like(error, ratio) if chosen is None else torch.where(better, torch.full_like(error, ratio), chosen)
	limit = largest * chosen
	return rotated(torch.maximum(torch.minimum(turned, limit), -limit))


def weight(file, name):
	"""The floating point weight of a linear: int8 convrot rows times their scale rotated back, or a plain float."""
	tensor = file.get_tensor(name + ".weight").to(device)
	if tensor.dtype != torch.int8:
		return tensor.float()
	quant = json.loads(bytes(file.get_tensor(name + ".comfy_quant").tolist()).decode()) if name + ".comfy_quant" in file.keys() else {}
	scale = file.get_tensor(name + ".weight_scale").to(device).float().reshape(-1, 1)
	rows = tensor.float() * scale
	return rotated(rows, quant.get("convrot_groupsize", 256)) if quant.get("convrot", True) else rows


def linears(file, layer):
	"""The q, k, v, o, ffn_up_gate and ffn_down weights of one layer, splitting a fused qkv into thirds."""
	keys = set(file.keys())
	found = {}
	if layer + "proj.qkv.weight" in keys:
		fused = weight(file, layer + "proj.qkv")
		third = fused.shape[0] // 3
		found["proj.q"], found["proj.k"], found["proj.v"] = fused[:third], fused[third:2 * third], fused[2 * third:]
	for name in LINEARS:
		if layer + name + ".weight" in keys:
			found[name] = weight(file, layer + name)
	return found


file = safe_open(source, "pt", device="cpu")
names = list(file.keys())
prefix = PREFIX if any(name.startswith(PREFIX) for name in names) else ""
layers = sorted({name[len(prefix):].split(".")[1] for name in names if name[len(prefix):].startswith("layers.")}, key=int)
tensors, metadata = {}, {"format_version": "1.0", "layers": {}}
skipped = (".comfy_quant", ".comfy_attention.config", ".weight_scale")
compare = safe_open(reference, "pt", device="cpu") if reference else None
errors, drifts, kept = [], [], []
for index in layers:
	source_layer = prefix + "layers." + index + "."
	for name, value in linears(file, source_layer).items():
		target = "layers." + index + "." + name
		if keep and re.search(keep, target):
			# mixed precision: this linear stays in the source's int8 convrot form, copied unchanged
			if file.get_tensor(source_layer + name + ".weight").dtype != torch.int8:
				raise ValueError("--keep needs an unfused int8 convrot source, " + target + " is not int8")
			for part in (".weight", ".weight_scale", ".comfy_quant"):
				tensors[target + part] = file.get_tensor(source_layer + name + part)
			kept.append(target)
			continue
		packed, scales = quantize_convrot_w4a4_weight(clipped(value).to(torch.bfloat16), 256, 64, 0)
		tensors[target + ".weight"] = packed.cpu()
		tensors[target + ".weight_scale"] = scales.cpu()
		metadata["layers"][target] = LAYER
		back = dequantize_convrot_w4a4_weight(packed, scales, 256, 64, torch.float32)
		errors.append(((back - value).norm() / value.norm()).item())
		if compare is not None:
			base = linears(compare, "layers." + index + ".").get(name)
			if base is not None:
				drifts.append(((value - base).norm() / base.norm()).item())
	print("layer", index, "done", flush=True)
for name in names:
	short = name[len(prefix):]
	linear = short.startswith("layers.") and any(("." + item + ".") in short for item in LINEARS + ("proj.qkv",))
	if linear or short.endswith(skipped) or short == "__metadata__":
		continue
	tensors[short] = file.get_tensor(name)
if "rope.bands" not in tensors:
	# the rope frequency bands every Moss checkpoint carries (some merges drop them): 10 ** (-index / 4) for 16 bands
	tensors["rope.bands"] = torch.pow(10.0, -torch.arange(16, dtype=torch.float32) / 4)
save_file(tensors, out, metadata={"format": "pt", "model": "moss-image", "_quantization_metadata": json.dumps(metadata)})
print("wrote", out, len(tensors), "tensors,", len(metadata["layers"]), "int4 linears,", len(kept), "int8 linears kept")
print("int4 round trip error: mean", sum(errors) / len(errors), "max", max(errors))
if drifts:
	print("distance to the reference model: mean", sum(drifts) / len(drifts), "max", max(drifts))
