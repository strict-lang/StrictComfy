"""Moss layers step by step in float32 torch from a step0 dump, to find where a port differs. Run with ComfyUI's python:
  python_embeded\\python.exe layer.py D:\\StrictDumps\\moss-base [--lora file] [--layers 42]
Prints every layer's error against ComfyUI's dumped outputs and writes layer 0's intermediates to layer0.safetensors next to the dump.
The LoRA is applied exactly in float32 ((alpha / rank) * up(down(x)) on the unrotated input), ComfyUI merges and requantizes it.
"""
import argparse
import json
import os
import sys
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.environ.get("COMFYUI", r"D:\ComfyUI_windows_portable\ComfyUI"))
from comfy.ldm.flux.math import rope  # noqa: E402

arguments = argparse.ArgumentParser()
arguments.add_argument("dumps")
arguments.add_argument("--model", default=r"D:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\moss-image-sft-dit-int8-convrot.safetensors")
arguments.add_argument("--lora", default="")
arguments.add_argument("--layers", type=int, default=42)
arguments.add_argument("--forced", action="store_true", help="every layer starts from ComfyUI's output of the layer before")
arguments.add_argument("--rotated-lora", action="store_true", help="add the LoRA delta to the rotated weight like a naive merge")
options = arguments.parse_args()
device = "cuda"
step = load_file(os.path.join(options.dumps, "step0.safetensors"))
weights = safe_open(options.model, "pt")
lora = safe_open(options.lora, "pt") if options.lora else None
h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
hadamard = h4
for _ in range(3):
	hadamard = torch.kron(hadamard, h4)
hadamard = (hadamard / 16).to(device)


def tensor(name, source=weights):
	return source.get_tensor(name).float().to(device)


def rotated(x):
	return (x.reshape(-1, x.shape[-1] // 256, 256) @ hadamard).reshape(x.shape)


def linear(x, name):
	quant = json.loads(bytes(weights.get_tensor(name + ".comfy_quant").tolist()))
	assert quant["convrot"] and quant["convrot_groupsize"] == 256
	output = rotated(x) @ (tensor(name + ".weight") * tensor(name + ".weight_scale")).T
	if lora is not None:
		prefix = "diffusion_model." + name + "."
		down, up = tensor(prefix + "lora_down.weight", lora), tensor(prefix + "lora_up.weight", lora)
		source = rotated(x) if options.rotated_lora else x
		output = output + float(lora.get_tensor(prefix + "alpha")) / down.shape[0] * ((source @ down.T) @ up.T)
	return output


def rms(x, weight):
	return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * weight


text = step["context"].shape[1]
height, width = step["x"].shape[2], step["x"].shape[3]
shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = step["shared_block_modulation"][0].float().to(device).reshape(6, -1)
ids = torch.zeros(height * width + text, 4)
ids[:height * width, 1] = text
ids[:height * width, 2] = (torch.arange(height) - (height - 1) / 2).repeat_interleave(width)
ids[:height * width, 3] = (torch.arange(width) - (width - 1) / 2).repeat(height)
ids[height * width:, 1] = torch.arange(text)
pe = torch.cat([rope(ids[None, :, i], 32, 10000) for i in range(4)], dim=-3)[0].to(device)


def headed(t, name):
	pairs = rms(t.reshape(-1, 24, 128), tensor(name)).unflatten(-1, (2, -1)).movedim(-2, -1).unsqueeze(-2)
	return (pe[:, None, ..., 0] * pairs[..., 0] + pe[:, None, ..., 1] * pairs[..., 1]).movedim(-1, -2).flatten(-2)


def block(x, index):
	prefix = f"layers.{index}."
	y = rms(x, tensor(prefix + "norm1.weight")) * (1 + scale_a) + shift_a
	q, k, v = (linear(y, prefix + "proj." + name) for name in "qkv")
	queries, keys = headed(q, prefix + "q_norm.weight"), headed(k, prefix + "k_norm.weight")
	merged = torch.nn.functional.scaled_dot_product_attention(queries.transpose(0, 1), keys.transpose(0, 1),
		v.reshape(-1, 24, 128).transpose(0, 1)).transpose(0, 1).reshape(-1, 3072)
	attended = x + gate_a * linear(merged, prefix + "proj.o")
	fed = rms(attended, tensor(prefix + "norm2.weight")) * (1 + scale_f) + shift_f
	hidden, gate = linear(fed, prefix + "proj.ffn_up_gate").chunk(2, dim=-1)
	activated = (4 * torch.tanh(gate / 4) * torch.sigmoid(gate)) * (25 * torch.tanh(hidden / 25))
	output = attended + gate_f * linear(activated, prefix + "proj.ffn_down")
	return output, {"y": y, "queries": queries.reshape(-1, 3072).contiguous(), "keys": keys.reshape(-1, 3072).contiguous(), "values": v,
		"merged": merged, "attended": attended, "activated": activated, "output": output}


with torch.no_grad():
	x = torch.cat([step["video_embedder"][0].float(), step["text_embedder"][0].float()]).to(device)
	errors = []
	for index in range(options.layers):
		x, parts = block(x, index)
		if index == 0:
			save_file({name: value.cpu() for name, value in parts.items()}, os.path.join(options.dumps, "layer0.safetensors"))
		expected = step[f"layers.{index}"][0].float().to(device)
		errors.append(f"{index}:{float((x - expected).norm() / expected.norm()):.3f}")
		x = expected if options.forced else x
	print("float32 layers against ComfyUI", " ".join(errors))
