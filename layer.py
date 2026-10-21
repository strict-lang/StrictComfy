"""Layer 0 of Moss step by step in float32 torch from a step0 dump, to find where a port differs. Run with ComfyUI's python:
  python_embeded\\python.exe layer.py D:\\StrictDumps\\moss-base   (writes layer0.safetensors next to the dump)
"""
import json
import os
import sys
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.environ.get("COMFYUI", r"D:\ComfyUI_windows_portable\ComfyUI"))
from comfy.ldm.flux.math import rope  # noqa: E402

dumps = sys.argv[1]
model = sys.argv[2] if len(sys.argv) > 2 else r"D:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\moss-image-sft-dit-int8-convrot.safetensors"
step = load_file(os.path.join(dumps, "step0.safetensors"))
weights = safe_open(model, "pt")
h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
hadamard = h4
for _ in range(3):
	hadamard = torch.kron(hadamard, h4)
hadamard = hadamard / 16


def rotated(x):
	return (x.reshape(-1, x.shape[-1] // 256, 256) @ hadamard).reshape(x.shape)


def linear(x, name):
	quant = json.loads(bytes(weights.get_tensor(name + ".comfy_quant").tolist()))
	assert quant["convrot"] and quant["convrot_groupsize"] == 256
	w = weights.get_tensor(name + ".weight").float() * weights.get_tensor(name + ".weight_scale").float()
	return rotated(x) @ w.T


def rms(x, weight):
	return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * weight


def norm(name):
	return weights.get_tensor(name).float()


x = torch.cat([step["video_embedder"][0].float(), step["text_embedder"][0].float()])
text = step["context"].shape[1]
height, width = step["x"].shape[2], step["x"].shape[3]
shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = step["shared_block_modulation"][0].float().reshape(6, -1)
ids = torch.zeros(height * width + text, 4)
ids[:height * width, 1] = text
ids[:height * width, 2] = (torch.arange(height) - (height - 1) / 2).repeat_interleave(width)
ids[:height * width, 3] = (torch.arange(width) - (width - 1) / 2).repeat(height)
ids[height * width:, 1] = torch.arange(text)
pe = torch.cat([rope(ids[None, :, i], 32, 10000) for i in range(4)], dim=-3)[0]
y = rms(x, norm("layers.0.norm1.weight")) * (1 + scale_a) + shift_a
q, k, v = (linear(y, "layers.0.proj." + name) for name in "qkv")


def headed(t, name):
	pairs = rms(t.reshape(-1, 24, 128), norm(name)).unflatten(-1, (2, -1)).movedim(-2, -1).unsqueeze(-2)
	return (pe[:, None, ..., 0] * pairs[..., 0] + pe[:, None, ..., 1] * pairs[..., 1]).movedim(-1, -2).flatten(-2)


queries, keys = headed(q, "layers.0.q_norm.weight"), headed(k, "layers.0.k_norm.weight")
merged = torch.nn.functional.scaled_dot_product_attention(queries.transpose(0, 1), keys.transpose(0, 1), v.reshape(-1, 24, 128).transpose(0, 1)).transpose(0, 1).reshape(-1, 3072)
attended = x + gate_a * linear(merged, "layers.0.proj.o")
fed = rms(attended, norm("layers.0.norm2.weight")) * (1 + scale_f) + shift_f
hidden, gate = linear(fed, "layers.0.proj.ffn_up_gate").chunk(2, dim=-1)
activated = (4 * torch.tanh(gate / 4) * torch.sigmoid(gate)) * (25 * torch.tanh(hidden / 25))
output = attended + gate_f * linear(activated, "layers.0.proj.ffn_down")
expected = step["layers.0"][0].float()
print("layer 0 error against ComfyUI", float((output - expected).norm() / expected.norm()))
save_file({"y": y, "queries": queries.reshape(-1, 3072).contiguous(), "keys": keys.reshape(-1, 3072).contiguous(), "values": v, "merged": merged,
	"attended": attended, "activated": activated, "output": output}, os.path.join(dumps, "layer0.safetensors"))
