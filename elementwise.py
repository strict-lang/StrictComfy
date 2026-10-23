"""Compare generated RoPE kernels with CUDA events; run with ComfyUI python_embeded.

Usage: elementwise.py baseline-directory candidate-directory
Each directory contains MossImage.ptx and MossImage.kernels.mlir.
"""
import ctypes as ct
import re
import statistics
import sys
from pathlib import Path

import torch


def checked(status):
	assert status == 0, f"CUDA driver error {status}"


def magic(divisor):
	if divisor <= 0:
		return 0
	bits = (divisor - 1).bit_length()
	first = min(bits, 1)
	return (((1 << bits) - divisor) * (1 << 32) // divisor + 1) | (first << 32) | ((bits - first) << 40)


torch.manual_seed(42)
torch.cuda.init()
driver = ct.WinDLL("nvcuda.dll")
driver.cuModuleLoadData.argtypes = [ct.POINTER(ct.c_void_p), ct.c_char_p]
driver.cuModuleGetFunction.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p]
driver.cuLaunchKernel.argtypes = [ct.c_void_p] + [ct.c_uint] * 7 + [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_void_p]
driver.cuModuleUnload.argtypes = [ct.c_void_p]
tokens, width, size = 3740, 3072, 128
count = tokens * width
combined = torch.randn(tokens, 3 * width, device="cuda", dtype=torch.bfloat16)
weights = torch.randn(size, device="cuda")
rotation = torch.randn(tokens, 2, size, device="cuda")
output = torch.empty(count, device="cuda")
failed = torch.zeros(1, device="cuda", dtype=torch.int64)
values = dict(count=count, result=output.data_ptr(), failed=failed.data_ptr(), source=0, columns=0, bits=64, results=32)
for name, value in dict(size=size, width=width, start=0, half=size // 2, heads=width // size).items():
	values.update({"f_" + name: value, "e_" + name: value, "m_" + name: magic(value)})
for name, tensor, bits in [("combined", combined, 17), ("weights", weights, 32), ("rotation", rotation, 32)]:
	values.update({"f_" + name + "_d": tensor.data_ptr(), "f_" + name + "_n": tensor.numel(), "f_" + name + "_b": bits})
queries = combined.float()[:, :width].reshape(tokens, width // size, size)
partner = torch.cat((queries[:, :, size // 2:], queries[:, :, :size // 2]), dim=2)
partner_weights = torch.cat((weights[size // 2:], weights[:size // 2]))
scale = (queries.square().mean(dim=2, keepdim=True) + 1e-6).rsqrt()
expected = ((queries * weights * rotation[:, 0:1] + partner * partner_weights * rotation[:, 1:2]) * scale).flatten()
loaded = []
for directory in sys.argv[1:]:
	path = Path(directory)
	source = (path / "MossImage.kernels.mlir").read_text()
	name, signature = re.search(r"llvm.func @(MossBlock_Rotated_5_gpu\d+sx)\(([^\n]+)\) attributes", source).groups()
	arguments = [ct.c_double(values[key]) if kind == "f64" else ct.c_uint64(values[key]) for key, kind in re.findall(r"%(\w+): ([^,]+)", signature)]
	pointers = (ct.c_void_p * len(arguments))(*[ct.cast(ct.byref(value), ct.c_void_p) for value in arguments])
	module, function = ct.c_void_p(), ct.c_void_p()
	checked(driver.cuModuleLoadData(ct.byref(module), (path / "MossImage.ptx").read_bytes()))
	checked(driver.cuModuleGetFunction(ct.byref(function), module, name.encode()))
	loaded.append((directory, module, function, arguments, pointers))
times = {directory: [] for directory, *_ in loaded}
for repeat in range(6):
	for directory, module, function, arguments, pointers in (loaded if repeat % 2 == 0 else reversed(loaded)):
		before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
		before.record()
		for iteration in range(50):
			checked(driver.cuLaunchKernel(function, (count // size * 32 + 255) // 256, 1, 1, 256, 1, 1, 0, None, pointers, None))
		after.record()
		after.synchronize()
		times[directory].append(before.elapsed_time(after) * 20)
		assert failed.item() == 0
		torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-5)
for directory, module, *_ in loaded:
	print(f"{directory}: median {statistics.median(times[directory]):.1f} us; runs {times[directory]}", flush=True)
	checked(driver.cuModuleUnload(module))
