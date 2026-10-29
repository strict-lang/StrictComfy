"""Run with ComfyUI python_embeded from StrictComfy after building MossImage.strict.
Synthetic CUDA-event timings and output checks; never changes generated PTX files.
Unsigned RoPE variants assume nonnegative indices for this experiment only.
"""
import ctypes as ct
import re
import statistics
from pathlib import Path

import torch
import comfy_kitchen.backends.cuda as kitchen

torch.manual_seed(42)
torch.cuda.init()
driver = ct.WinDLL("nvcuda.dll")
driver.cuModuleLoadData.argtypes = [ct.POINTER(ct.c_void_p), ct.c_char_p]
driver.cuModuleGetFunction.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p]
driver.cuLaunchKernel.argtypes = [ct.c_void_p] + [ct.c_uint] * 7 + [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_void_p]
driver.cuModuleUnload.argtypes = [ct.c_void_p]


def checked(status):
	assert status == 0, f"CUDA driver error {status}"


def magic(divisor):
	bits = (divisor - 1).bit_length()
	first = min(bits, 1)
	return (((1 << bits) - divisor) * (1 << 32) // divisor + 1) | (first << 32) | ((bits - first) << 40)


def timed(launch):
	times = []
	for repeat in range(7):
		before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
		before.record()
		for iteration in range(100):
			launch()
		after.record()
		after.synchronize()
		times.append(before.elapsed_time(after) * 10)
	return statistics.median(times)


name = "MossBlock_Rotated_6_gpu14sx"
source = Path("MossImage.ptx").read_text()
start = source.index(".visible .entry " + name + "(")
end = source.index("\n}", start) + 2
kernel = source[start:end]
assert "call.uni" not in kernel
header = "\n".join(re.findall(r"^\.(?:version|target|address_size).*", source, re.M)) + "\n"
unsigned, negatives = re.subn(r"setp.lt.s64\s+(%p\d+), %rd\d+, 0;", r"setp.eq.u32 \1, 0, 1;", kernel)
unsigned, floors = re.subn(r"shr.s64\s+(%rd\d+), %rd\d+, 63;", r"mov.u64 \1, 0;", unsigned)
unsigned, absolutes = re.subn(r"max.s64\s+(%rd\d+), (%rd\d+), %rd\d+;", r"mov.b64 \1, \2;", unsigned)
unsigned = unsigned.replace("div.s64", "div.u64")
assert (negatives, floors, absolutes) == (2, 2, 2), (negatives, floors, absolutes)
tokens, width, size = 3740, 3072, 128
count = tokens * width
for bits, dtype in [(32, torch.float32), (17, torch.bfloat16)]:
	combined = torch.randn(tokens, 3 * width, device="cuda", dtype=dtype)
	weights = torch.randn(size, device="cuda")
	rotation = torch.randn(tokens, 2 * size, device="cuda")
	scales = torch.randn(tokens * 3 * width // size, device="cuda")
	output = torch.empty(count, device="cuda")
	failed = torch.zeros(2, device="cuda", dtype=torch.int32)
	baseline = None
	for offset in (0, width):
		values = [count, output.data_ptr(), failed.data_ptr(), 0, 0, 0, float(width), float(size), float(offset), float(size // 2), combined.data_ptr(), combined.numel(), bits, weights.data_ptr(), size, 32, rotation.data_ptr(), rotation.numel(), 32, scales.data_ptr(), scales.numel(), 32, 32, width, magic(width), size, magic(size), offset, magic(offset) if offset else 0, size // 2, magic(size // 2)]
		arguments = [ct.c_double(value) if isinstance(value, float) else ct.c_uint64(value) for value in values]
		pointers = (ct.c_void_p * len(arguments))(*(ct.addressof(value) for value in arguments))
		constants = {index: values[index] for index in [12, 15, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]}
		pattern = r"ld.param.b64\s+(%rd\d+), \[" + name + r"_param_(\d+)\];"
		specialized = re.sub(pattern, lambda match: f"mov.b64 {match[1]}, {constants[int(match[2])]};" if int(match[2]) in constants else match[0], unsigned)
		for label, body in [("baseline", kernel), ("unsigned", unsigned), ("specialized", specialized)]:
			module, function = ct.c_void_p(), ct.c_void_p()
			checked(driver.cuModuleLoadData(ct.byref(module), (header + body).encode()))
			checked(driver.cuModuleGetFunction(ct.byref(function), module, name.encode()))
			def launch():
				checked(driver.cuLaunchKernel(function, (count + 255) // 256, 1, 1, 256, 1, 1, 0, torch.cuda.current_stream().cuda_stream, pointers, None))
			for repeat in range(10):
				launch()
			torch.cuda.synchronize()
			assert failed.sum().item() == 0
			if label == "baseline":
				baseline = output.clone()
				rows = combined[:, offset:offset + width].float().reshape(tokens, width // size, size)
				weighted = rows * weights
				paired = torch.cat((weighted[:, :, size // 2:], weighted[:, :, :size // 2]), dim=2)
				reference = (weighted * rotation[:, None, :size] + paired * rotation[:, None, size:]) * scales.reshape(tokens, 3 * width // size)[:, offset // size:(offset + width) // size, None]
				torch.testing.assert_close(output, reference.flatten(), rtol=1e-6, atol=1e-6)
			else:
				assert torch.equal(output, baseline), f"{label} changed output"
			print(bits, offset, label, f"{timed(launch):.2f} us", "checks passed", flush=True)
			checked(driver.cuModuleUnload(module))

name = "strict_device_wholes"
start = source.index(".visible .entry " + name + "(")
kernel = source[start:source.index("\n}", start) + 2]
shared = "\n".join(re.findall(r"^(?:\.extern )?\.shared .*", source, re.M))
helpers = "\n".join(re.search(r"^\.visible \.func[^\n]* " + helper + r"\([\s\S]*?^}", source, re.M)[0] for helper in ("strict_device_float8_read", "strict_device_byte_read"))
module, function = ct.c_void_p(), ct.c_void_p()
print("loading GEMM", flush=True)
checked(driver.cuModuleLoadData(ct.byref(module), (header + shared + "\n" + helpers + "\n" + kernel).encode()))
checked(driver.cuModuleGetFunction(ct.byref(function), module, name.encode()))
driver.cuFuncSetAttribute.argtypes = [ct.c_void_p, ct.c_int, ct.c_int]
checked(driver.cuFuncSetAttribute(function, 8, 73728))
for rows, width in [(9216, 3072), (3072, 3072), (16384, 3072), (3072, 8192)]:
	tokens = 3740
	inputs = torch.randint(-127, 128, (tokens, width), dtype=torch.int8, device="cuda")
	weights = torch.randint(-127, 128, (rows, width), dtype=torch.int8, device="cuda")
	input_scales = 2.0 ** torch.randint(-9, -5, (tokens, 1), device="cuda", dtype=torch.float32)
	weight_scales = 2.0 ** torch.randint(-9, -5, (rows,), device="cuda", dtype=torch.float32)
	bias = torch.zeros(rows, device="cuda")
	output = torch.empty((tokens, rows), device="cuda", dtype=torch.bfloat16)
	reference = torch.empty_like(output)
	stride = width + width // 32 * 4
	header_bytes = (rows + 8 + 127) & -128
	packed_weights = torch.zeros(header_bytes + rows * stride, dtype=torch.uint8, device="cuda")
	packed_weights[:8].view(torch.int64).fill_((rows << 32) | (1 << 31) | stride)
	packed_weights[8:8 + rows].fill_(8)
	packed_rows = packed_weights[header_bytes:].reshape(rows, stride)
	packed_rows[:, :width].copy_(weights.view(torch.uint8))
	packed_rows[:, width:].view(torch.float32).copy_(weight_scales[:, None])
	packed_inputs = torch.cat((inputs.flatten().view(torch.uint8), input_scales.expand(tokens, width // 32).contiguous().flatten().view(torch.uint8)))
	failed = torch.zeros(2, dtype=torch.int32, device="cuda")
	values = [tokens * rows, output.data_ptr(), failed.data_ptr(), 0, 0, 0, packed_weights.data_ptr(), width, 8, packed_inputs.data_ptr(), inputs.numel(), 8, float(width), float(rows), bias.data_ptr(), rows, 32, 0, 0, 0, 17]
	arguments = [ct.c_double(value) if isinstance(value, float) else ct.c_uint64(value) for value in values]
	pointers = (ct.c_void_p * len(arguments))(*(ct.addressof(value) for value in arguments))
	stream = torch.cuda.current_stream().cuda_stream
	kitchen_bias = bias.to(torch.bfloat16)
	def strict():
		checked(driver.cuLaunchKernel(function, ((tokens + 127) // 128) * ((rows + 255) // 256), 1, 1, 256, 1, 1, 73728, stream, pointers, None))
	def cutlass():
		wrapped = [kitchen._wrap_for_dlpack(value) for value in (inputs, weights, input_scales, weight_scales, kitchen_bias, reference)]
		assert kitchen._C.cutlass_int8_dequant(*wrapped, kitchen.DTYPE_TO_CODE[torch.bfloat16], stream)
	strict()
	cutlass()
	torch.cuda.synchronize()
	assert failed.sum().item() == 0
	torch.testing.assert_close(output, reference, rtol=0, atol=0)
	for label, launch in [("Strict", strict), ("CUTLASS", cutlass)]:
		for repeat in range(10):
			launch()
		print(tokens, rows, width, label, f"{timed(launch):.2f} us", "bit-exact", flush=True)
checked(driver.cuModuleUnload(module))
