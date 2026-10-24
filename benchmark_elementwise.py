"""Check and time Moss 720x1280 kernels: benchmark_elementwise.py baseline candidate.

Use ComfyUI's python_embeded. Each directory needs MossImage.ptx and
MossImage.kernels.mlir. CUDA graphs keep Python launch gaps out of timings.
STRICT_BENCH_KERNEL selects a substring; STRICT_BENCH_PROFILE only validates
one full-size bfloat16 launch per variant (RoPE uses the model format mix), for ncu.
STRICT_BENCH_GENERIC forces generic kernels; STRICT_BENCH_BOUNDS checks invalid lengths. STRICT_BENCH_GEMM=1 selects
int8 GEMM; STRICT_BENCH_BITS=32 switches its default bfloat16 output to float32.
STRICT_BENCH_GEMM_FAST selects a uniform-row specialization when present;
STRICT_BENCH_WHOLE_GUARD checks both its rejection and the legacy fallback.
"""
import ctypes as ct
import os
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


def report(calls, label):
	graphs = []
	for directory, launch, *_ in calls:
		graph = torch.cuda.CUDAGraph(keep_graph=True)
		with torch.cuda.graph(graph):
			for iteration in range(50):
				launch()
		nodes = ct.c_size_t()
		checked(driver.cuGraphGetNodes(ct.c_void_p(graph.raw_cuda_graph()), None, ct.byref(nodes)))
		assert nodes.value >= 50, f"Incomplete CUDA graph for {directory}: {nodes.value} nodes"
		graph.replay()
		graphs.append((directory, graph))
	torch.cuda.synchronize()
	times = {directory: [] for directory, _ in graphs}
	for repeat in range(6):
		for directory, graph in (graphs if repeat % 2 == 0 else reversed(graphs)):
			before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
			before.record()
			graph.replay()
			after.record()
			after.synchronize()
			times[directory].append(before.elapsed_time(after) * 20)
	for directory, timings in times.items():
		print(f"{label} {directory}: {statistics.median(timings):.1f} us {timings}", flush=True)


def parameters(signature, values):
	arguments = [ct.c_double(values[key]) if kind == "f64" else ct.c_uint64(values[key]) for key, kind in re.findall(r"%(\w+): ([^,]+)", signature)]
	pointers = (ct.c_void_p * len(arguments))(*[ct.cast(ct.byref(value), ct.c_void_p) for value in arguments])
	return arguments, pointers


torch.manual_seed(42)
torch.cuda.init()
context = torch.empty(1, device="cuda")
driver = ct.WinDLL("nvcuda.dll")
driver.cuModuleLoadData.argtypes = [ct.POINTER(ct.c_void_p), ct.c_char_p]
driver.cuModuleGetFunction.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p]
driver.cuLaunchKernel.argtypes = [ct.c_void_p] + [ct.c_uint] * 7 + [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_void_p]
driver.cuModuleUnload.argtypes = [ct.c_void_p]
driver.cuFuncSetAttribute.argtypes = [ct.c_void_p, ct.c_int, ct.c_int]
loaded = []
for directory in sys.argv[1:]:
	path = Path(directory)
	module = ct.c_void_p()
	checked(driver.cuModuleLoadData(ct.byref(module), (path / "MossImage.ptx").read_bytes()))
	loaded.append((directory, module, (path / "MossImage.kernels.mlir").read_text()))

kernels = ["MossBlock_Activated_2", "MossBlock_Gated_4", "Batch_Columns_4", "MossBlock_Rotated_5", "MossBlock_Modulated_4"]
kernels = [kernel for kernel in kernels if os.getenv("STRICT_BENCH_KERNEL", "") in kernel]
kernels = [kernel for kernel in kernels if all("@" + kernel + "_gpu" in source for _, _, source in loaded)]
for kernel in ([] if os.getenv("STRICT_BENCH_GEMM") else kernels):
	width = 8192 if "Activated" in kernel else 3072
	full_count = 3740 * width
	counts = [128, 256, 3072, 4096, full_count] if "Rotated" in kernel else [1, 255, 257, 1023, 1025, full_count - 3, full_count]
	counts = [width, 2 * width, full_count] if "Modulated" in kernel else counts
	counts = [full_count] if os.getenv("STRICT_BENCH_PROFILE") else counts
	for count in counts:
		formats = [(32, False), (17, False), (16, False)] + ([(17, True)] if "Rotated" in kernel or "Modulated" in kernel else [])
		for bits, mixed in ([(17, "Rotated" in kernel)] if os.getenv("STRICT_BENCH_PROFILE") else formats):
			dtype = {32: torch.float32, 17: torch.bfloat16, 16: torch.float16}[bits]
			rows = (count + width - 1) // width
			values = {}
			keep = []
			def number(name, value):
				values.update({"f_" + name: value, "e_" + name: value, "m_" + name: magic(value)})
			def tensor(name, shape):
				tensor_bits = 32 if mixed and name in ("weights", "rotation", "modulation") else bits
				data = torch.randn(shape, device="cuda", dtype=torch.float32 if tensor_bits == 32 else dtype)
				keep.append(data)
				values.update({"f_" + name + "_d": data.data_ptr(), "f_" + name + "_n": data.numel(), "f_" + name + "_b": tensor_bits})
				return data.float()
			number("width", width)
			if "Activated" in kernel:
				expanded = tensor("expanded", (rows, 2, width))
				hidden, gate = expanded[:, 0], expanded[:, 1]
				expected = (4 * (2 / (1 + torch.exp(-gate / 2)) - 1) / (1 + torch.exp(-gate)))
				expected = expected * 25 * (2 / (1 + torch.exp(-hidden * 2 / 25)) - 1)
				expected = expected.flatten()[:count].bfloat16()
				output_dtype, result_bits = torch.bfloat16, 17
			elif "Modulated" in kernel:
				number("part", 0)
				input_data = tensor("input", (rows, width))
				weights = tensor("weights", width)
				modulation = tensor("modulation", (6, width))
				scale = (input_data.square().mean(dim=1, keepdim=True) + 1e-6).rsqrt()
				factor = scale * weights * (1 + modulation[1])
				expected = (input_data * factor + modulation[0]).flatten()[:count].bfloat16()
				output_dtype, result_bits = torch.bfloat16, 17
			elif "Gated" in kernel:
				number("part", 2)
				input_data = tensor("input", count)
				update = tensor("update", count)
				modulation = tensor("modulation", (3, width))
				expected = input_data + update * modulation[2].repeat(rows)[:count]
				output_dtype, result_bits = torch.float32, 32
			elif "Rotated" in kernel:
				number("size", 128)
				number("half", 64)
				number("heads", width // 128)
				number("start", 0)
				combined = tensor("combined", (rows, 3, width))
				weights = tensor("weights", 128)
				rotation = tensor("rotation", (rows, 2, 128))
				queries = combined[:, 0].reshape(rows, width // 128, 128)
				partner = torch.cat((queries[:, :, 64:], queries[:, :, :64]), dim=2)
				partner_weights = torch.cat((weights[64:], weights[:64]))
				scale = (queries.square().mean(dim=2, keepdim=True) + 1e-6).rsqrt()
				expected = ((queries * weights * rotation[:, 0:1] + partner * partner_weights * rotation[:, 1:2]) * scale).flatten()[:count]
				output_dtype, result_bits = torch.float32, 32
			else:
				number("count", width)
				number("width", 3 * width)
				number("start", 2 * width)
				source = tensor("values", (rows, 3, width))
				expected = source[:, 2].flatten()[:count]
				output_dtype, result_bits = torch.float32, 32
			output = torch.empty(count + 32, device="cuda", dtype=output_dtype)
			failed = torch.zeros(1, device="cuda", dtype=torch.int64)
			values.update(count=count, result=output.data_ptr(), failed=failed.data_ptr(), source=0, columns=0, bits=64, results=result_bits)
			calls = []
			for directory, module, source in loaded:
				match = re.search(r"llvm.func @(" + kernel + r"_gpu\d+sx)\(([^\n]+)\) attributes[^\n]+\{(.*?)\n  \}", source, re.S)
				assert match, kernel
				name, signature, body = match.groups()
				input_formats = [values[key] for key in re.findall(r"%(f_\w+_b): i64", signature)]
				format_index = (0 if all(value == 32 for value in input_formats) else
					1 if all(value == 17 for value in input_formats) else
					2 if input_formats == [17] + [32] * (len(input_formats) - 1) else -1)
				if input_formats and format_index >= 0 and not os.getenv("STRICT_BENCH_GENERIC"):
					formatted = name + "f" + str(format_index)
					if "@" + formatted + "(" in source:
						name = formatted
				arguments, pointers = parameters(signature, values)
				function = ct.c_void_p()
				checked(driver.cuModuleGetFunction(ct.byref(function), module, name.encode()))
				batch = 4 if "%batchSize = arith.constant 4" in body else 1
				threads = count // 128 * 32 if "Rotated" in kernel else (count + batch - 1) // batch
				threads = rows * 256 if "Modulated" in kernel else threads
				blocks = (threads + 255) // 256
				def launch(function=function, blocks=blocks, pointers=pointers):
					stream = ct.c_void_p(torch.cuda.current_stream().cuda_stream)
					checked(driver.cuLaunchKernel(function, blocks, 1, 1, 256, 1, 1, 0, stream, pointers, None))
				output.fill_(123)
				launch()
				torch.cuda.synchronize()
				assert failed.item() == 0, (kernel, directory)
				torch.testing.assert_close(output[:count], expected, rtol=0.008 if result_bits == 17 else 2e-5, atol=0.008 if result_bits == 17 else 2e-5, msg=lambda detail: f"{kernel} {directory} count={count} bits={bits}\n{detail}")
				assert (output[count:] == 123).all(), (kernel, directory, "tail overwrite")
				if os.getenv("STRICT_BENCH_BOUNDS") and count == counts[0]:
					for key in (key for key in values if key.startswith("f_") and key.endswith("_n")):
						invalid = dict(values, **{key: 0})
						invalid_arguments, invalid_pointers = parameters(signature, invalid)
						failed.zero_()
						checked(driver.cuLaunchKernel(function, blocks, 1, 1, 256, 1, 1, 0,
							ct.c_void_p(torch.cuda.current_stream().cuda_stream), invalid_pointers, None))
						torch.cuda.synchronize()
						assert failed.item() == 1, (kernel, directory, key, "bounds failure missing")
						assert (output[count:] == 123).all(), (kernel, directory, key, "tail overwrite")
					failed.zero_()
				calls.append((directory, launch, arguments, pointers))
			if count != full_count or os.getenv("STRICT_BENCH_PROFILE"):
				continue
			report(calls, f"{kernel} input={bits}{"/32/32" if mixed else ""}")
	print(f"{kernel}: formats and tail guards passed", flush=True)
if os.getenv("STRICT_BENCH_GEMM"):
	result_bits = int(os.getenv("STRICT_BENCH_BITS", "17"))
	output_dtype = torch.bfloat16 if result_bits == 17 else torch.float32
	for tokens, width, rows in [(129, 1024, 257), (3740, 3072, 9216), (3740, 3072, 3072), (3740, 3072, 16384), (3740, 8192, 3072)]:
		weights = torch.randint(-8, 9, (rows, width), device="cuda", dtype=torch.int8)
		inputs = torch.randint(-8, 9, (tokens, width), device="cuda", dtype=torch.int8)
		header, stride = (rows + 8 + 127) & -128, (width * 9 // 8 + 15) & -16
		packed_weights = torch.zeros(header + rows * stride, device="cuda", dtype=torch.uint8)
		packed_weights.view(torch.int64)[0] = (rows << 32) | (1 << 31) | stride
		packed_weights[8:8 + rows] = 8
		weight_rows = packed_weights[header:].view(rows, stride)
		weight_rows[:, :width] = weights.view(torch.uint8)
		weight_rows.view(torch.float32)[:, width // 4:width * 9 // 32] = 0.01
		packed_inputs = torch.empty(tokens * width * 9 // 8, device="cuda", dtype=torch.uint8)
		packed_inputs[:tokens * width] = inputs.flatten().view(torch.uint8)
		packed_inputs[tokens * width:].view(torch.float32).fill_(0.02)
		bias = torch.zeros(1, device="cuda")
		failed = torch.zeros(1, device="cuda", dtype=torch.int64)
		output = torch.empty(tokens * rows + 32, device="cuda", dtype=output_dtype)
		expected = ((inputs.float() @ weights.float().T).flatten() * (torch.tensor(0.01, device="cuda") * 0.02)).to(output_dtype)
		values = dict(count=tokens * rows, result=output.data_ptr(), failed=failed.data_ptr(), source=0, columns=width, bits=64,
			rowsData=packed_weights.data_ptr(), rowsCount=width, rowsBits=8, input=packed_inputs.data_ptr(), inputCount=tokens * width,
			inputBits=8, width=width, rows=rows, biases=bias.data_ptr(), biasesCount=1, biasesBits=32, bases=0, basesCount=0, basesBits=64, results=result_bits)
		calls = []
		for directory, module, source in loaded:
			kernel_name = "strict_device_wholes_fast" if os.getenv("STRICT_BENCH_GEMM_FAST") and "@strict_device_wholes_fast(" in source else "strict_device_wholes"
			name, signature = re.search(r"llvm.func @(" + kernel_name + r")\(([^\n]+)\) attributes", source).groups()
			arguments, pointers = parameters(signature, values)
			function = ct.c_void_p()
			checked(driver.cuModuleGetFunction(ct.byref(function), module, name.encode()))
			checked(driver.cuFuncSetAttribute(function, 8, 73728))
			blocks = ((rows + 255) // 256) * ((tokens + 127) // 128)
			def launch(function=function, blocks=blocks, pointers=pointers):
				stream = ct.c_void_p(torch.cuda.current_stream().cuda_stream)
				checked(driver.cuLaunchKernel(function, blocks, 1, 1, 256, 1, 1, 73728, stream, pointers, None))
			output.fill_(123)
			launch()
			torch.cuda.synchronize()
			assert failed.item() == 0
			torch.testing.assert_close(output[:tokens * rows], expected, rtol=2e-5, atol=2e-5)
			assert (output[tokens * rows:] == 123).all(), "GEMM tail overwrite"
			if os.getenv("STRICT_BENCH_WHOLE_GUARD") and tokens == 129:
				packed_weights.view(torch.int64)[0] = (rows << 32) | stride
				failed.zero_()
				launch()
				torch.cuda.synchronize()
				expected_failure = int(name.endswith("_fast"))
				assert failed.item() == expected_failure, (directory, "whole GEMM layout guard")
				assert (output[tokens * rows:] == 123).all(), "GEMM rejection tail overwrite"
				packed_weights.view(torch.int64)[0] = (rows << 32) | (1 << 31) | stride
				failed.zero_()
			calls.append((directory, launch, arguments, pointers))
		report(calls, f"GEMM {tokens}x{width}x{rows} result={result_bits}")
for _, module, _ in loaded:
	checked(driver.cuModuleUnload(module))
