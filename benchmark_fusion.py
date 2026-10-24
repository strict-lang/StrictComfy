"""Measure fusions before building them: benchmark_fusion.py <folder with MossImage.kernels.mlir> [tokens].

Use ComfyUI's python_embeded. Appends diagnostic kernels to a copy of the build's device module (in <folder>/fusion):
B1 computes Activated's situ per number inside strict_device_hadamard_tokens (its index by shift and mask, an upper bound),
B3 computes Gated's residual inside Modulated (row sum of squares and numbers, the residual written once). Both check
bit-identity against the generated kernels they replace and time CUDA graphs of 20 launches. Never used in production.
"""
import ctypes as ct, re, statistics, subprocess, sys
from pathlib import Path
import torch

folder = Path(sys.argv[1])
tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 3631
work = folder / "fusion"
work.mkdir(exist_ok=True)
source = (folder / "MossImage.kernels.mlir").read_text(encoding="utf-8")


def text_of(header):
	start = source.index(header)
	return source[start:source.index("\n  }\n", start) + 4]


def kernel_named(pattern):
	return re.search(r"  llvm.func @(" + pattern + r")\(([^\n]+)\) attributes([^\n]+)\{(.*?)\n  \}\n", source, re.S)


def hadamard_with_situ():
	run = text_of("  func.func private @strict_device_hadamard_tokens_run(")
	run = run.replace("@strict_device_hadamard_tokens_run(%source: !llvm.ptr, %count: i64, %width: i64, %target: !llvm.ptr, %bits: i64)",
		"@fusion_hadamard_run(%source: !llvm.ptr, %count: i64, %width: i64, %target: !llvm.ptr, %bits: i64, %failed: !llvm.ptr)")
	run, reads = re.subn(r"func\.call @strict_device_floats\(%source, %bits, (%at\d?)\) : \(!llvm\.ptr, i64, i64\) -> f32",
		r"func.call @fusion_situ(%failed, %source, %width, \1) : (!llvm.ptr, !llvm.ptr, i64, i64) -> f32", run)
	assert reads == 8
	element = """  func.func private @fusion_situ(%failed: !llvm.ptr, %expanded: !llvm.ptr, %width: i64, %at: i64) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %none = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %zero = arith.constant 0.0 : f32
    %shift = arith.constant 13 : i64
    %mask = arith.constant 8191 : i64
    %row = arith.shrui %at, %shift : i64
    %column = arith.andi %at, %mask : i64
    %doubled = arith.shli %row, %one : i64
    %rowStart = arith.shli %doubled, %shift : i64
    %first = arith.addi %rowStart, %column : i64
    %second = arith.addi %first, %width : i64
    %hidden = func.call @strict_device_bfloat16_read(%expanded, %first) : (!llvm.ptr, i64) -> f32
    %gate = func.call @strict_device_bfloat16_read(%expanded, %second) : (!llvm.ptr, i64) -> f32
    %situ = func.call @MossBlock_Situ_2_devices(%failed, %none, %one, %zero, %none, %one, %hidden, %none, %none, %gate, %none, %none) : (!llvm.ptr, i64, i64, f32, i64, i64, f32, i64, i64, f32, i64, i64) -> !llvm.struct<(f32, i64, i64)>
    %value = llvm.extractvalue %situ[0] : !llvm.struct<(f32, i64, i64)>
    %rounded = func.call @strict_device_bfloat16(%value) : (f32) -> bf16
    %back = arith.extf %rounded : bf16 to f32
    return %back : f32
  }
"""
	kernel = """  llvm.func @fusion_activated_hadamard(%source: !llvm.ptr, %count: i64, %width: i64, %target: !llvm.ptr, %bits: i64, %failed: !llvm.ptr) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 1024, 1, 1>} {
    func.call @fusion_hadamard_run(%source, %count, %width, %target, %bits, %failed) : (!llvm.ptr, i64, i64, !llvm.ptr, i64, !llvm.ptr) -> ()
    llvm.return
  }
"""
	return element + run + kernel


def modulated_with_gated(name):
	head = text_of("  func.func private @Qwen3Block_HeadScale_4_devices(")
	head = head.replace("@Qwen3Block_HeadScale_4_devices(", "@fusion_head_scale(", 1)
	head = head.replace(") -> !llvm.struct<(f32, i64, i64)> {", ", %g_update_d: !llvm.ptr, %g_modulation_d: !llvm.ptr, %g_part: i64) -> !llvm.struct<(f32, i64, i64)> {", 1)
	head, calls = re.subn(r"func\.call @strict_device_squares\(([^)]*)\) : \(([^)]*)\) -> f32",
		r"func.call @fusion_squares(\1, %g_update_d, %g_modulation_d, %g_part) : (\2, !llvm.ptr, !llvm.ptr, i64) -> f32", head)
	assert calls == 1
	kernel = kernel_named(name).group(0).replace("llvm.func @" + name + "(", "llvm.func @fusion_gated_modulated(", 1)
	kernel = kernel.replace(") attributes", ", %g_update_d: !llvm.ptr, %g_part: i64, %residual: !llvm.ptr) attributes", 1)
	kernel, calls = re.subn(r"func\.call @Qwen3Block_HeadScale_4_devices\(([^)]*)\) : \(([^)]*)\) -> !llvm\.struct<\(f32, i64, i64\)>",
		r"func.call @fusion_head_scale(\1, %g_update_d, %f_modulation_d, %g_part) : (\2, !llvm.ptr, !llvm.ptr, i64) -> !llvm.struct<(f32, i64, i64)>", kernel)
	assert calls == 1
	register = re.search(r"%(d\d+)d = llvm\.load %v_input_d", kernel).group(1)
	read = f"    %{register} = func.call @strict_device_floats(%{register}d, %{register}b, %{register}k) : (!llvm.ptr, i64, i64) -> f32\n"
	assert kernel.count(read) == 1
	kernel = kernel.replace(read, f"""    %{register}channel = arith.subi %{register}k, %groupStart : i64
    %{register} = func.call @fusion_gated(%{register}d, %g_update_d, %f_modulation_d, %g_part, %{register}channel, %{register}k) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> f32
    %{register}residual = llvm.getelementptr %residual[%{register}k] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %{register}, %{register}residual : f32, !llvm.ptr
""")
	added = """  func.func private @fusion_gated(%input: !llvm.ptr, %update: !llvm.ptr, %modulation: !llvm.ptr, %part: i64, %channel: i64, %at: i64) -> f32 attributes {llvm.passthrough = ["alwaysinline"]} {
    %inputSlot = llvm.getelementptr %input[%at] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %residual = llvm.load %inputSlot : !llvm.ptr -> f32
    %change = func.call @strict_device_bfloat16_read(%update, %at) : (!llvm.ptr, i64) -> f32
    %gateAt = arith.addi %part, %channel : i64
    %gateSlot = llvm.getelementptr %modulation[%gateAt] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %gate = llvm.load %gateSlot : !llvm.ptr -> f32
    %product = arith.mulf %change, %gate : f32
    %gated = arith.addf %residual, %product : f32
    return %gated : f32
  }
  func.func private @fusion_squares(%failed: !llvm.ptr, %data: !llvm.ptr, %bits: i64, %dataCount: i64, %count: i64, %start: i64, %lane: i64, %lanes: i64, %update: !llvm.ptr, %modulation: !llvm.ptr, %part: i64) -> f32 {
    %zero = arith.constant 0.0 : f32
    %two = arith.constant 2 : i64
    %fourFirst = arith.shli %lane, %two : i64
    %fourStep = arith.shli %lanes, %two : i64
    cf.br ^fours(%fourFirst, %zero : i64, f32)
  ^fours(%fourIndex: i64, %fourSum: f32):
    %fourMore = arith.cmpi slt, %fourIndex, %count : i64
    cf.cond_br %fourMore, ^fourBody, ^reduce(%fourSum : f32)
  ^fourBody:
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %fourAt = arith.addi %start, %fourIndex : i64
    %slot = llvm.getelementptr %data[%fourAt] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %inputs = llvm.load %slot {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %updateSlot = llvm.getelementptr %update[%fourAt] : (!llvm.ptr, i64) -> !llvm.ptr, bf16
    %updatesHalf = llvm.load %updateSlot {alignment = 8 : i64} : !llvm.ptr -> vector<4xbf16>
    %updates = arith.extf %updatesHalf : vector<4xbf16> to vector<4xf32>
    %gateAt = arith.addi %part, %fourIndex : i64
    %gateSlot = llvm.getelementptr %modulation[%gateAt] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %gates = llvm.load %gateSlot {alignment = 16 : i64} : !llvm.ptr -> vector<4xf32>
    %products = arith.mulf %updates, %gates : vector<4xf32>
    %four = arith.addf %inputs, %products : vector<4xf32>
    %squared = arith.mulf %four, %four fastmath<contract> : vector<4xf32>
    %s0 = llvm.extractelement %squared[%c0 : i32] : vector<4xf32>
    %s1 = llvm.extractelement %squared[%c1 : i32] : vector<4xf32>
    %s2 = llvm.extractelement %squared[%c2 : i32] : vector<4xf32>
    %s3 = llvm.extractelement %squared[%c3 : i32] : vector<4xf32>
    %added0 = arith.addf %fourSum, %s0 fastmath<contract> : f32
    %added1 = arith.addf %added0, %s1 fastmath<contract> : f32
    %added2 = arith.addf %added1, %s2 fastmath<contract> : f32
    %added3 = arith.addf %added2, %s3 fastmath<contract> : f32
    %fourFollowing = arith.addi %fourIndex, %fourStep : i64
    cf.br ^fours(%fourFollowing, %added3 : i64, f32)
  ^reduce(%total: f32):
    %reduced = func.call @strict_device_lanes_sums(%total, %lane, %lanes) : (f32, i64, i64) -> f32
    return %reduced : f32
  }
"""
	return added + head + kernel


activated = kernel_named(r"MossBlock_Activated_\d+_gpu\d+sxf1").group(1)
gated = kernel_named(r"MossBlock_Gated_\d+_gpu\d+sx").group(1)
modulated = kernel_named(r"MossBlock_Modulated_\d+_gpu\d+sxf0").group(1)
closing = source.rstrip().rfind("}")
(work / "fusion.kernels.mlir").write_text(source[:closing] + hadamard_with_situ() + modulated_with_gated(modulated) + "}\n", encoding="utf-8")
tools = "C:/msys64/ucrt64/bin/"
base = str(work / "fusion")
print("compiling", flush=True)
subprocess.run([tools + "mlir-opt", base + ".kernels.mlir", "--symbol-dce", "--convert-scf-to-cf", "--convert-arith-to-llvm", "--convert-func-to-llvm",
	"--convert-cf-to-llvm", "--reconcile-unrealized-casts", "-o", base + ".llvm.mlir"], check=True)
subprocess.run([tools + "mlir-translate", "--mlir-disable-threading", "--mlir-to-llvmir", base + ".llvm.mlir", "-o", base + ".gpu.ll"], check=True)
subprocess.run(["C:/Program Files/LLVM/bin/clang.exe", "--target=nvptx64-nvidia-cuda", "-O3", "-fno-slp-vectorize", "-S", "-Wno-override-module",
	"-Xclang", "-target-feature", "-Xclang", "+ptx78", "-march=sm_80", base + ".gpu.ll", "-o", base + ".ptx"], check=True)
compiled = (work / "fusion.kernels.mlir").read_text(encoding="utf-8")
torch.manual_seed(42)
torch.cuda.init()
context = torch.empty(1, device="cuda")
driver = ct.WinDLL("nvcuda.dll")
driver.cuModuleLoadData.argtypes = [ct.POINTER(ct.c_void_p), ct.c_char_p]
driver.cuModuleGetFunction.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p]
driver.cuLaunchKernel.argtypes = [ct.c_void_p] + [ct.c_uint] * 7 + [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_void_p]


def checked(status):
	assert status == 0, f"CUDA driver error {status}"


def magic(divisor):
	bits = (divisor - 1).bit_length()
	first = min(bits, 1)
	return (((1 << bits) - divisor) * (1 << 32) // divisor + 1) | (first << 32) | ((bits - first) << 40)


module = ct.c_void_p()
print("loading PTX (the first load compiles it for this GPU, minutes)", flush=True)
checked(driver.cuModuleLoadData(ct.byref(module), (work / "fusion.ptx").read_bytes()))
failed = torch.zeros(1, device="cuda", dtype=torch.int64)


def kernel(name, values, blocks, threads):
	handle = ct.c_void_p()
	checked(driver.cuModuleGetFunction(ct.byref(handle), module, name.encode()))
	signature, body = re.search(r"llvm.func @" + name + r"\(([^\n]+)\) attributes[^\n]+\{(.*?)\n  \}", compiled, re.S).groups()
	arguments = [ct.c_double(values[key]) if kind == "f64" else ct.c_uint64(values[key]) for key, kind in re.findall(r"%(\w+): ([^,]+)", signature)]
	pointers = (ct.c_void_p * len(arguments))(*[ct.cast(ct.byref(value), ct.c_void_p) for value in arguments])
	if blocks is None:
		total = (threads + 3) // 4 if "%batchSize = arith.constant 4" in body else threads
		blocks, threads = (total + 255) // 256, 256
	def launch(handle=handle, blocks=blocks, threads=threads, pointers=pointers, arguments=arguments):
		checked(driver.cuLaunchKernel(handle, blocks, 1, 1, threads, 1, 1, 0, ct.c_void_p(torch.cuda.current_stream().cuda_stream), pointers, None))
	return launch


def numbers(name, value):
	return {"f_" + name: value, "e_" + name: value, "m_" + name: magic(value) if value > 0 else 0}


def listed(name, tensor, bits, count=None):
	return {"f_" + name + "_d": tensor.data_ptr(), "f_" + name + "_n": count or tensor.numel(), "f_" + name + "_b": bits}


def compare(label, separate, together, outputs):
	separate()
	together()
	torch.cuda.synchronize()
	assert failed.item() == 0, label + ": a kernel read outside of a list"
	for name, first, second in outputs:
		print(f"{label} {name}: {(first != second).sum().item()} of {first.numel()} differ", flush=True)
	graphs = {}
	for name, launch in (("separate", separate), ("fused", together)):
		graph = torch.cuda.CUDAGraph(keep_graph=True)
		with torch.cuda.graph(graph):
			for iteration in range(20):
				launch()
		graph.replay()
		graphs[name] = graph
	torch.cuda.synchronize()
	times = {name: [] for name in graphs}
	for repeat in range(8):
		for name, graph in (list(graphs.items()) if repeat % 2 == 0 else list(reversed(graphs.items()))):
			before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
			before.record()
			graph.replay()
			after.record()
			after.synchronize()
			times[name].append(before.elapsed_time(after) * 1000 / 20)
	separate_us, fused_us = statistics.median(times["separate"]), statistics.median(times["fused"])
	print(f"{label}: separate {separate_us:.1f} us, fused {fused_us:.1f} us, {(separate_us - fused_us) * 42 / 1000:.2f} ms per step per pair of each layer", flush=True)


common = {"failed": failed.data_ptr(), "source": 0, "columns": 0, "bits": 64}
width = 8192
count = tokens * width
expanded = torch.randn(tokens, 2, width, device="cuda").bfloat16()
situ = torch.empty(count + 32, device="cuda", dtype=torch.bfloat16)
target = count + count // 32 * 4 + 256
reference, fused = torch.zeros(target, device="cuda", dtype=torch.uint8), torch.zeros(target, device="cuda", dtype=torch.uint8)
activate = kernel(activated, dict(common, count=count, result=situ.data_ptr(), results=17, **numbers("width", width), **listed("expanded", expanded, 17)), None, count)
hadamard = kernel("strict_device_hadamard_tokens", {"source": situ.data_ptr(), "count": count, "width": width, "target": reference.data_ptr(), "bits": 17}, count // width, width // 8)
rotate = kernel("fusion_activated_hadamard", {"source": expanded.data_ptr(), "count": count, "width": width, "target": fused.data_ptr(), "bits": 17, "failed": failed.data_ptr()}, count // width, width // 8)
compare("B1 Activated into Hadamard", lambda: (activate(), hadamard()), rotate, [("int8 numbers and scales", reference, fused)])
width = 3072
count = tokens * width
residual = torch.randn(count, device="cuda") * 4
update = torch.randn(count, device="cuda").bfloat16()
modulation = torch.randn(6 * width, device="cuda") * 0.3
weights = 1 + torch.randn(width, device="cuda") * 0.1
attended, normed = torch.empty(count + 32, device="cuda"), torch.empty(count + 32, device="cuda", dtype=torch.bfloat16)
fused_residual, fused_normed = torch.empty(count + 32, device="cuda"), torch.empty(count + 32, device="cuda", dtype=torch.bfloat16)
gate = kernel(gated, dict(common, count=count, result=attended.data_ptr(), results=32, **numbers("part", 2), **numbers("width", width),
	**listed("input", residual, 32), **listed("update", update, 17), **listed("modulation", modulation, 32)), None, count)
modulated_values = dict(common, count=count, result=normed.data_ptr(), results=17, **numbers("part", 3), **numbers("width", width),
	**listed("input", attended, 32, count), **listed("weights", weights, 32), **listed("modulation", modulation, 32))
modulate = kernel(modulated, modulated_values, tokens, 256)
together = kernel("fusion_gated_modulated", dict(modulated_values, result=fused_normed.data_ptr(), f_input_d=residual.data_ptr(), g_update_d=update.data_ptr(),
	g_part=2 * width, residual=fused_residual.data_ptr()), tokens, 256)
compare("B3 Gated into Modulated", lambda: (gate(), modulate()), together,
	[("residual", attended[:count].view(torch.int32), fused_residual[:count].view(torch.int32)), ("normed", normed[:count].view(torch.int16), fused_normed[:count].view(torch.int16))])
