"""Prepare shared prompts/noise and run Strict Moss speed/quality comparisons.
Use ComfyUI's python_embeded. Images, models and logs stay outside the repository.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
sys.dont_write_bytecode = True

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path("D:/StrictDumps/moss-speed-comparison"))
parser.add_argument("--prepare", action="store_true")
parser.add_argument("--download", action="store_true")
parser.add_argument("--report", action="store_true")
parser.add_argument("--merge", action="store_true")
parser.add_argument("--run", action="store_true")
parser.add_argument("--cases", nargs="*")
parser.add_argument("--variants", nargs="*")
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--resident-runs", type=int, default=1, help="generations per loaded model/decoder process")
parser.add_argument("--compressed", action="store_true", help="compress PNGs (requires --format png)")
parser.add_argument("--format", choices=["jpg", "png"], default="jpg" if os.name == "nt" else "png", help="output format (default JPEG on Windows)")
parser.add_argument("--jpg_quality", type=int, default=85, help="JPEG compression quality, 0-100 (default 85)")
options = parser.parse_args()
if not 0 <= options.jpg_quality <= 100:
	parser.error("JPEG quality must be between 0 and 100")
if options.compressed and options.format != "png":
	parser.error("--compressed requires --format png")
if options.prepare and (options.run or options.merge):
	parser.error("Run --prepare separately so the text encoder releases GPU memory before inference or merging")
root = options.root.resolve()
root.mkdir(parents=True, exist_ok=True)
repository = Path(__file__).resolve().parent
sys.path.insert(0, str(repository))
manifest = root / "comparison.json"
if not manifest.exists():
	models = "D:/ComfyUI_windows_portable/ComfyUI/models"
	base = models + "/diffusion_models/moss-image-sft-dit-int8-convrot.safetensors"
	manifest.write_text(json.dumps({
		"width": 720, "height": 1280, "seeds": [42, 1234],
		"prompts": {
			"lighthouse": "A weathered lighthouse on a rocky cliff at sunset, powerful waves below, warm golden light, detailed realistic photograph",
			"portrait": "Portrait photograph of an adult woman in a green wool sweater holding a ceramic mug with both hands, cozy cafe, soft window light, natural skin texture",
			"menu": "A straight-on photograph of a cafe chalkboard sign reading TODAY'S SPECIAL, Flat White $4.50, Blueberry Muffin $3.00, neat readable lettering, a small drawing of a coffee cup, warm wooden frame"
		},
		"vae": models + "/vae/flux2-vae.safetensors",
		"variants": {
			"base25": {"model": base, "steps": 25, "guidance": 3},
			"spectrum25": {"model": base, "steps": 25, "guidance": 3, "mode": "on", "warmup": 8, "period": 3},
			"turbo8": {"model": base, "lora": models + "/loras/moss/dmd_turbo_lora.safetensors", "steps": 8, "guidance": 1},
			"spectrum8": {"model": base, "lora": models + "/loras/moss/dmd_turbo_lora.safetensors", "steps": 8, "guidance": 1, "mode": "on", "warmup": 3, "period": 2},
			"akhaliq4": {"model": str(root / "merged/akhaliq-int8.safetensors"), "steps": 4, "guidance": 1},
			"rapid4": {"model": str(root / "merged/darask0-int8.safetensors"), "adapter": str(root / "darask0"), "steps": 4, "guidance": 1},
			"akhaliq4-unmerged": {"model": base, "lora": str(root / "akhaliq/lora.safetensors"), "steps": 4, "guidance": 1, "optional": True},
			"rapid4-unmerged": {"model": base, "lora": str(root / "darask0/lora.safetensors"), "adapter": str(root / "darask0"), "steps": 4, "guidance": 1, "optional": True}
		}
	}, indent=2), encoding="utf-8")
settings = json.loads(manifest.read_text(encoding="utf-8"))
if options.variants and set(options.variants) - set(settings["variants"]):
	parser.error("Unknown variant: " + ", ".join(sorted(set(options.variants) - set(settings["variants"]))))
assert options.repeat > 0 and options.resident_runs > 0, "repeat counts must be positive"
assert settings["width"] > 0 and settings["width"] % 16 == 0, "width must be a positive multiple of 16"
assert settings["height"] > 0 and settings["height"] % 16 == 0, "height must be a positive multiple of 16"
if options.download:
	from huggingface_hub import hf_hub_download
	adapters = [("darask0", "darask0/MOSS-Image_4-step_adapter_lora", "c88f33bcb3251eb39943d0635e74b650732977c8", ["lora.safetensors", "time_embedding.safetensors", "dataspace.safetensors"]),
		("akhaliq", "akhaliq/MOSS-Image-SFT-4-steps", "2ded95c37a029360245926fe72e475e30039bbfb", ["lora.safetensors"])]
	for name, repository_id, revision, files in adapters:
		for filename in files:
			hf_hub_download(repository_id, filename, revision=revision, local_dir=str(root / name))
			print("downloaded", name, filename, flush=True)
if options.prepare:
	# Reuse dump.py's ComfyUI initialization and text-encoder configuration.
	saved_arguments = sys.argv
	sys.argv = ["dump.py", "moss"]
	import dump
	sys.argv = saved_arguments
	torch = dump.torch
	from safetensors.torch import save_file
	clip = dump.comfy.sd.load_clip([dump.folder_paths.get_full_path_or_raise("text_encoders", dump.options.te)],
		embedding_directory=dump.folder_paths.get_folder_paths("embeddings"), clip_type=dump.comfy.sd.CLIPType.MOSS_IMAGE)
	for name, prompt in settings["prompts"].items():
		started = time.perf_counter()
		context = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))[0][0].detach().cpu().float().contiguous()
		for seed in settings["seeds"]:
			folder = root / (name + "-" + str(seed))
			folder.mkdir(exist_ok=True)
			noise = torch.randn((1, 128, settings["height"] // 16, settings["width"] // 16), generator=torch.Generator("cpu").manual_seed(seed))
			save_file({"context": context, "noise": noise}, str(folder / "input.safetensors"), metadata={"prompt": prompt, "seed": str(seed)})
		print("prepared", name, "tokens", context.shape[1], "seconds", round(time.perf_counter() - started, 2), flush=True)
	# Generation runs in separate processes after this encoder process exits.
if options.merge:
	(root / "merged").mkdir(exist_ok=True)
	base = settings["variants"]["base25"]["model"]
	for name in ["akhaliq", "darask0"]:
		with (root / ("merge-" + name + ".log")).open("w", encoding="utf-8") as log:
			subprocess.run([sys.executable, str(repository / "merge.py"), base, str(root / "merged" / (name + "-int8.safetensors")), str(root / name / "lora.safetensors")], cwd=repository, stdout=log, stderr=subprocess.STDOUT, timeout=150, check=True)
		print("merged", name, flush=True)
if options.run:
	cache = root / "cuda-cache"
	cache.mkdir(exist_ok=True)
	from safetensors.torch import load_file, save_file
	import torch
	for name in settings["prompts"]:
		for seed in settings["seeds"]:
			case = name + "-" + str(seed)
			if options.cases and case not in options.cases:
				continue
			inputs = load_file(str(root / case / "input.safetensors"))
			for label, variant in settings["variants"].items():
				if not options.variants and variant.get("optional", False):
					continue
				if options.variants and label not in options.variants:
					continue
				folder = root / case / label
				folder.mkdir(exist_ok=True)
				configuration = dict(inputs)
				for key, default in [("steps", 8), ("guidance", 1), ("warmup", 8), ("period", 2)]:
					configuration[key] = torch.tensor([variant.get(key, default)], dtype=torch.float32)
				configuration["runs"] = torch.tensor([options.resident_runs], dtype=torch.float32)
				configuration["compressed"] = torch.tensor([int(options.compressed)], dtype=torch.float32)
				configuration["jpeg"] = torch.tensor([float(options.format == "jpg")], dtype=torch.float32)
				configuration["quality"] = torch.tensor([options.jpg_quality], dtype=torch.float32)
				save_file(configuration, str(folder / "generation.safetensors"))
				mode = variant.get("mode", "off")
				command = [str(repository / "MossGenerate.exe"), str(folder), variant["model"], variant.get("lora", ""), settings["vae"], mode, variant.get("adapter", "")]
				for run in range(options.repeat):
					started = time.perf_counter()
					with (folder / ("run-" + str(run) + ".log")).open("w", encoding="utf-8") as log:
						result = subprocess.run(command, cwd=repository, env=dict(os.environ, STRICT_GPU="all", CUDA_CACHE_PATH=str(cache), CUDA_CACHE_MAXSIZE="4294967296"), stdout=log, stderr=subprocess.STDOUT, timeout=240)
					text = (folder / ("run-" + str(run) + ".log")).read_text(encoding="utf-8")
					assert result.returncode == 0, (case, label, result.returncode, text[-2000:])
					suffix = "-" + str(options.resident_runs - 1) if options.resident_runs > 1 else ""
					image_name = "strict-" + mode + suffix + "." + options.format
					assert (folder / image_name).exists(), (case, label, text[-1000:])
					sampling_runs = [int(value) for value in re.findall(r"sampling (\d+) ms", text)]
					generation_runs = [int(value) for value in re.findall(r"generation \d+ in (\d+) ms", text)]
					assert len(sampling_runs) == options.resident_runs, text
					setup = re.findall(r"text setup at ([0-9.]+)", text)
					first = re.findall(r"(?:step|evaluation) 0 at ([0-9.]+)", text)
					denoising = sampling_runs[-1] - 1000 * (float(first[-1]) - float(setup[-1])) if setup and first else None
					measurement = {"denoising_ms": round(denoising, 1) if denoising is not None else None, "case": case, "variant": label, "run": run, "sampling_ms": sampling_runs[-1], "sampling_runs_ms": sampling_runs, "generation_runs_ms": generation_runs, "resident_runs": options.resident_runs, "compressed_png": options.compressed, "image_name": image_name, "wall_seconds": time.perf_counter() - started, "command": command, "configuration": variant}
					measurement["output_format"] = options.format
					measurement["jpeg_quality"] = options.jpg_quality if options.format == "jpg" else None
					(folder / ("run-" + str(run) + ".json")).write_text(json.dumps(measurement, indent=2), encoding="utf-8")
					print(json.dumps(measurement), flush=True)
if options.report:
	import html
	import statistics
	labels = [label for label, variant in settings["variants"].items() if not variant.get("optional", False)]
	measurements = {label: [] for label in labels}
	pages = ["<!doctype html><meta charset='utf-8'><title>Moss speed and quality comparison</title>",
		"<style>body{background:#171b21;color:#eee;font:16px system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:8px;text-align:left;vertical-align:top;border:1px solid #444}img{width:200px;height:auto}a{color:#aad4ff}p{max-width:1100px}</style>",
		"<h1>Moss: sampling speed and visible quality</h1><p>" + str(settings["width"]) + " x " + str(settings["height"]) + "; shared prompt and CPU-generated noise within each row. Click an image for full resolution. All speed presets use int8 Convrot weights and bf16 residual/QK. Base25 uses Euler/shift 5, ordinary CFG 3 with zero conditioning; no negative block skip or ER-SDE. Four-step models are separately trained adapters, not shortened base sampling. Spectrum uses fixed periodic forecasts, not upstream adaptive scheduling.</p>",
		"<p>Sampling includes text projection/setup and initial weight uploads, but excludes model loading and decode/save. Denoising starts at step 0. CUDA JIT cache is warmed; first-use compilation can add tens of seconds. Each image is a quality tradeoff, not an equivalence check.</p>"]
	for name, prompt in settings["prompts"].items():
		pages += ["<h2>" + html.escape(name) + "</h2><p>" + html.escape(prompt) + "</p><table><tr><th>Seed</th>" + "".join("<th>" + html.escape(label) + "</th>" for label in labels) + "</tr>"]
		for seed in settings["seeds"]:
			case = name + "-" + str(seed)
			pages.append("<tr><th>" + str(seed) + "</th>")
			for label in labels:
				folder = root / case / label
				files = sorted(folder.glob("run-*.json"), key=lambda path: path.stat().st_mtime_ns)
				data = json.loads(files[-1].read_text(encoding="utf-8")) if files else {}
				image = folder / data.get("image_name", "strict-" + settings["variants"][label].get("mode", "off") + ".png")
				if not files or not image.exists():
					pages.append("<td>Pending</td>"); continue
				measurements[label].append(data)
				link = image.relative_to(root).as_posix()
				last_generation = "<br>Last generation " + str(round(data["generation_runs_ms"][-1] / 1000, 2)) + " s" if data.get("generation_runs_ms") else ""
				pages.append("<td><a href='" + link + "'><img loading='lazy' src='" + link + "'></a><br>Sampling " + str(round(data["sampling_ms"] / 1000, 2)) + " s" + last_generation + "<br>Process (" + str(data.get("resident_runs", 1)) + " images) " + str(round(data["wall_seconds"], 2)) + " s</td>")
			pages.append("</tr>")
		pages.append("</table>")
	summary = {}
	for label, rows in measurements.items():
		if rows:
			summary[label] = {"images": len(rows), "sampling_median_ms": statistics.median(row["sampling_ms"] for row in rows), "sampling_range_ms": [min(row["sampling_ms"] for row in rows), max(row["sampling_ms"] for row in rows)], "process_median_s": statistics.median(row["wall_seconds"] for row in rows)}
			generations = [row["generation_runs_ms"][-1] for row in rows if row.get("generation_runs_ms")]
			if generations:
				summary[label]["generation_last_median_ms"] = statistics.median(generations)
				summary[label]["resident_runs"] = sorted(set(row.get("resident_runs", 1) for row in rows))
	(root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
	(root / "index.html").write_text("\n".join(pages), encoding="utf-8")
	print(json.dumps(summary, indent=2))
	print("gallery", root / "index.html")
