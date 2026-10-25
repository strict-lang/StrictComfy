"""User-facing Moss generation; ComfyUI encodes text, Strict samples and decodes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def lora_spec(text):
	path, separator, suffix = text.rpartition(':')
	if not separator or (len(path) == 1 and path.isalpha() and suffix.startswith(('\\', '/'))):
		return text, 1.0
	try:
		strength = float(suffix)
	except ValueError:
		raise ValueError('Invalid LoRA strength: ' + text) from None
	if not path or not math.isfinite(strength):
		raise ValueError('LoRA path and finite strength required: ' + text)
	return path, strength


def expand_prompt(text, rng):
	groups = [['']]
	for character in text:
		if character == '{':
			groups.append([''])
		elif character == '}':
			if len(groups) == 1:
				raise ValueError('Unmatched } in prompt')
			groups[-2][-1] += rng.choice(groups.pop())
		elif character == '|' and len(groups) > 1:
			groups[-1].append('')
		else:
			groups[-1][-1] += character
	if len(groups) != 1:
		raise ValueError('Unclosed { in prompt')
	return groups[0][0]


def reserve_output(base):
	base = Path(base)
	base.parent.mkdir(parents=True, exist_ok=True)
	pattern = re.compile(re.escape(base.stem) + r'(\d{4,})' + re.escape(base.suffix), re.IGNORECASE)
	numbers = [int(match[1]) for path in base.parent.iterdir() if (match := pattern.fullmatch(path.name))]
	number = max(numbers, default=0) + 1
	while True:
		path = base.with_name(f'{base.stem}{number:04d}{base.suffix}')
		if not path.with_suffix(path.suffix + '.json').exists():
			try:
				return path, path.open('xb')
			except FileExistsError:
				pass
		number += 1


def arguments(argv=None):
	default_comfy = next((Path(drive + ':/ComfyUI_windows_portable/ComfyUI') for drive in ('D', 'C')
		if Path(drive + ':/ComfyUI_windows_portable/ComfyUI').is_dir()), Path('ComfyUI'))
	default_comfy = Path(os.environ.get('COMFYUI', default_comfy))
	files = [('model', 'diffusion_models', 'moss-image-sft-dit-int8-convrot.safetensors', 'Base int8 Convrot model'),
		('vae', 'vae', 'flux2-vae.safetensors', 'Image VAE'),
		('te', 'text_encoders', 'qwen3.5_9b_int8_convrot.safetensors', 'Text encoder')]
	parser = argparse.ArgumentParser(prog='MossImage.exe', description=__doc__, add_help=False,
		allow_abbrev=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--help', action='help', help='show this help message and exit')
	for name, folder, filename, description in files:
		parser.add_argument('--' + name, default=argparse.SUPPRESS,
			help=f'{description} (default: {default_comfy / "models" / folder / filename})')
	parser.add_argument('--prompt', type=Path, default=Path('prompt.txt'), help='UTF-8 prompt file; supports {a|b|}')
	parser.add_argument('--steps', type=int, default=8, help='Diffusion sampling steps; 8 automatically applies the DMD turbo LoRA')
	parser.add_argument('--count', type=int, default=1, help='images per loaded model; fresh seed/prompt choices each time')
	parser.add_argument('--lora', action='append', default=[], help='repeat for each path[:strength], default strength 1')
	parser.add_argument('--lora_mode', choices=['merged', 'runtime'], default='merged',
		help='merged: cached INT8 stochastic merge (fast, approximate); runtime: floating adapters')
	parser.add_argument('--merge_only', action='store_true', help='prepare/print the cached merged model without encoding or generating')
	parser.add_argument('--spectrum', choices=['off', 'spectrum8_w3p3'], default='off',
		help='optional 3-forecast eight-step predictor; changes the sampling result')
	parser.add_argument('--output', type=Path, default=Path('MossImage.jpg'), help='numbered output base; never overwritten')
	parser.add_argument('--json_output', action='store_true', help='write a per-image JSON settings sidecar')
	parser.add_argument('--width', type=int, default=720, help='Image width in pixels')
	parser.add_argument('--height', type=int, default=1280, help='Image height in pixels')
	parser.add_argument('--seed', type=int, default=None, help='random by default; fixes prompt choices and noise')
	parser.add_argument('--guidance', type=float, default=1, help='Classifier-free guidance (CFG) scale')
	parser.add_argument('--jpg_quality', type=int, default=85, help='JPEG compression quality, 0-100; does not affect inference')
	parser.add_argument('--comfyui', type=Path, default=default_comfy, help='Installed ComfyUI directory')
	parser.add_argument('--cache', type=Path, default=ROOT / 'artifacts' / 'moss', help='Prompt, merged-model, runtime-adapter and CUDA caches')
	parser.add_argument('--python', type=Path, help='Python interpreter for the native launcher; use ComfyUI Python')
	args = parser.parse_args(argv)
	for name, folder, filename, description in files:
		if not hasattr(args, name):
			setattr(args, name, str(args.comfyui / 'models' / folder / filename))
	if args.steps < 1 or args.count < 1 or min(args.width, args.height) < 16 or args.width % 16 or args.height % 16:
		parser.error('count/steps must be positive; width/height must be positive multiples of 16')
	if not 0 <= args.jpg_quality <= 100 or not math.isfinite(args.guidance):
		parser.error('jpg_quality must be 0-100 and guidance finite')
	if args.seed is not None and not 0 <= args.seed < 2**63:
		parser.error('seed must be between 0 and 2^63-1')
	if args.output.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
		parser.error('output extension must be .jpg, .jpeg or .png')
	if args.spectrum != 'off' and args.steps != 8:
		parser.error('spectrum8_w3p3 requires --steps 8')
	if args.merge_only and args.lora_mode != 'merged':
		parser.error('--merge_only requires --lora_mode merged')
	return args


def spectrum_config(args):
	return ('off', 5, 2) if args.spectrum == 'off' else ('on', 3, 3)


def jobs(args, template):
	rng = random.Random(args.seed) if args.seed is not None else random.SystemRandom()
	result, used = [], set()
	for index in range(args.count):
		seed = args.seed if index == 0 and args.seed is not None else rng.getrandbits(63)
		while seed in used:
			seed = rng.getrandbits(63)
		used.add(seed)
		result.append({'seed': seed, 'prompt': expand_prompt(template, random.Random(seed))})
	return result


def located(name, folder):
	path = Path(name).expanduser()
	for candidate in (path, folder / path):
		if candidate.is_file():
			return candidate.resolve()
	raise FileNotFoundError(f'File not found: {name} (also checked {folder})')


def baked_loras(metadata):
	value = (metadata or {}).get('strict_lora_merge')
	if not isinstance(value, str):
		return []
	marker = json.loads(value)
	if marker.get('algorithm') != 'convrot-int8-stochastic-v1':
		raise ValueError('Unsupported merged-model metadata; rebuild the cached merge')
	return marker.get('loras', [])


def selected_loras(args, models, metadata=None):
	selected = [(located(path, models / 'loras'), strength) for path, strength in map(lora_spec, args.lora)]
	has_turbo = any(Path(str(path).replace('\\', '/')).name == 'dmd_turbo_lora.safetensors' and strength != 0
		for path, strength in baked_loras(metadata))
	explicit_turbo = any(path.name == 'dmd_turbo_lora.safetensors' for path, strength in selected)
	if args.steps == 8 and not has_turbo and not explicit_turbo:
		turbo = located('moss/dmd_turbo_lora.safetensors', models / 'loras')
		selected.insert(0, (turbo, 1.0))
	return selected


def identity(path):
	stat = path.stat()
	return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def validate_model(path):
	from safetensors import safe_open
	with safe_open(path, framework='pt') as checkpoint:
		key = 'layers.0.proj.q.comfy_quant'
		if key not in checkpoint.keys():
			raise ValueError('--model must be an int8 Convrot Moss checkpoint')
		quant = json.loads(bytes(checkpoint.get_tensor(key).tolist()))
		if not quant.get('convrot') or quant.get('convrot_groupsize') != 256 or quant.get('format') != 'int8_tensorwise':
			raise ValueError('--model must use int8 Convrot with group size 256')
		return checkpoint.metadata() or {}


def cache_key(data):
	return hashlib.sha256(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()


def cached_merge(args, model, loras):
	active = [(path, strength) for path, strength in loras if strength != 0]
	if not active:
		return model
	script = ROOT / 'merge.py'
	version = hashlib.sha256(script.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
	key = cache_key(['convrot-int8-stochastic-v1', version, identity(model),
		[(identity(path), strength) for path, strength in active]])
	path = args.cache / ('merged-stochastic-v1-' + key + '.safetensors')
	if path.exists():
		from safetensors import safe_open
		with safe_open(path, framework='pt') as file:
			if not baked_loras(file.metadata()):
				raise ValueError('Invalid cached merge: ' + str(path))
		print('Using cached merged model: ' + str(path), flush=True)
		return path
	print('Preparing cached stochastic INT8 merge; source files stay unchanged', flush=True)
	with tempfile.TemporaryDirectory(dir=args.cache) as temporary:
		output = Path(temporary) / 'merged.safetensors'
		command = [sys.executable, str(script), str(model), str(output),
			*[str(source) + ':' + repr(strength) for source, strength in active], '--split-qkv']
		subprocess.run(command, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
			OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONUNBUFFERED='1'))
		os.replace(output, path)
	return path


def encoded(request, output):
	os.environ['COMFYUI'] = request['comfyui']
	sys.argv = ['dump.py', 'moss']
	import dump
	from safetensors.torch import save_file
	clip = dump.comfy.sd.load_clip([request['te']],
		embedding_directory=dump.folder_paths.get_folder_paths('embeddings'), clip_type=dump.comfy.sd.CLIPType.MOSS_IMAGE)
	for index, prompt in enumerate(request['prompts']):
		print(f'Encoding prompt {index + 1}/{len(request["prompts"])}', flush=True)
		context = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))[0][0].detach().cpu().float().contiguous()
		save_file({'context': context}, str(output / f'{index}.safetensors'), metadata={'prompt': prompt})


def conditioning(args, prompts, encoder):
	paths, missing = [], {}
	for prompt in prompts:
		request = {'prompt': prompt, 'te': str(encoder), 'comfyui': str(args.comfyui.resolve())}
		key = cache_key([1, request, identity(encoder), identity(args.comfyui / 'comfy' / 'sd.py')])
		path = args.cache / ('context-' + key + '.safetensors')
		paths.append(path)
		if not path.exists():
			missing[path] = prompt
	if missing:
		print(f'Preparing {len(missing)} uncached prompts with one text-encoder load', flush=True)
		with tempfile.TemporaryDirectory(dir=args.cache) as temporary:
			folder = Path(temporary)
			request = {'prompts': list(missing.values()), 'te': str(encoder), 'comfyui': str(args.comfyui.resolve())}
			(folder / 'request.json').write_text(json.dumps(request), encoding='utf-8')
			subprocess.run([sys.executable, str(ROOT / 'moss.py'), '_encode', str(folder / 'request.json'),
				str(folder)], check=True)
			for index, path in enumerate(missing):
				os.replace(folder / f'{index}.safetensors', path)
	else:
		print('Using cached prompt conditioning', flush=True)
	return paths


def lora_factors(file, prefix):
	for down_name, up_name in (('lora_down', 'lora_up'), ('lora_A', 'lora_B')):
		if prefix + down_name + '.weight' in file.keys():
			down = file.get_tensor(prefix + down_name + '.weight').float()
			up = file.get_tensor(prefix + up_name + '.weight').float()
			alpha = float(file.get_tensor(prefix + 'alpha')) if prefix + 'alpha' in file.keys() else down.shape[0]
			return down, up, alpha / down.shape[0]
	return None


def runtime_adapter(args, model, loras):
	from contextlib import ExitStack
	import torch
	from safetensors import safe_open
	from safetensors.torch import save_file
	active = [(path, strength) for path, strength in loras if strength != 0]
	if not active:
		return ''
	key = cache_key([1, identity(ROOT / 'moss.py'), identity(model), [(identity(path), strength) for path, strength in active]])
	path = args.cache / ('adapter-' + key + '.safetensors')
	if path.exists():
		print('Using cached runtime LoRA stack', flush=True)
		return str(path)
	with ExitStack() as stack:
		base = stack.enter_context(safe_open(model, framework='pt'))
		files = [stack.enter_context(safe_open(source, framework='pt')) for source, strength in active]
		names = sorted(key[:-7] for key in base.keys() if re.fullmatch(r'layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)\.weight', key))
		prefixes = {'diffusion_model.' + name + '.' for name in names}
		provided = [{key[:-len(ending)] for key in file.keys() for ending in ('lora_down.weight', 'lora_A.weight')
			if key.endswith(ending)} for file in files]
		if len(active) == 1 and active[0][1] == 1 and provided[0] == prefixes:
			if all(prefix + ending in files[0].keys() for prefix in prefixes
				for ending in ('lora_down.weight', 'lora_up.weight', 'alpha')):
				print('Applying original runtime LoRA: ' + str(active[0][0]), flush=True)
				return str(active[0][0])
		tensors, used = {}, [set() for file in files]
		for name in names:
			prefix = 'diffusion_model.' + name + '.'
			rows, columns = base.get_slice(name + '.weight').get_shape()
			downs, ups = [], []
			for index, (file, (source, strength)) in enumerate(zip(files, active)):
				own = prefix
				part = lora_factors(file, own)
				if part is None and name.rsplit('.', 1)[1] in ('q', 'k', 'v'):
					own = 'diffusion_model.' + name.rsplit('.', 1)[0] + '.qkv.'
					part = lora_factors(file, own)
					if part is not None:
						down, up, scale = part
						if up.shape[0] % 3:
							raise ValueError('Invalid fused qkv LoRA shape: ' + str(source))
						part = down, up.chunk(3, dim=0)['qkv'.index(name.rsplit('.', 1)[1])], scale
				if part is not None:
					down, up, scale = part
					if down.ndim != 2 or up.ndim != 2 or down.shape[1] != columns or tuple(up.shape) != (rows, down.shape[0]):
						raise ValueError('LoRA shape does not match ' + name + ': ' + str(source))
					downs.append(down)
					ups.append(up * (scale * strength))
					used[index].add(own)
			if not downs:
				downs, ups = [torch.zeros((1, columns))], [torch.zeros((rows, 1))]
			tensors[prefix + 'lora_down.weight'] = torch.cat(downs, dim=0)
			tensors[prefix + 'lora_up.weight'] = torch.cat(ups, dim=1)
			tensors[prefix + 'alpha'] = torch.tensor(float(tensors[prefix + 'lora_down.weight'].shape[0]))
		for index, (source, strength) in enumerate(active):
			if not used[index] or provided[index] - used[index]:
				raise ValueError('Unsupported LoRA targets or format (runtime supports Moss layer projections): ' + str(source))
		with tempfile.TemporaryDirectory(dir=args.cache) as temporary:
			output = Path(temporary) / 'adapter.safetensors'
			save_file(tensors, str(output), metadata={'runtime_loras': json.dumps([(str(source), strength) for source, strength in active])})
			os.replace(output, path)
	print('Prepared runtime LoRA stack; base checkpoint unchanged', flush=True)
	return str(path)


def generated(args, model, vae, contexts, plan, folder, adapter=''):
	import torch
	from safetensors.torch import load_file, save_file
	mode, warmup, period = spectrum_config(args)
	for index, (context, job) in enumerate(zip(contexts, plan)):
		data = load_file(str(context))
		data['noise'] = torch.randn((1, 128, args.height // 16, args.width // 16),
			generator=torch.Generator('cpu').manual_seed(job['seed']))
		for name, value in {'steps': args.steps, 'guidance': args.guidance, 'warmup': warmup, 'period': period,
			'jpeg': int(args.output.suffix.lower() != '.png'), 'quality': args.jpg_quality}.items():
			data[name] = torch.tensor([value], dtype=torch.float32)
		save_file(data, str(folder / f'generation-{index}.safetensors'))
		if index == 0:
			data['jobs'] = torch.tensor([len(plan)], dtype=torch.float32)
			save_file(data, str(folder / 'generation.safetensors'))
	environment = dict(os.environ, STRICT_GPU='all', CUDA_CACHE_PATH=str(args.cache / 'cuda'), CUDA_CACHE_MAXSIZE='4294967296')
	(args.cache / 'cuda').mkdir(exist_ok=True)
	command = [str(ROOT / 'MossGenerate.exe'), str(folder), str(model), adapter, str(vae), mode, '']
	worker = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
	completed = 0
	try:
		for line in worker.stdout:
			if line.startswith('evaluation '):
				line = 'step ' + line[len('evaluation '):]
			print(line, end='', flush=True)
			match = re.fullmatch(r'generation (\d+) in (\d+) ms\s*', line)
			if match:
				index = int(match[1])
				if index != completed or index >= len(plan):
					raise RuntimeError('Unexpected generation index from native backend')
				completed += 1
				suffix = f'-{index}' if len(plan) > 1 else ''
				extension = '.png' if args.output.suffix.lower() == '.png' else '.jpg'
				yield index, folder / ('strict-' + mode + suffix + extension), int(match[2])
		code = worker.wait()
		if code:
			raise subprocess.CalledProcessError(code, command)
		if completed != len(plan):
			raise RuntimeError('Native backend did not complete the requested image count; rebuild MossGenerate.strict')
	finally:
		if worker.poll() is None:
			worker.terminate()
			worker.wait()
		worker.stdout.close()


def main(argv=None):
	args = arguments(argv)
	started = time.perf_counter()
	models = args.comfyui / 'models'
	model = located(args.model, models / 'diffusion_models')
	metadata = validate_model(model)
	loras = selected_loras(args, models, metadata)
	args.cache = args.cache.resolve()
	args.cache.mkdir(parents=True, exist_ok=True)
	loaded = cached_merge(args, model, loras) if args.lora_mode == 'merged' else model
	if args.merge_only:
		print('Merged model: ' + str(loaded), flush=True)
		return 0
	vae = located(args.vae, models / 'vae')
	encoder = located(args.te, models / 'text_encoders')
	if not (ROOT / 'MossGenerate.exe').is_file():
		raise FileNotFoundError('Build MossGenerate.strict first: ../Strict/strict.exe build MossGenerate.strict')
	plan = jobs(args, args.prompt.read_text(encoding='utf-8-sig').strip())
	for index, job in enumerate(plan):
		print(f'Image {index + 1}/{args.count}, seed {job["seed"]}\nPrompt: {job["prompt"]}', flush=True)
	print('Input model: ' + str(model), flush=True)
	print('LoRA mode: ' + args.lora_mode + ', Spectrum: ' + args.spectrum, flush=True)
	for path, strength in loras:
		print(f'LoRA: {path}, strength {strength}', flush=True)
	contexts = conditioning(args, [job['prompt'] for job in plan], encoder)
	adapter = runtime_adapter(args, model, loras) if args.lora_mode == 'runtime' else ''
	from contextlib import closing
	with tempfile.TemporaryDirectory(dir=args.cache) as temporary:
		with closing(generated(args, loaded, vae, contexts, plan, Path(temporary), adapter)) as images:
			for index, image, generation_ms in images:
				output, handle = reserve_output(args.output.resolve())
				try:
					with handle, image.open('rb') as source:
						shutil.copyfileobj(source, handle)
					if args.json_output:
						settings = dict(plan[index], model=str(model), loaded_model=str(loaded),
							loras=[(str(path), strength) for path, strength in loras], baked_loras=baked_loras(metadata),
							lora_mode=args.lora_mode, spectrum=args.spectrum,
							steps=args.steps, width=args.width, height=args.height, guidance=args.guidance, vae=str(vae), te=str(encoder),
							jpg_quality=args.jpg_quality if output.suffix.lower() != '.png' else None, seconds=time.perf_counter() - started,
							batch_index=index + 1, count=args.count, generation_ms=generation_ms)
						with output.with_suffix(output.suffix + '.json').open('x', encoding='utf-8') as sidecar:
							json.dump(settings, sidecar, indent=2, ensure_ascii=False)
				except BaseException:
					handle.close()
					output.unlink(missing_ok=True)
					raise
				print(f'Saved {output}', flush=True)
	print(f'Total {time.perf_counter() - started:.2f} s for {args.count} image(s)', flush=True)
	return 0


if __name__ == '__main__':
	try:
		if len(sys.argv) > 1 and sys.argv[1] == '_encode':
			encoded(json.loads(Path(sys.argv[2]).read_text(encoding='utf-8')), Path(sys.argv[3]))
		else:
			raise SystemExit(main())
	except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
		print('Moss: ' + str(error), file=sys.stderr)
		raise SystemExit(1)
