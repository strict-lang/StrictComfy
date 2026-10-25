"""Moss images (MossImage.exe) and H3 videos with sound (H3Video.exe): ComfyUI encodes the prompts and decodes H3 videos,
Strict samples (and decodes Moss images)."""
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
import threading
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
FAMILIES = {
	'moss': {'program': 'MossImage.exe', 'backend': 'MossGenerate.exe', 'noun': 'Image', 'output': 'MossImage.jpg',
		'suffixes': ('.jpg', '.jpeg', '.png'), 'width': 720, 'height': 1280, 'spectrum': 'spectrum8_w3p3', 'linear': 'layers.0.proj.q',
		'files': [('model', 'diffusion_models', 'moss-image-sft-dit-int8-convrot.safetensors', 'Base int8 Convrot model'),
			('vae', 'vae', 'flux2-vae.safetensors', 'Image VAE'),
			('te', 'text_encoders', 'qwen3.5_9b_int8_convrot.safetensors', 'Text encoder')]},
	'h3': {'program': 'H3Video.exe', 'backend': 'H3Video.exe', 'noun': 'Video', 'output': 'H3Video.mp4',
		'suffixes': ('.mp4',), 'width': 480, 'height': 832, 'spectrum': 'spectrum8_w1p2', 'linear': 'blocks.0.attn.qkv_proj',
		'files': [('model', 'diffusion_models', 'Minimax_H3_FL2VA_PRUNED_Turbo_8step_v1.0_768p_comfyui_INT8_.safetensors', 'int8 Convrot model'),
			('vae', 'vae', 'minimax_h3_video_vae_int8_convrot.safetensors', 'Video VAE'),
			('audio_vae', 'vae', 'minimax_h3_audio_vae_fp32.safetensors', 'Audio VAE'),
			('te', 'text_encoders', 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'Text encoder')]}}
SPECTRUM = re.compile(r'spectrum(\d+)_w(\d+)p(\d+)')


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


def arguments(argv=None, family='moss'):
	settings = FAMILIES[family]
	default_comfy = next((Path(drive + ':/ComfyUI_windows_portable/ComfyUI') for drive in ('D', 'C')
		if Path(drive + ':/ComfyUI_windows_portable/ComfyUI').is_dir()), Path('ComfyUI'))
	default_comfy = Path(os.environ.get('COMFYUI', default_comfy))
	files = settings['files']
	parser = argparse.ArgumentParser(prog=settings['program'], description=__doc__, add_help=False,
		allow_abbrev=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--help', action='help', help='show this help message and exit')
	for name, folder, filename, description in files:
		parser.add_argument('--' + name, default=argparse.SUPPRESS,
			help=f'{description} (default: {default_comfy / "models" / folder / filename})')
	parser.add_argument('--prompt', type=Path, default=Path('prompt.txt'), help='UTF-8 prompt file; supports {a|b|}')
	parser.add_argument('--steps', type=int, default=8,
		help='Diffusion sampling steps' + ('; 8 automatically applies the DMD turbo LoRA' if family == 'moss' else ''))
	parser.add_argument('--count', type=int, default=1, help='outputs per loaded model; fresh seed/prompt choices each time')
	parser.add_argument('--lora', action='append', default=[], help='repeat for each path[:strength], default strength 1')
	parser.add_argument('--lora_mode', choices=['merged', 'runtime'], default='merged',
		help='merged: cached INT8 stochastic merge (fast, approximate); runtime: ' +
		('floating adapters' if family == 'moss' else 'merged by Strict at every load, no cached model file'))
	parser.add_argument('--merge_only', action='store_true', help='prepare/print the cached merged model without encoding or generating')
	parser.add_argument('--spectrum', default='off',
		help=f'off, or spectrum<steps>_w<warmup>p<period> like {settings["spectrum"]}: forecasts steps; changes the sampling result')
	parser.add_argument('--output', type=Path, default=Path(settings['output']), help='numbered output base; never overwritten')
	parser.add_argument('--json_output', action='store_true', help='write a per-output JSON settings sidecar')
	parser.add_argument('--width', type=int, default=settings['width'], help='Width in pixels')
	parser.add_argument('--height', type=int, default=settings['height'], help='Height in pixels')
	if family == 'h3':
		parser.add_argument('--length', type=int, default=124, help='Frames at 24 fps, snapped up to 17k+5 like ComfyUI')
		parser.add_argument('--shift', type=float, default=12, help='Flow shift of the video sigmas, the audio follows with shift 3')
		parser.add_argument('--image', type=Path, help='First frame (image to video), resized to width and height')
		parser.add_argument('--last_image', type=Path, help='Last frame, cover-cropped to width and height')
		parser.add_argument('--routing', type=float, default=0, help='Sparse attention like Sol-Attn above 0: key blocks whose '
			'pooled score stays below this many standard deviations are pooled (1 is ~0.85 s faster per step on a 4090); '
			'changes the sampling result')
	parser.add_argument('--seed', type=int, default=None, help='random by default; fixes prompt choices and noise')
	if family == 'moss':
		parser.add_argument('--guidance', type=float, default=1, help='Classifier-free guidance (CFG) scale')
		parser.add_argument('--jpg_quality', type=int, default=85, help='JPEG compression quality, 0-100; does not affect inference')
	parser.add_argument('--comfyui', type=Path, default=default_comfy, help='Installed ComfyUI directory')
	parser.add_argument('--cache', type=Path, default=ROOT / 'artifacts' / family, help='Prompt, merged-model, runtime-adapter and CUDA caches')
	parser.add_argument('--python', type=Path, help='Python interpreter for the native launcher; use ComfyUI Python')
	args = parser.parse_args(argv)
	args.family = family
	for name, folder, filename, description in files:
		if not hasattr(args, name):
			setattr(args, name, str(args.comfyui / 'models' / folder / filename))
	if args.steps < 1 or args.count < 1 or min(args.width, args.height) < 16 or args.width % 16 or args.height % 16:
		parser.error('count/steps must be positive; width/height must be positive multiples of 16')
	if family == 'moss' and (not 0 <= args.jpg_quality <= 100 or not math.isfinite(args.guidance)):
		parser.error('jpg_quality must be 0-100 and guidance finite')
	if family == 'h3' and (args.length < 5 or not math.isfinite(args.shift) or args.shift <= 0):
		parser.error('length must be at least 5 frames and shift positive')
	if args.seed is not None and not 0 <= args.seed < 2**63:
		parser.error('seed must be between 0 and 2^63-1')
	if args.output.suffix.lower() not in settings['suffixes']:
		parser.error('output extension must be ' + ' or '.join(settings['suffixes']))
	spectrum = SPECTRUM.fullmatch(args.spectrum)
	if args.spectrum != 'off' and spectrum is None:
		parser.error('--spectrum must be off or spectrum<steps>_w<warmup>p<period>')
	if spectrum and int(spectrum[1]) != args.steps:
		parser.error(f'{args.spectrum} requires --steps {spectrum[1]}')
	if args.merge_only and args.lora_mode != 'merged':
		parser.error('--merge_only requires --lora_mode merged')
	return args


def spectrum_config(args):
	spectrum = SPECTRUM.fullmatch(args.spectrum)
	return ('off', 5, 2) if spectrum is None else ('on', int(spectrum[2]), int(spectrum[3]))


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
	if args.family == 'moss' and args.steps == 8 and not has_turbo and not explicit_turbo:
		turbo = located('moss/dmd_turbo_lora.safetensors', models / 'loras')
		selected.insert(0, (turbo, 1.0))
	return selected


def identity(path):
	stat = path.stat()
	return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def validate_model(path, family='moss'):
	from safetensors import safe_open
	with safe_open(path, framework='pt') as checkpoint:
		names = [prefix + FAMILIES[family]['linear'] + '.comfy_quant' for prefix in ('', 'model.diffusion_model.')]
		key = next((name for name in names if name in checkpoint.keys()), None)
		if key is None:
			raise ValueError('--model must be an int8 Convrot checkpoint for ' + FAMILIES[family]['program'])
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
	family = request['family']
	sys.argv = ['dump.py', family]
	import dump
	from safetensors.torch import save_file
	types = dump.comfy.sd.CLIPType
	clip = dump.comfy.sd.load_clip([request['te']], embedding_directory=dump.folder_paths.get_folder_paths('embeddings'),
		clip_type=types.MINIMAX if family == 'h3' else types.MOSS_IMAGE)
	images = request.get('images', [])
	if any(images):
		from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
		vae = dump.comfy.sd.VAE(sd=dump.comfy.utils.load_torch_file(request['vae']))
		first, last = [dump.image_tensor(path) if path else None for path in images]
	for index, prompt in enumerate(request['prompts']):
		print(f'Encoding prompt {index + 1}/{len(request["prompts"])}', flush=True)
		if any(images):
			conditioning = MiniMaxH3ImageToVideo.execute(clip, vae, prompt, request['width'], request['height'], request['length'],
				first_frame=first, last_frame=last).result[0]
		else:
			conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
		tensors = dump.h3_condition(conditioning) if family == 'h3' else {'context': conditioning[0][0]}
		save_file({name: value.detach().cpu().float().contiguous() for name, value in tensors.items()},
			str(output / f'{index}.safetensors'), metadata={'prompt': prompt})


def decoding(request):
	"""Worker: imports ComfyUI ("decoder started") and loads its video and audio VAEs while the backend samples, then
	decodes every H3 latent of the jobs on stdin into <prefix>video.mp4."""
	os.environ['COMFYUI'] = request['comfyui']
	sys.argv = ['dump.py', 'h3', '--audio-vae', request['audio_vae']]
	import dump
	print('decoder started', flush=True)
	from safetensors import safe_open
	vae = dump.comfy.sd.VAE(sd=dump.comfy.utils.load_torch_file(request['vae']))
	dump.audio_decoder()
	for job in json.loads(sys.stdin.read()):
		with safe_open(job['generation'], framework='pt') as file:
			shapes = file.get_slice('video').get_shape(), file.get_slice('audio').get_shape()
		dump.decoded_latent(vae, job['latent'], shapes, request['folder'], job['prefix'])


def framing(args):
	"""The H3 frame images and what their conditioning depends on, empty without images."""
	images = [str(path) if path else '' for path in (getattr(args, 'image', None), getattr(args, 'last_image', None))]
	if not any(images):
		return {}
	return {'images': images, 'width': args.width, 'height': args.height, 'length': args.length, 'vae': str(args.vae)}


def conditioning(args, prompts, encoder):
	paths, missing = [], {}
	framed = framing(args)
	for prompt in prompts:
		request = {'prompt': prompt, 'te': str(encoder), 'comfyui': str(args.comfyui.resolve()), **framed}
		inputs = [Path(path) for path in framed.get('images', []) if path] + ([Path(framed['vae'])] if framed else [])
		key = cache_key([1, request, identity(encoder), identity(args.comfyui / 'comfy' / 'sd.py'), *map(identity, inputs)])
		path = args.cache / ('context-' + key + '.safetensors')
		paths.append(path)
		if not path.exists():
			missing[path] = prompt
	if missing:
		print(f'Preparing {len(missing)} uncached prompts with one text-encoder load', flush=True)
		with tempfile.TemporaryDirectory(dir=args.cache) as temporary:
			folder = Path(temporary)
			request = {'prompts': list(missing.values()), 'te': str(encoder), 'comfyui': str(args.comfyui.resolve()),
				'family': args.family, **framed}
			(folder / 'request.json').write_text(json.dumps(request), encoding='utf-8')
			subprocess.run([sys.executable, str(ROOT / 'generate.py'), '_encode', str(folder / 'request.json'),
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
	key = cache_key([1, identity(ROOT / 'generate.py'), identity(model), [(identity(path), strength) for path, strength in active]])
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


def runtime_spec(loras):
	"""The path=strength;... argument H3Video.exe merges at load."""
	active = [(str(path), strength) for path, strength in loras if strength != 0]
	if any(';' in path or '=' in path for path, strength in active):
		raise ValueError('LoRA paths for --lora_mode runtime must not contain ; or =')
	return ';'.join(f'{path}={strength}' for path, strength in active)


def h3_noise(args, seed):
	"""Video and audio noise of one H3 job like ComfyUI's RandomNoise: one generator, the video latent first."""
	import torch
	frames = args.length + (5 - args.length) % 17
	latent = 2 if frames <= 5 else (frames - 5) // 17 * 5 + 2
	shapes = (1, 24, latent, args.height // 16, args.width // 16), (1, 32, 2, round(frames / 24 * 40))
	generator = torch.Generator('cpu').manual_seed(seed)
	return tuple(torch.randn(shape, generator=generator)[0] for shape in shapes)


def h3_rows(latents, seed):
	"""ComfyUI's conditioning rows of H3 keyframe latents: 2x2 patches, 0.999 latent and 0.001 noise, the seeded noise
	restarting for every keyframe."""
	import torch
	rows = []
	for latent in latents:
		channels, frames, height, width = latent.shape
		patches = latent.float().reshape(channels, frames, height // 2, 2, width // 2, 2).permute(1, 2, 4, 0, 3, 5)
		patches = patches.reshape(-1, channels * 4)
		noise = torch.randn(patches.shape, generator=torch.Generator('cpu').manual_seed(seed))
		rows.append(0.999 * patches + 0.001 * noise)
	return torch.cat(rows)


def backend(args, folder, model, vae, adapter):
	mode, warmup, period = spectrum_config(args)
	if args.family == 'h3':
		return [str(ROOT / 'H3Video.exe'), str(folder), str(model), 'strict', adapter, str(args.steps), f'{args.shift:g}',
			str(period if mode == 'on' else 0), str(warmup if mode == 'on' else 1), '1', f'{args.routing:g}']
	return [str(ROOT / 'MossGenerate.exe'), str(folder), str(model), adapter, str(vae), mode, '']


def generated(args, model, vae, contexts, plan, folder, adapter=''):
	import torch
	from safetensors.torch import load, save_file
	mode, warmup, period = spectrum_config(args)
	for index, (context, job) in enumerate(zip(contexts, plan)):
		data = load(Path(context).read_bytes())
		if args.family == 'h3':
			latents = [data.pop(f'keyframe.{number}') for number in range(len(data.get('keyframes', [])))]
			if latents:
				data['rows'] = h3_rows(latents, job['seed'])
			data['video'], data['audio'] = h3_noise(args, job['seed'])
		else:
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
	command = backend(args, folder, model, vae, adapter)
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
				name = f'strict-{index}-latent.safetensors' if args.family == 'h3' else 'strict-' + mode + suffix + extension
				yield index, folder / name, int(match[2])
		code = worker.wait()
		if code:
			raise subprocess.CalledProcessError(code, command)
		if completed != len(plan):
			raise RuntimeError('Native backend did not complete the requested count; rebuild ' + Path(command[0]).stem + '.strict')
	finally:
		if worker.poll() is None:
			worker.terminate()
			worker.wait()
		worker.stdout.close()


def decoded(args, latents, folder):
	"""Decodes the latents H3Video.exe writes in one ComfyUI worker: it imports ComfyUI and loads the VAEs (about 20 s)
	while the backend loads and samples, and gets the jobs once the backend gave the GPU back."""
	request = {'comfyui': str(args.comfyui.resolve()), 'vae': str(args.vae), 'audio_vae': str(args.audio_vae), 'folder': str(folder)}
	(folder / 'decode.json').write_text(json.dumps(request), encoding='utf-8')
	command = [sys.executable, str(ROOT / 'generate.py'), '_decode', str(folder / 'decode.json')]
	worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
		text=True, encoding='utf-8', errors='replace', bufsize=1)
	def forwarded():
		for line in worker.stdout:
			print(line, end='', flush=True)
	forwarding = threading.Thread(target=forwarded, daemon=True)
	forwarding.start()
	try:
		latents = list(latents)
		if worker.poll() is not None:
			raise subprocess.CalledProcessError(worker.returncode, command)
		worker.stdin.write(json.dumps([{'latent': str(path), 'generation': str(folder / f'generation-{index}.safetensors'),
			'prefix': f'strict-{index}-'} for index, path, milliseconds in latents]))
		worker.stdin.close()
		code = worker.wait()
	except BaseException:
		worker.kill()
		raise
	forwarding.join()
	if code:
		raise subprocess.CalledProcessError(code, command)
	return [(index, folder / f'strict-{index}-video.mp4', milliseconds) for index, path, milliseconds in latents]


def main(argv=None):
	argv = sys.argv[1:] if argv is None else argv
	if not argv or argv[0] not in FAMILIES:
		raise ValueError('The first argument must be ' + ' or '.join(FAMILIES))
	args = arguments(argv[1:], argv[0])
	started = time.perf_counter()
	models = args.comfyui / 'models'
	model = located(args.model, models / 'diffusion_models')
	metadata = validate_model(model, args.family)
	loras = selected_loras(args, models, metadata)
	args.cache = args.cache.resolve()
	args.cache.mkdir(parents=True, exist_ok=True)
	loaded = cached_merge(args, model, loras) if args.lora_mode == 'merged' else model
	if args.merge_only:
		print('Merged model: ' + str(loaded), flush=True)
		return 0
	vae = args.vae = located(args.vae, models / 'vae')
	if args.family == 'h3':
		args.audio_vae = located(args.audio_vae, models / 'vae')
		args.image, args.last_image = [located(path, Path.cwd()) if path else None for path in (args.image, args.last_image)]
	encoder = located(args.te, models / 'text_encoders')
	name = Path(FAMILIES[args.family]['backend']).stem
	if not (ROOT / (name + '.exe')).is_file():
		raise FileNotFoundError(f'Build {name}.strict first: ../Strict/strict.exe build {name}.strict')
	plan = jobs(args, args.prompt.read_text(encoding='utf-8-sig').strip())
	for index, job in enumerate(plan):
		print(f'{FAMILIES[args.family]["noun"]} {index + 1}/{args.count}, seed {job["seed"]}\nPrompt: {job["prompt"]}', flush=True)
	print('Input model: ' + str(model), flush=True)
	print('LoRA mode: ' + args.lora_mode + ', Spectrum: ' + args.spectrum, flush=True)
	for path, strength in loras:
		print(f'LoRA: {path}, strength {strength}', flush=True)
	contexts = conditioning(args, [job['prompt'] for job in plan], encoder)
	adapter = '' if args.lora_mode == 'merged' else runtime_spec(loras) if args.family == 'h3' else runtime_adapter(args, model, loras)
	from contextlib import closing
	with tempfile.TemporaryDirectory(dir=args.cache) as temporary:
		with closing(generated(args, loaded, vae, contexts, plan, Path(temporary), adapter)) as images:
			outputs = decoded(args, images, Path(temporary)) if args.family == 'h3' else images
			for index, image, generation_ms in outputs:
				output, handle = reserve_output(args.output.resolve())
				try:
					with handle, image.open('rb') as source:
						shutil.copyfileobj(source, handle)
					if args.json_output:
						settings = dict(plan[index], model=str(model), loaded_model=str(loaded),
							loras=[(str(path), strength) for path, strength in loras], baked_loras=baked_loras(metadata),
							lora_mode=args.lora_mode, spectrum=args.spectrum,
							steps=args.steps, width=args.width, height=args.height, vae=str(vae), te=str(encoder),
							seconds=time.perf_counter() - started, batch_index=index + 1, count=args.count, generation_ms=generation_ms)
						settings.update({name: getattr(args, name) for name in ('guidance', 'length', 'shift') if hasattr(args, name)})
						settings.update({name: str(getattr(args, name)) for name in ('image', 'last_image') if getattr(args, name, None)})
						if args.family == 'moss':
							settings['jpg_quality'] = args.jpg_quality if output.suffix.lower() != '.png' else None
						with output.with_suffix(output.suffix + '.json').open('x', encoding='utf-8') as sidecar:
							json.dump(settings, sidecar, indent=2, ensure_ascii=False)
				except BaseException:
					handle.close()
					output.unlink(missing_ok=True)
					raise
				print(f'Saved {output}', flush=True)
	print(f'Total {time.perf_counter() - started:.2f} s for {args.count} {FAMILIES[args.family]["noun"].lower()}(s)', flush=True)
	return 0


if __name__ == '__main__':
	try:
		if len(sys.argv) > 1 and sys.argv[1] == '_encode':
			encoded(json.loads(Path(sys.argv[2]).read_text(encoding='utf-8')), Path(sys.argv[3]))
		elif len(sys.argv) > 1 and sys.argv[1] == '_decode':
			decoding(json.loads(Path(sys.argv[2]).read_text(encoding='utf-8')))
		else:
			raise SystemExit(main())
	except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
		print(Path(sys.argv[0]).stem + ': ' + str(error), file=sys.stderr)
		raise SystemExit(1)
