"""The ComfyUI side of the Strict Moss and H3 samplers: quality presets, LoRA patches as adapters, cached int8 merges
(merge.py) and the resident MossImage.exe or H3Video.exe the nodes send their conditioning to."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
FAMILIES = {
	'moss': {'backend': 'MossImage', 'width': 720, 'height': 1280, 'linear': 'layers.0.proj.q'},
	'h3': {'backend': 'H3Video', 'width': 480, 'height': 832, 'linear': 'blocks.0.attn.qkv_proj'}}
SPECTRUM = re.compile(r'spectrum(\d+)_w(\d+)p(\d+)')
SKIP = re.compile(r'(\d+):(\d+)')
# H3 skip: the blocks every other evaluated step after the first two reuses, wider spans are faster
SKIPS = ['5:45', '10:40', '15:35', 'off']
# H3 sampler_name and scheduler like ComfyUI's (beta57 from RES4LYF), H3Video.exe gets their index; NOISY samplers add new
# noise every step but the last
SAMPLERS = ['euler', 'res_multistep', 'er_sde', 'euler_ancestral', 'dpmpp_2m_sde', 'gradient_estimation', 'lcm']
SCHEDULERS, NOISY = ['simple', 'sgm_uniform', 'normal', 'beta', 'beta57', 'linear_quadratic', 'hyperflow'], {'er_sde', 'euler_ancestral',
	'dpmpp_2m_sde', 'lcm'}
# the model and turbo LoRA of ComfyUI's H3 template (video_minimax_h3_t2v)
H3_BASE, H3_TURBO = 'minimax_h3_fl2va_pruned_int8_convrot.safetensors', 'minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors:1'
H3_TAOMATE = 'h3/minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors:1'
# H3 quality: model, steps, routing, Spectrum warmup and period (forecast steps), LoRAs, description
QUALITIES = {
	'best': (H3_BASE, 8, 0, '', [H3_TURBO], 'the turbo LoRA of the H3 template, 8 steps'),
	'fast': (H3_BASE, 8, 1, '', [H3_TURBO], 'best with routed attention like Sol-Attn'),
	'faster': (H3_BASE, 8, 1, 'w1p2', [H3_TURBO], 'fast with Spectrum (5 of 8 steps evaluated)'),
	'fastest4steps': (H3_BASE, 4, 1, '', [H3_TURBO], '4 routed turbo steps'),
	'fastest2steps': (H3_BASE, 2, 1, '', [H3_TAOMATE], 'the TaoMate 3-step LoRA instead, 2 routed steps, lowest quality')}
# Moss quality: steps, guidance, Spectrum warmup and period, description; 8 steps or fewer add a turbo LoRA
TURBO_LORAS = ('dmd_turbo_lora.safetensors', 'moss_image_sft_4step_v2.0.safetensors')
MOSS_QUALITIES = {
	'best': (8, 1, '', 'the DMD turbo LoRA, 8 steps'),
	'best25steps': (25, 2, '', 'without the turbo LoRA, 25 steps with cfg 2'),
	'best50steps': (50, 2, '', 'without the turbo LoRA, 50 steps with cfg 2'),
	'fast': (8, 1, 'w3p3', 'best with Spectrum (5 of 8 steps evaluated)'),
	'faster': (8, 1, 'w2p3', 'best with Spectrum (4 of 8 steps evaluated)'),
	'fastest': (4, 1, 'w2p2', '4 turbo steps with Spectrum (3 evaluated)'),
	'fastest2steps': (2, 1, '', '2 turbo steps, softer and about half the images usable')}


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


def arguments(argv=None, family='moss'):
	"""The settings of a node run from arguments like MossImage.exe's and H3Video.exe's, invalid ones exit like argparse."""
	settings = FAMILIES[family]
	parser = argparse.ArgumentParser(prog=settings['backend'], allow_abbrev=False)
	parser.add_argument('--quality', choices=QUALITIES if family == 'h3' else MOSS_QUALITIES, default='best')
	parser.add_argument('--steps', type=int, default=argparse.SUPPRESS)
	parser.add_argument('--spectrum', default=argparse.SUPPRESS)
	parser.add_argument('--lora_mode', choices=['merged', 'runtime'], default='merged' if family == 'moss' else 'runtime')
	parser.add_argument('--skip', default='5:45' if family == 'h3' else 'off')
	parser.add_argument('--width', type=int, default=settings['width'])
	parser.add_argument('--height', type=int, default=settings['height'])
	if family == 'h3':
		parser.add_argument('--length', type=int, default=124)
		parser.add_argument('--sampler', choices=SAMPLERS, default='euler')
		parser.add_argument('--scheduler', choices=SCHEDULERS, default='simple')
		parser.add_argument('--vae', default='minimax_h3_video_vae_int8_convrot.safetensors')
		parser.add_argument('--audio_vae', default='minimax_h3_audio_vae_fp32.safetensors')
		parser.add_argument('--decode', action='store_true')
	else:
		parser.add_argument('--guidance', type=float, default=argparse.SUPPRESS)
	parser.add_argument('--cache', type=Path, default=ROOT / 'artifacts' / family)
	args = parser.parse_args(argv)
	args.family = family
	quality(args)
	if args.steps < 1 or min(args.width, args.height) < 16 or args.width % 16 or args.height % 16:
		parser.error('steps must be positive; width/height must be positive multiples of 16')
	if family == 'moss' and not math.isfinite(args.guidance):
		parser.error('guidance must be finite')
	if family == 'h3' and args.length < 5:
		parser.error('length must be at least 5 frames')
	layers = 50 if family == 'h3' else 41
	if args.skip != 'off' and not (SKIP.fullmatch(args.skip) and int(args.skip.split(':')[0]) <
			int(args.skip.split(':')[1]) <= layers):
		parser.error(f'--skip must be off or first:last with first below last and last at most {layers}')
	spectrum = SPECTRUM.fullmatch(args.spectrum)
	if args.spectrum != 'off' and spectrum is None:
		parser.error('--spectrum must be off or spectrum<steps>_w<warmup>p<period>')
	if spectrum and int(spectrum[1]) != args.steps:
		parser.error(f'{args.spectrum} requires --steps {spectrum[1]}')
	return args


def quality(args):
	if args.family == 'h3':
		args.model, steps, args.routing, spectrum, loras = QUALITIES[args.quality][:5]
		args.lora = list(loras)
	else:
		steps, guidance, spectrum = MOSS_QUALITIES[args.quality][:3]
		args.guidance = getattr(args, 'guidance', guidance)
		args.lora = []
	args.steps = getattr(args, 'steps', steps)
	args.spectrum = getattr(args, 'spectrum', f'spectrum{args.steps}_{spectrum}' if spectrum else 'off')


def skipped_blocks(args):
	"""The first and the last block --skip reuses, 0 and 0 for off."""
	return (0, 0) if args.skip == 'off' else tuple(int(block) for block in args.skip.split(':'))


def spectrum_config(args):
	spectrum = SPECTRUM.fullmatch(args.spectrum)
	return ('off', 5, 2) if spectrum is None else ('on', int(spectrum[2]), int(spectrum[3]))


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
	"""The given LoRAs, after the turbo LoRA at 8 steps or fewer: akhaliq's 4-step one at 4 or fewer when installed, else DMD."""
	selected = [(located(path, models / 'loras'), strength) for path, strength in map(lora_spec, args.lora)]
	has_turbo = any(Path(str(path).replace('\\', '/')).name in TURBO_LORAS and strength != 0
		for path, strength in baked_loras(metadata))
	explicit_turbo = any(path.name in TURBO_LORAS for path, strength in selected)
	if args.family == 'moss' and args.steps <= 8 and not has_turbo and not explicit_turbo:
		four = models / 'loras/moss' / TURBO_LORAS[1]
		turbo = four.resolve() if args.steps <= 4 and four.is_file() else located('moss/' + TURBO_LORAS[0], models / 'loras')
		selected.insert(0, (turbo, 1.0))
	return selected


def identity(path):
	stat = path.stat()
	return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def native(path, family='moss'):
	"""True when Strict loads the checkpoint as it is, False for a Moss checkpoint merge.py converts first
	(bf16, fp16 or fp8 linears, fused qkv, ComfyUI's model prefix)."""
	from safetensors import safe_open
	with safe_open(path, framework='pt') as checkpoint:
		return layout(checkpoint, family)


def layout(checkpoint, family):
	names = [prefix + FAMILIES[family]['linear'] + '.comfy_quant' for prefix in ('', 'model.diffusion_model.')]
	key = next((name for name in names if name in checkpoint.keys()), None)
	if key is None:
		layouts = [prefix + 'layers.0.proj.' + name + '.weight' for prefix in ('', 'model.diffusion_model.') for name in ('q', 'qkv')]
		if family == 'moss' and any(name in checkpoint.keys() for name in layouts):
			return False
		raise ValueError('The model must be an int8 Convrot checkpoint for ' + FAMILIES[family]['backend'] +
			(', or a bf16, fp16 or fp8 Moss checkpoint' if family == 'moss' else ''))
	quant = json.loads(bytes(checkpoint.get_tensor(key).tolist()))
	if not quant.get('convrot') or quant.get('convrot_groupsize') != 256 or quant.get('format') != 'int8_tensorwise':
		raise ValueError('The model must use int8 Convrot with group size 256')
	return not key.startswith('model.') or family != 'moss'


def validate_model(path, family='moss'):
	from safetensors import safe_open
	with safe_open(path, framework='pt') as checkpoint:
		layout(checkpoint, family)
		return checkpoint.metadata() or {}


def cache_key(data):
	return hashlib.sha256(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()


def cached_merge(args, model, loras):
	from safetensors import safe_open
	active = [(path, strength) for path, strength in loras if strength != 0]
	if not active and native(model, args.family):
		return model
	script = ROOT / 'merge.py'
	version = hashlib.sha256(script.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
	key = cache_key(['convrot-int8-stochastic-v1', version, identity(model),
		[(identity(path), strength) for path, strength in active]])
	path = args.cache / ('merged-stochastic-v1-' + key + '.safetensors')
	if path.exists():
		with safe_open(path, framework='pt') as file:
			if 'strict_lora_merge' not in (file.metadata() or {}):
				raise ValueError('Invalid cached merge: ' + str(path))
		print('Using cached int8 Convrot model: ' + str(path), flush=True)
		os.utime(path)
		return path
	for old in sorted(args.cache.glob('merged-stochastic-v1-*.safetensors'), key=lambda file: file.stat().st_mtime)[:-2]:
		old.unlink()
	print('Please wait: ' + ('merging LoRAs into an' if active else 'converting to an') +
		' optimized int8 Convrot model, it is reused on the next run!', flush=True)
	with tempfile.TemporaryDirectory(dir=args.cache) as temporary:
		output = Path(temporary) / 'merged.safetensors'
		command = [sys.executable, str(script), str(model), str(output),
			*[str(source) + ':' + repr(strength) for source, strength in active], '--split-qkv']
		subprocess.run(command, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
			OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONUNBUFFERED='1'))
		os.replace(output, path)
	return path


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


# the files of patched_adapter per cache folder and LoRA patches, with their adapters so no other object gets their ids
PATCHED = {}


def patched_adapter(cache, patches):
	"""The LoRA patches of a ComfyUI MODEL (LoraLoader nodes) as one stacked adapter file in runtime_adapter's layout,
	named by its content so cached merges and adapters stay valid; the adapters of cached LoraLoader nodes (the same for
	every run) give their file without converting and hashing their tensors again."""
	import torch
	from safetensors.torch import save_file
	identity = (str(cache), tuple((key, tuple((entry[0], id(entry[1]), *entry[2:]) for entry in entries))
		for key, entries in sorted(patches.items())))
	if identity in PATCHED:
		return PATCHED[identity][1]
	tensors = {}
	for key, entries in sorted(patches.items()):
		downs, ups = [], []
		for strength, adapter, model_strength, offset, function in entries:
			weights = getattr(adapter, 'weights', ())
			if (getattr(adapter, 'name', '') != 'lora' or model_strength != 1 or offset is not None or function is not None
				or any(value is not None for value in weights[3:])):
				raise ValueError('Only plain LoRA patches are supported, not on ' + key)
			up, down, alpha = weights[:3]
			if strength != 0:
				downs.append(down.float().cpu().flatten(1))
				ups.append(up.float().cpu().flatten(1) * (strength * (1 if alpha is None else alpha / down.shape[0])))
		if downs:
			prefix = key.removesuffix('.weight') + '.'
			tensors[prefix + 'lora_down.weight'] = torch.cat(downs).contiguous()
			tensors[prefix + 'lora_up.weight'] = torch.cat(ups, dim=1).contiguous()
			tensors[prefix + 'alpha'] = torch.tensor(float(tensors[prefix + 'lora_down.weight'].shape[0]))
	if not tensors:
		return None
	content = hashlib.sha256()
	for name, value in tensors.items():
		content.update(name.encode('utf-8'))
		content.update(value.numpy().tobytes())
	path = cache / ('patches-' + content.hexdigest() + '.safetensors')
	if not path.exists():
		with tempfile.TemporaryDirectory(dir=cache) as temporary:
			save_file(tensors, str(Path(temporary) / 'adapter.safetensors'))
			os.replace(Path(temporary) / 'adapter.safetensors', path)
	PATCHED[identity] = ([entry[1] for entries in patches.values() for entry in entries], path)
	return path


def releasable(prompt, node):
	"""Whether the Load Diffusion Model behind the model input of node in a ComfyUI prompt (through LoRA loaders) feeds
	only Strict samplers, so its weights can be freed: Strict loads the checkpoint itself and reads only its file and
	the LoRA patches."""
	chain, source = [], prompt[node]['inputs'].get('model')
	while isinstance(source, list) and source[0] in prompt and source[0] not in chain:
		chain.append(source[0])
		upstream = prompt[source[0]]
		source = upstream['inputs'].get('model') if upstream['class_type'] in ('LoraLoader', 'LoraLoaderModelOnly') else None
	if not chain or prompt[chain[-1]]['class_type'] != 'UNETLoader':
		return False
	readers = [(other['class_type'], value) for key, other in prompt.items() if key not in chain
		for value in other['inputs'].values() if isinstance(value, list) and len(value) == 2 and value[0] in chain]
	return all(kind in ('StrictH3Sampler', 'StrictH3LatentSampler', 'StrictMossSampler') or prompt[value[0]]['class_type'] == 'LoraLoader' and value[1] == 1
		for kind, value in readers)


def turbo_patched(patches, turbo):
	"""Whether a LoraLoader node already applied the turbo LoRA, recognized by its first lora_down (lora_A) tensor."""
	import torch
	from safetensors import safe_open
	with safe_open(turbo, framework='pt') as file:
		name = min(key for key in file.keys() if key.endswith(('.lora_down.weight', '.lora_A.weight')))
		down = file.get_tensor(name).float()
	key = name.rsplit('.lora_', 1)[0]
	key = key if key.startswith('diffusion_model.') else 'diffusion_model.' + key
	return any(getattr(entry[1], 'name', '') == 'lora' and entry[1].weights[1].shape == down.shape and
		torch.equal(entry[1].weights[1].float().cpu(), down) for entry in patches.get(key + '.weight', []))


def unapplied(loras, checkpoint, patches):
	"""The H3 quality LoRAs the model does not have yet: no turbo LoRA for a turbo checkpoint, none a LoraLoader applied."""
	baked = 'turbo' in Path(checkpoint).name.lower()
	return [(path, strength) for path, strength in loras if not (baked and 'turbo' in Path(path).name.lower()) and
		not (patches and turbo_patched(patches, path))]


def lora_model(args, model, loras):
	"""The checkpoint the backend loads and its LoRA argument: a cached merge of the LoRAs, or the checkpoint (converted
	once when Strict can not load it) with the LoRAs H3Video.exe merges at load or the adapter MossImage.exe merges at load."""
	if args.lora_mode == 'merged':
		return cached_merge(args, model, loras), ''
	loaded = cached_merge(args, model, [])
	return loaded, runtime_spec(loras) if args.family == 'h3' else runtime_adapter(args, loaded, loras)


def runtime_spec(loras):
	"""The path=strength;... argument H3Video.exe merges at load."""
	active = [(str(path), strength) for path, strength in loras if strength != 0]
	if any(';' in path or '=' in path for path, strength in active):
		raise ValueError('LoRA paths for --lora_mode runtime must not contain ; or =')
	return ';'.join(f'{path}={strength}' for path, strength in active)


def h3_noise(args, seed):
	"""Video and audio noise of one H3 job like ComfyUI's RandomNoise: one generator, the video latent first, then for
	a NOISY sampler the noise of every step but the last (video.noise.<step> and audio.noise.<step>)."""
	import torch
	frames = args.length + (5 - args.length) % 17
	latent = 2 if frames <= 5 else (frames - 5) // 17 * 5 + 2
	shapes = (1, 24, latent, args.height // 16, args.width // 16), (1, 32, 2, round(frames / 24 * 40))
	generator = torch.Generator('cpu').manual_seed(seed)
	steps = range(args.steps - 1 if args.sampler in NOISY else 0)
	names = ['video', 'audio'] + [f'{stream}.noise.{step}' for step in steps for stream in ('video', 'audio')]
	return {name: torch.randn(shapes[index % 2], generator=generator)[0] for index, name in enumerate(names)}


def h3_condition(conditioning):
	"""Qwen3-VL states, token tags and keyframe latents with their frame indices of a ComfyUI H3 conditioning, as
	conditioning.safetensors keeps them."""
	import torch
	tags = conditioning[0][1].get('minimax_token_tags')
	tags = torch.ones(conditioning[0][0].shape[1]) if tags is None else tags
	condition = {'context': conditioning[0][0][0], 'tags': tags.to(torch.float32).flatten()}
	keyframes = conditioning[0][1].get('minimax_keyframes') or []
	for index, keyframe in enumerate(keyframes):
		condition[f'keyframe.{index}'] = keyframe['latent'][0]
	if keyframes:
		condition['keyframes'] = torch.tensor([float(keyframe['resolved_frame_index']) for keyframe in keyframes])
	return condition


def h3_arguments(quality, video, audio, skip='5:45', sampler='euler', scheduler='simple', decode=False):
	"""The quality, skip, sampler and scheduler settings for a ComfyUI H3 latent of shapes video [B, 24, T, H, W]
	and audio [B, 32, 2, T]: its size and frame count, decode for H3Video.exe to decode the videos too."""
	frames = (video[2] - 2) // 5 * 17 + 5
	if len(video) != 5 or video[1] != 24 or tuple(audio) != (video[0], 32, 2, round(frames / 24 * 40)):
		raise ValueError('latent must be an H3 video and audio latent (MiniMax H3 Image to Video or Empty MiniMax H3 AV Latent)')
	return arguments(['--quality', quality, '--skip', skip, '--sampler', sampler, '--scheduler', scheduler,
		'--width', str(video[4] * 16), '--height', str(video[3] * 16), '--length', str(frames)] + ['--decode'] * decode, 'h3')


def h3_latents(numbers, video, audio):
	"""The video and the audio latent of one job from the numbers H3Video.exe writes (the video, then the audio), shaped
	like the ComfyUI latent of one batch entry."""
	size = math.prod(video[1:])
	return numbers[:size].reshape(1, *video[1:]), numbers[size:].reshape(1, *audio[1:])


def skipping(executor):
	"""A submit to executor that drops work while the last submitted still runs (previews of forecast steps that come
	faster than they decode)."""
	running = [None]
	def submit(function, *arguments):
		if running[0] is None or running[0].done():
			running[0] = executor.submit(function, *arguments)
	return submit


def preview_batch(video):
	"""The latent frames of an H3 latent of shape video (ComfyUI's, [B, 24, T, H, W]) a preview decodes at once: 4 at
	480x832 (0.6 GB of GPU memory for taeh3), 1 from 720p on."""
	return max(1, 4 * 30 * 52 // (video[3] * video[4]))


def h3_preview(numbers, video, decoder):
	"""An animated WebP of the denoised video H3Video.exe writes for a step (numbers, video the ComfyUI latent shape),
	decoder turning a batch of latent frames at their own size into images: every latent frame (17 / 5 video frames, about
	7 per second, playing in real time) scaled to at most 512 pixels."""
	import torch
	from PIL import Image
	latents, batch, frames = numbers.reshape(video[1:]).movedim(1, 0), preview_batch(video), []
	for first in range(0, len(latents), batch):
		images = decoder(latents[first:first + batch]).float()
		images = torch.nn.functional.interpolate(images, scale_factor=min(1, 512 / max(images.shape[-2:])), mode='bilinear',
			antialias=True)
		frames += [Image.fromarray(image.clamp(0, 1).mul(255).byte().movedim(0, -1).cpu().numpy()) for image in images]
	buffer = io.BytesIO()
	frames[0].save(buffer, 'WEBP', save_all=True, append_images=frames[1:], duration=round(17 / 5 / 24 * 1000), loop=0,
		quality=85, method=0)
	return buffer.getvalue()


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


def executable(family):
	"""The family's backend in Release/Windows (.exe) or Release/Linux (no extension)."""
	name = FAMILIES[family]['backend']
	windows = sys.platform == 'win32'
	return str(ROOT / 'Release' / ('Windows' if windows else 'Linux') / (name + '.exe' if windows else name))


def backend(args, model, adapter):
	"""The resident backend command, it reads one job folder per input line (H3Video.exe decodes jobs with decode)."""
	mode, warmup, period = spectrum_config(args)
	if args.family == 'h3':
		return [executable(args.family), '-', str(model), 'strict', adapter, str(args.steps), '12',
			str(period if mode == 'on' else 0), str(warmup if mode == 'on' else 1), '1', f'{args.routing:g}',
			*map(str, skipped_blocks(args)), str(SAMPLERS.index(args.sampler)), str(SCHEDULERS.index(args.scheduler)),
			str(args.vae), str(args.audio_vae)]
	return [executable(args.family), '-', str(model), adapter, mode]


def prepared(args, contexts, plan, folder):
	import torch
	from safetensors.torch import load, save_file
	mode, warmup, period = spectrum_config(args)
	for index, (context, job) in enumerate(zip(contexts, plan)):
		data = load(Path(context).read_bytes())
		if args.family == 'h3':
			latents = [data.pop(f'keyframe.{number}') for number in range(len(data.get('keyframes', [])))]
			if latents:
				data['rows'] = h3_rows(latents, job['seed'])
			data |= h3_noise(args, job['seed']) | ({'decode': torch.ones(1)} if args.decode else {})
		else:
			data['noise'] = torch.randn((1, 128, args.height // 16, args.width // 16),
				generator=torch.Generator('cpu').manual_seed(job['seed']))
			for name, value in {'steps': args.steps, 'guidance': args.guidance, 'warmup': warmup, 'period': period}.items():
				data[name] = torch.tensor([value], dtype=torch.float32)
			data['span'] = torch.tensor(skipped_blocks(args), dtype=torch.float32)
		save_file(data, str(folder / f'generation-{index}.safetensors'))
		if index == 0:
			data['jobs'] = torch.tensor([len(plan)], dtype=torch.float32)
			save_file(data, str(folder / 'generation.safetensors'))


def started(args, command):
	if not Path(command[0]).is_file():
		raise FileNotFoundError(command[0] + ' is missing, reinstall StrictComfy or build it with: strict build ' +
			Path(command[0]).stem + '.strict ' + Path(command[0]).parent.name.lower())
	if not os.access(command[0], os.X_OK):
		Path(command[0]).chmod(Path(command[0]).stat().st_mode | 0o111)
	environment = dict(os.environ, STRICT_GPU='all', CUDA_CACHE_PATH=str(args.cache / 'cuda'), CUDA_CACHE_MAXSIZE='4294967296')
	(args.cache / 'cuda').mkdir(exist_ok=True)
	return subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)


def collected(args, worker, plan, folder, stepped=None):
	"""The latents of plan (H3: the frames, the audio next to them) from the backend's output lines, as soon as each is done,
	stepped called with the line of each step."""
	mode = spectrum_config(args)[0]
	completed, decoded = 0, args.family == 'h3' and args.decode
	for line in worker.stdout:
		if line.startswith('evaluation '):
			line = 'step ' + line[len('evaluation '):]
		print(line, end='', flush=True)
		if stepped and line.startswith('step '):
			stepped(line)
		match = re.fullmatch(('decoded' if decoded else 'generation') + r' (\d+) in (\d+) ms\s*', line)
		if match:
			index = int(match[1])
			if index != completed or index >= len(plan):
				raise RuntimeError('Unexpected generation index from native backend')
			completed += 1
			suffix = f'-{index}' if len(plan) > 1 else ''
			name = f'strict-{index}-{"decoded.rgb" if decoded else "latent.safetensors"}' if args.family == 'h3' else f'strict-{mode}{suffix}-latent.safetensors'
			yield index, folder / name, int(match[2])
			if completed == len(plan):
				return
	code = worker.wait()
	if code:
		raise subprocess.CalledProcessError(code, worker.args)
	raise RuntimeError('Native backend did not complete the requested count; rebuild ' + Path(worker.args[0]).stem + '.strict')


def stopped(worker):
	if worker.poll() is None:
		worker.terminate()
		worker.wait()
	worker.stdout.close()


class Resident:
	"""One MossImage.exe or H3Video.exe that keeps its model loaded between runs: each run writes its folder to the
	backend's input, other settings (model, LoRAs, Spectrum, steps, routing) start a new one."""
	worker, command = None, None

	def generated(self, args, model, contexts, plan, folder, adapter='', stepped=None):
		prepared(args, contexts, plan, folder)
		command = backend(args, model, adapter)
		if self.worker is None or self.worker.poll() is not None or self.command != command:
			self.close()
			self.worker, self.command = started(args, command), command
		try:
			self.worker.stdin.write(str(folder) + '\n')
			self.worker.stdin.flush()
			yield from collected(args, self.worker, plan, folder, stepped)
		except BaseException:
			self.close()
			raise

	def close(self):
		if self.worker is not None:
			stopped(self.worker)
			self.worker = None
