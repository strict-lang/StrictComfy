"""User-facing Moss generation; ComfyUI encodes text, Strict samples and decodes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
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
	parser = argparse.ArgumentParser(prog='MossImage.exe', description=__doc__)
	parser.add_argument('--prompt', type=Path, default=Path('prompt.txt'), help='UTF-8 prompt file; supports {a|b|}')
	parser.add_argument('--model', required=True, help='int8 Convrot Moss checkpoint, base or premerged turbo')
	parser.add_argument('--steps', type=int, default=8)
	parser.add_argument('--lora', action='append', default=[], help='repeat for each path[:strength], default strength 1')
	parser.add_argument('--output', type=Path, default=Path('MossImage.jpg'), help='numbered output base; never overwritten')
	parser.add_argument('--width', type=int, default=720)
	parser.add_argument('--height', type=int, default=1280)
	parser.add_argument('--seed', type=int, default=None, help='random by default; fixes prompt choices and noise')
	parser.add_argument('--guidance', type=float, default=1.0)
	parser.add_argument('--quality', type=int, default=85)
	parser.add_argument('--vae', default='flux2-vae.safetensors')
	parser.add_argument('--te', default='qwen3.5_9b_int8_convrot.safetensors')
	default_comfy = next((Path(drive + ':/ComfyUI_windows_portable/ComfyUI') for drive in ('D', 'C')
		if Path(drive + ':/ComfyUI_windows_portable/ComfyUI').is_dir()), Path('ComfyUI'))
	parser.add_argument('--comfyui', type=Path, default=Path(os.environ.get('COMFYUI', default_comfy)))
	parser.add_argument('--cache', type=Path, default=ROOT / 'artifacts' / 'moss')
	parser.add_argument('--python', type=Path, help='Python interpreter for the native launcher; use ComfyUI Python')
	args = parser.parse_args(argv)
	if args.steps < 1 or min(args.width, args.height) < 16 or args.width % 16 or args.height % 16:
		parser.error('steps must be positive; width/height must be positive multiples of 16')
	if not 0 <= args.quality <= 100 or not math.isfinite(args.guidance):
		parser.error('quality must be 0-100 and guidance finite')
	if args.seed is not None and not 0 <= args.seed < 2**63:
		parser.error('seed must be between 0 and 2^63-1')
	if args.output.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
		parser.error('output extension must be .jpg, .jpeg or .png')
	return args


def located(name, folder):
	path = Path(name).expanduser()
	for candidate in (path, folder / path):
		if candidate.is_file():
			return candidate.resolve()
	raise FileNotFoundError(f'File not found: {name} (also checked {folder})')


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


def cache_key(data):
	return hashlib.sha256(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()


def encoded(request, output):
	os.environ['COMFYUI'] = request['comfyui']
	sys.argv = ['dump.py', 'moss']
	import dump
	from safetensors.torch import save_file
	clip = dump.comfy.sd.load_clip([request['te']],
		embedding_directory=dump.folder_paths.get_folder_paths('embeddings'), clip_type=dump.comfy.sd.CLIPType.MOSS_IMAGE)
	context = clip.encode_from_tokens_scheduled(clip.tokenize(request['prompt']))[0][0].detach().cpu().float().contiguous()
	save_file({'context': context}, str(output), metadata={'prompt': request['prompt']})


def conditioning(args, prompt, encoder):
	request = {'prompt': prompt, 'te': str(encoder), 'comfyui': str(args.comfyui.resolve())}
	key = cache_key([1, request, identity(encoder), identity(args.comfyui / 'comfy' / 'sd.py')])
	path = args.cache / ('context-' + key + '.safetensors')
	if not path.exists():
		print('Encoding prompt...', flush=True)
		with tempfile.TemporaryDirectory(dir=args.cache) as temporary:
			folder = Path(temporary)
			(folder / 'request.json').write_text(json.dumps(request), encoding='utf-8')
			subprocess.run([sys.executable, str(ROOT / 'moss.py'), '_encode', str(folder / 'request.json'),
				str(folder / 'context.safetensors')], check=True)
			os.replace(folder / 'context.safetensors', path)
	else:
		print('Using cached prompt conditioning', flush=True)
	return path


def merged(args, model, loras):
	active = [(path, strength) for path, strength in loras if strength != 0]
	if not active:
		return model
	key = cache_key([2, identity(ROOT / 'merge.py'), identity(model), [(identity(path), strength) for path, strength in active]])
	path = args.cache / ('model-' + key + '.safetensors')
	if not path.exists():
		print('Merging weighted LoRA stack once; subsequent calls reuse the cached model', flush=True)
		with tempfile.TemporaryDirectory(dir=args.cache) as temporary:
			output = Path(temporary) / 'model.safetensors'
			subprocess.run([sys.executable, str(ROOT / 'merge.py'), str(model), str(output),
				*[str(path) + ':' + repr(strength) for path, strength in active], '--split-qkv'], check=True)
			os.replace(output, path)
	else:
		print('Using cached LoRA stack', flush=True)
	return path


def generated(args, model, vae, context, seed, folder):
	import torch
	from safetensors.torch import load_file, save_file
	data = load_file(str(context))
	data['noise'] = torch.randn((1, 128, args.height // 16, args.width // 16),
		generator=torch.Generator('cpu').manual_seed(seed))
	for name, value in {'steps': args.steps, 'guidance': args.guidance, 'warmup': 5, 'period': 2,
		'jpeg': int(args.output.suffix.lower() != '.png'), 'quality': args.quality}.items():
		data[name] = torch.tensor([value], dtype=torch.float32)
	save_file(data, str(folder / 'generation.safetensors'))
	environment = dict(os.environ, STRICT_GPU='all', CUDA_CACHE_PATH=str(args.cache / 'cuda'), CUDA_CACHE_MAXSIZE='4294967296')
	(args.cache / 'cuda').mkdir(exist_ok=True)
	subprocess.run([str(ROOT / 'MossGenerate.exe'), str(folder), str(model), '', str(vae), 'off', ''],
		cwd=ROOT, env=environment, check=True)
	return folder / ('strict-off.png' if args.output.suffix.lower() == '.png' else 'strict-off.jpg')


def main(argv=None):
	args = arguments(argv)
	started = time.perf_counter()
	models = args.comfyui / 'models'
	model = located(args.model, models / 'diffusion_models')
	validate_model(model)
	vae = located(args.vae, models / 'vae')
	encoder = located(args.te, models / 'text_encoders')
	loras = [(located(path, models / 'loras'), strength) for path, strength in map(lora_spec, args.lora)]
	if not (ROOT / 'MossGenerate.exe').is_file():
		raise FileNotFoundError('Build MossGenerate.strict first: ../Strict/strict.exe build MossGenerate.strict')
	seed = args.seed if args.seed is not None else secrets.randbits(32)
	prompt = expand_prompt(args.prompt.read_text(encoding='utf-8-sig').strip(), random.Random(seed))
	print(f'Seed: {seed}\nPrompt: {prompt}', flush=True)
	args.cache = args.cache.resolve()
	args.cache.mkdir(parents=True, exist_ok=True)
	context = conditioning(args, prompt, encoder)
	weights = merged(args, model, loras)
	output, handle = reserve_output(args.output.resolve())
	try:
		with handle, tempfile.TemporaryDirectory(dir=args.cache) as temporary:
			image = generated(args, weights, vae, context, seed, Path(temporary))
			with image.open('rb') as source:
				shutil.copyfileobj(source, handle)
		metadata = dict(prompt=prompt, seed=seed, model=str(model), loras=[(str(path), strength) for path, strength in loras],
			steps=args.steps, width=args.width, height=args.height, guidance=args.guidance, vae=str(vae), te=str(encoder),
			quality=args.quality if output.suffix.lower() != '.png' else None, seconds=time.perf_counter() - started)
		with output.with_suffix(output.suffix + '.json').open('x', encoding='utf-8') as sidecar:
			json.dump(metadata, sidecar, indent=2, ensure_ascii=False)
	except BaseException:
		handle.close()
		output.unlink(missing_ok=True)
		raise
	print(f'Saved {output}\nTotal {time.perf_counter() - started:.2f} s', flush=True)
	return 0


if __name__ == '__main__':
	try:
		if len(sys.argv) > 1 and sys.argv[1] == '_encode':
			encoded(json.loads(Path(sys.argv[2]).read_text(encoding='utf-8')), Path(sys.argv[3]))
		else:
			raise SystemExit(main())
	except (OSError, ValueError, subprocess.CalledProcessError) as error:
		print('Moss: ' + str(error), file=sys.stderr)
		raise SystemExit(1)
