"""ComfyUI nodes: Moss images (MossGenerate.exe) and H3 videos with sound (H3Video.exe) sampled by Strict from ComfyUI's
conditioning and latent (Moss also its model and LoRAs), decoded by ComfyUI."""
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import tempfile
import torch
from safetensors.torch import load, save_file
import comfy.model_management
import comfy.nested_tensor
from comfy.taesd.taesd import Block, Clamp, conv
import comfy.utils
import folder_paths
from server import PromptServer
from . import generate

RESIDENT, RESIDENT_H3 = generate.Resident(), generate.Resident()
TINY, PREVIEWS = {}, ThreadPoolExecutor(1)
WEB_DIRECTORY = './web'


def checkpoint(model):
	"""The checkpoint file of a MODEL from Load Diffusion Model."""
	init = model.cached_patcher_init
	if init is None or init[0].__name__ != 'load_diffusion_model':
		raise ValueError('Load the model with Load Diffusion Model (UNETLoader)')
	return Path(init[1][0])


def released(model, prompt, node):
	"""Moves the weights of model to the meta device when only Strict samplers read it (generate.releasable): Strict loads
	the checkpoint itself, names and shapes stay for LoRA loaders that run again (19 GB less for an H3 checkpoint)."""
	if model is not None and prompt and generate.releasable(prompt, str(node)):
		model.model.to('meta')


def tiny_decoder():
	"""models/vae_approx/taeh3.safetensors on the GPU, a TAESD decoder of single frames: its keys are positional modules,
	the gaps are the input Clamp, the ReLU after the first conv and upsamples. None without the file."""
	path = folder_paths.get_full_path('vae_approx', 'taeh3.safetensors')
	if path and path not in TINY:
		weights = comfy.utils.load_torch_file(path, safe_load=True)
		layers = []
		for index in range(1 + max(int(key.split('.')[0]) for key in weights)):
			block, plain = weights.get(f'{index}.conv.0.weight'), weights.get(f'{index}.weight')
			layers.append(Block(block.shape[1], block.shape[0]) if block is not None else
				conv(plain.shape[1], plain.shape[0], bias=f'{index}.bias' in weights) if plain is not None else
				Clamp() if index == 0 else torch.nn.ReLU() if index == 2 else torch.nn.Upsample(scale_factor=2))
		decoder = torch.nn.Sequential(*layers)
		decoder.load_state_dict(weights)
		TINY[path] = decoder.eval().to(comfy.model_management.get_torch_device(), torch.float16).to(memory_format=torch.channels_last)
	return TINY.get(path)


def decoded(decoder, latents):
	"""taeh3 of latent frames at their own size (generate.preview_batch of them, 0.6 GB of GPU memory at 480x832 that
	stays cached for the next batch: Windows pages instead of failing an allocation past H3Video's budget)."""
	with torch.inference_mode():
		return decoder(latents.to(decoder[1].weight.device).half().contiguous(memory_format=torch.channels_last))


def previewed(decoder, numbers, video, node):
	"""Shows the denoised video H3Video.exe wrote for a step on the node (web/h3_preview.js)."""
	image = generate.h3_preview(numbers, video, lambda latents: decoded(decoder, latents))
	PromptServer.instance.send_sync('strict_h3_preview', {'node': node, 'image': base64.b64encode(image).decode()},
		PromptServer.instance.client_id)


class StrictMossSampler:
	@classmethod
	def INPUT_TYPES(cls):
		return {'required': {
			'model': ('MODEL', {'tooltip': 'Load Diffusion Model with a Moss checkpoint (int8 Convrot, fp8 or bf16, others are converted '
				'once), optionally through LoraLoaderModelOnly nodes'}),
			'positive': ('CONDITIONING',),
			'latent_image': ('LATENT', {'tooltip': 'Empty Flux 2 Latent: width, height and batch size'}),
			'vae': ('VAE',),
			'seed': ('INT', {'default': 0, 'min': 0, 'max': 2 ** 63 - 1, 'control_after_generate': True}),
			'steps': ('INT', {'default': 8, 'min': 1, 'max': 200, 'tooltip': 'quality custom only: 8 or fewer add the DMD turbo LoRA, '
				'use cfg 1 then, without it (turbo off and no turbo LoraLoader) 25-50 steps with cfg 4'}),
			'cfg': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 30.0, 'step': 0.1, 'round': 0.01,
				'tooltip': 'quality custom only: above 1 guides away from an empty (zero) prompt, every step then runs the model twice'}),
			'spectrum': ('STRING', {'default': 'off', 'tooltip': 'quality custom only: off or spectrum<steps>_w<warmup>p<period>, '
				'like spectrum8_w3p3: forecasts steps'}),
			'lora_mode': (['merged', 'runtime'], {'tooltip': 'merged: cached int8 merge of the LoRAs (fast), runtime: floating adapters'}),
			'turbo': (['auto', 'off'], {'tooltip': 'auto: adds loras/moss/dmd_turbo_lora.safetensors at 8 steps or fewer unless a LoraLoader or '
				'the checkpoint already has it, off for checkpoints with another turbo LoRA baked in'}),
			'quality': (list(generate.MOSS_QUALITIES) + ['custom'], {'tooltip': 'presets of steps, cfg and spectrum (sampling '
				'times for 720x1280 on a 3080): ' + '; '.join(name + ': ' + preset[-1] for name, preset in generate.MOSS_QUALITIES.items()) +
				'; custom: the steps, cfg and spectrum above'})},
			'hidden': {'unique_id': 'UNIQUE_ID', 'prompt': 'PROMPT'}}
	RETURN_TYPES = ('IMAGE',)
	FUNCTION = 'sample'
	CATEGORY = 'sampling/strict'
	DESCRIPTION = ('Euler sampling of Moss in Strict (StrictComfy MossGenerate.exe), the Strict process keeps the model loaded '
		'for the next run with the same model, LoRAs and Spectrum')

	def sample(self, model, positive, latent_image, vae, seed, quality, steps, cfg, spectrum, lora_mode, turbo, unique_id=None, prompt=None):
		checkpointed = checkpoint(model)
		metadata = generate.validate_model(checkpointed)
		if len(positive) != 1:
			raise ValueError('positive must be a single conditioning')
		batch, channels, height, width = latent_image['samples'].shape
		if channels != 128:
			raise ValueError('latent_image must be a Flux 2 latent with 128 channels (Empty Flux 2 Latent)')
		try:
			chosen = ['--steps', str(steps), '--guidance', repr(cfg), '--spectrum', spectrum] if quality == 'custom' else ['--quality', quality]
			args = generate.arguments(chosen + ['--lora_mode', lora_mode, '--width', str(width * 16), '--height', str(height * 16)])
		except SystemExit:
			raise ValueError('Invalid StrictMossSampler settings, see the console') from None
		args.cache.mkdir(parents=True, exist_ok=True)
		patches = generate.patched_adapter(args.cache, model.patches)
		turbos = [] if turbo == 'off' else generate.selected_loras(args, Path(folder_paths.models_dir), metadata)
		loras = [(path, strength) for path, strength in turbos if not generate.turbo_patched(model.patches, path)]
		loras += [(patches, 1.0)] if patches else []
		released(model, prompt, unique_id)
		loaded, adapter = generate.lora_model(args, checkpointed, loras)
		comfy.model_management.unload_all_models()
		comfy.model_management.soft_empty_cache()
		progress = comfy.utils.ProgressBar(batch)
		latents = []
		with tempfile.TemporaryDirectory(dir=args.cache) as temporary:
			folder = Path(temporary)
			save_file({'context': positive[0][0].float().cpu().contiguous()}, str(folder / 'context.safetensors'))
			plan = [{'seed': (seed + index) % 2 ** 63} for index in range(batch)]
			with closing(RESIDENT.generated(args, loaded, '', [folder / 'context.safetensors'] * batch, plan, folder, adapter)) as outputs:
				for index, path, milliseconds in outputs:
					latents.append(load(path.read_bytes())['latent'].reshape(1, channels, height, width))
					progress.update(1)
					comfy.model_management.throw_exception_if_processing_interrupted()
		return (vae.decode(torch.cat(latents)),)


class StrictH3Sampler:
	@classmethod
	def INPUT_TYPES(cls):
		return {'required': {
			'positive': ('CONDITIONING', {'tooltip': 'MiniMax H3 Image to Video (text to video, or with first and last frames)'}),
			'latent': ('LATENT', {'tooltip': 'The video and audio latent of MiniMax H3 Image to Video or Empty MiniMax H3 AV '
				'Latent: width, height, length and batch size'}),
			'quality': (list(generate.QUALITIES), {'default': 'fast', 'tooltip': 'Model, steps, attention and Spectrum; sampling '
				'times for 124 frames on a 4090: ' + '; '.join(name + ': ' + preset[-1] for name, preset in generate.QUALITIES.items())}),
			'seed': ('INT', {'default': 0, 'min': 0, 'max': 2 ** 63 - 1, 'control_after_generate': True}),
			'skip': (list(generate.SKIPS), {'default': '5:45', 'tooltip': 'Blocks first:last (of 50) that every other '
				'evaluated step after the first two reuses from the last full step instead of computing them (changes the '
				'result), faster for 8 routed steps of a 10 s clip on a 4090: ' +
				', '.join(f'{name} {faster} %' for name, faster in generate.SKIPS.items())})},
			'optional': {'sampler_name': (generate.SAMPLERS, {'default': 'euler', 'tooltip': 'res_multistep is second order '
				'(the default of the H3 workflows), er_sde adds new noise every step; the same speed for each'}),
				'scheduler': (generate.SCHEDULERS, {'default': 'simple'}),
				'model': ('MODEL', {'tooltip': 'Load Diffusion Model (optionally through LoraLoaderModelOnly) instead '
					'of the model of the quality, the LoRAs of the quality still apply'})},
			'hidden': {'unique_id': 'UNIQUE_ID', 'prompt': 'PROMPT'}}
	RETURN_TYPES = ('LATENT',)
	FUNCTION = 'sample'
	CATEGORY = 'sampling/strict'
	DESCRIPTION = ('Samples H3 video and sound in Strict (StrictComfy H3Video.exe) instead of SamplerCustomAdvanced: decode '
		'the latent with VAE Decode and VAE Decode Audio. The Strict process keeps the model loaded for the next run with '
		'the same quality and model.')

	def sample(self, positive, latent, quality, seed, skip='5:45', sampler_name='euler', scheduler='simple', model=None,
			unique_id=None, prompt=None):
		samples = latent['samples']
		if not getattr(samples, 'is_nested', False) or len(samples.tensors) != 2:
			raise ValueError('latent must be an H3 video and audio latent (MiniMax H3 Image to Video)')
		if len(positive) != 1:
			raise ValueError('positive must be a single conditioning')
		video, audio = (tuple(tensor.shape) for tensor in samples.tensors)
		args = generate.h3_arguments(quality, video, audio, skip, sampler_name, scheduler)
		args.cache.mkdir(parents=True, exist_ok=True)
		models = Path(folder_paths.models_dir)
		preset = Path(args.model).name
		chosen = checkpoint(model) if model else Path(folder_paths.get_full_path_or_raise('diffusion_models', preset))
		patches = None if model is None else generate.patched_adapter(args.cache, model.patches)
		loras = generate.selected_loras(args, models, generate.validate_model(chosen, 'h3')) + ([(patches, 1.0)] if patches else [])
		released(model, prompt, unique_id)
		loaded, adapter = generate.lora_model(args, chosen, loras)
		comfy.model_management.unload_all_models()
		comfy.model_management.soft_empty_cache()
		decoder = tiny_decoder()
		if decoder:
			decoded(decoder, torch.zeros(generate.preview_batch(video), *video[1:2], *video[3:]))
		progress, preview = comfy.utils.ProgressBar(video[0] * args.steps), generate.skipping(PREVIEWS)
		def stepped(line):
			progress.update(1)
			path = folder / f'generation-{len(pairs)}-step{line.split()[1]}.safetensors'
			if decoder and path.exists():
				preview(previewed, decoder, load(path.read_bytes())['video'], video, unique_id)
			comfy.model_management.throw_exception_if_processing_interrupted()
		pairs = []
		with tempfile.TemporaryDirectory(dir=args.cache) as temporary:
			folder = Path(temporary)
			context = folder / 'context.safetensors'
			condition = {name: value.detach().cpu().float().contiguous() for name, value in generate.h3_condition(positive).items()}
			save_file(condition | ({'preview': torch.ones(1)} if decoder else {}), str(context))
			plan = [{'seed': (seed + index) % 2 ** 63} for index in range(video[0])]
			with closing(RESIDENT_H3.generated(args, loaded, '', [context] * video[0], plan, folder, adapter, stepped)) as outputs:
				for index, path, milliseconds in outputs:
					pairs.append(generate.h3_latents(load(path.read_bytes())['latent'], video, audio))
		joined = [torch.cat([pair[stream] for pair in pairs]) for stream in (0, 1)]
		return ({'samples': comfy.nested_tensor.NestedTensor(joined)},)


NODE_CLASS_MAPPINGS = {'StrictMossSampler': StrictMossSampler, 'StrictH3Sampler': StrictH3Sampler}
NODE_DISPLAY_NAME_MAPPINGS = {'StrictMossSampler': 'Strict Moss Sampler', 'StrictH3Sampler': 'Strict H3 Sampler'}
