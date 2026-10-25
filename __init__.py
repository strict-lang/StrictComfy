"""ComfyUI nodes: Moss images sampled by Strict (MossGenerate.exe) from ComfyUI's model, LoRAs, conditioning and latent,
decoded with ComfyUI's VAE."""
from contextlib import closing
from pathlib import Path
import tempfile
import torch
from safetensors.torch import load, save_file
import comfy.model_management
import comfy.utils
import folder_paths
from . import generate

RESIDENT = generate.Resident()


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
			'steps': ('INT', {'default': 8, 'min': 1, 'max': 200, 'tooltip': '8 with the DMD turbo LoRA and cfg 1, '
				'without it (turbo off and no turbo LoraLoader) 25-50 steps with cfg 4'}),
			'cfg': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 30.0, 'step': 0.1, 'round': 0.01,
				'tooltip': 'Above 1 guides away from an empty (zero) prompt, every step then runs the model twice'}),
			'spectrum': ('STRING', {'default': 'off', 'tooltip': 'off or spectrum<steps>_w<warmup>p<period>, like spectrum8_w3p3: forecasts steps'}),
			'lora_mode': (['merged', 'runtime'], {'tooltip': 'merged: cached int8 merge of the LoRAs (fast), runtime: floating adapters'}),
			'turbo': (['auto', 'off'], {'tooltip': 'auto: adds loras/moss/dmd_turbo_lora.safetensors at 8 steps unless a LoraLoader or '
				'the checkpoint already has it, off for checkpoints with another turbo LoRA baked in'})}}
	RETURN_TYPES = ('IMAGE',)
	FUNCTION = 'sample'
	CATEGORY = 'sampling/strict'
	DESCRIPTION = ('Euler sampling of Moss in Strict (StrictComfy MossGenerate.exe), the Strict process keeps the model loaded '
		'for the next run with the same model, LoRAs and Spectrum')

	def sample(self, model, positive, latent_image, vae, seed, steps, cfg, spectrum, lora_mode, turbo):
		init = model.cached_patcher_init
		if init is None or init[0].__name__ != 'load_diffusion_model':
			raise ValueError('Load the Moss model with Load Diffusion Model (UNETLoader)')
		checkpoint = Path(init[1][0])
		metadata = generate.validate_model(checkpoint)
		if len(positive) != 1:
			raise ValueError('positive must be a single conditioning')
		batch, channels, height, width = latent_image['samples'].shape
		if channels != 128:
			raise ValueError('latent_image must be a Flux 2 latent with 128 channels (Empty Flux 2 Latent)')
		try:
			args = generate.arguments(['--steps', str(steps), '--guidance', repr(cfg), '--spectrum', spectrum,
				'--lora_mode', lora_mode, '--width', str(width * 16), '--height', str(height * 16)])
		except SystemExit:
			raise ValueError('Invalid StrictMossSampler settings, see the console') from None
		args.cache.mkdir(parents=True, exist_ok=True)
		patches = generate.patched_adapter(args.cache, model.patches)
		turbos = [] if turbo == 'off' else generate.selected_loras(args, Path(folder_paths.models_dir), metadata)
		loras = [(path, strength) for path, strength in turbos if not generate.turbo_patched(model.patches, path)]
		loras += [(patches, 1.0)] if patches else []
		loaded = generate.cached_merge(args, checkpoint, loras if lora_mode == 'merged' else [])
		adapter = '' if lora_mode == 'merged' else generate.runtime_adapter(args, loaded, loras)
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


NODE_CLASS_MAPPINGS = {'StrictMossSampler': StrictMossSampler}
NODE_DISPLAY_NAME_MAPPINGS = {'StrictMossSampler': 'Strict Moss Sampler'}
