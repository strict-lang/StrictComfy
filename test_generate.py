"""Checks of the ComfyUI nodes' side (generate.py, merge.py) and of the executables' options; run with ComfyUI's Python.
No full model required."""
import importlib.util
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


class GenerationCommandTests(unittest.TestCase):
	def module(self):
		self.assertIsNotNone(importlib.util.find_spec('generate'), 'Local generation command is missing')
		import generate
		return generate

	def tensors(self, path):
		from safetensors.torch import load
		return load(Path(path).read_bytes())

	def test_comfyui_nodes_compile(self):
		compile((ROOT / '__init__.py').read_text(encoding='utf-8'), '__init__.py', 'exec')

	def test_executable_help(self):
		result = subprocess.run([self.module().executable('moss'), '--help'], capture_output=True, text=True)
		self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
		self.assertIn('--prompt', result.stdout)
		self.assertIn('--lora', result.stdout)
		self.assertIn('--skip', result.stdout)
		self.assertNotIn('-h,', result.stdout)
		self.assertIn('--quality', result.stdout)

	def test_defaults(self):
		args = self.module().arguments([])
		self.assertEqual((args.width, args.height, args.steps, args.guidance, args.lora_mode, args.spectrum, args.lora),
			(720, 1280, 8, 1, 'merged', 'off', []))
		self.assertEqual(args.cache, ROOT / 'artifacts' / 'moss')
		with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
			self.module().arguments(['--quality'])

	def test_spectrum_preset_reaches_backend(self):
		module = self.module()
		with redirect_stderr(io.StringIO()):
			try:
				args = module.arguments(['--spectrum', 'spectrum8_w3p3'])
			except SystemExit:
				self.fail('Spectrum preset is not exposed by the CLI')
		self.assertEqual(module.spectrum_config(args), ('on', 3, 3))

	def test_cached_merge_runs_once_and_certified_model_avoids_double_turbo(self):
		from unittest.mock import patch
		import torch
		from safetensors.torch import save_file
		module = self.module()
		self.assertTrue(hasattr(module, 'cached_merge'), 'Cached model merge is missing')
		with tempfile.TemporaryDirectory() as temporary:
			folder = Path(temporary)
			base, lora = folder / 'base.safetensors', folder / 'dmd_turbo_lora.safetensors'
			base.write_bytes(b'original checkpoint')
			lora.write_bytes(b'adapter identity')
			args = module.arguments(['--cache', str(folder)])
			marker = {'algorithm': 'convrot-int8-stochastic-v1', 'loras': [[str(lora), 1.0]]}
			def build(command, **kwargs):
				self.assertEqual(Path(command[2]), base)
				save_file({'fixture': torch.zeros(1)}, command[3], metadata={'strict_lora_merge': json.dumps(marker)})
			with patch.object(module.subprocess, 'run', side_effect=build) as run:
				merged = module.cached_merge(args, base, [(lora, 1)])
				self.assertTrue(merged.is_file())
				self.assertEqual(module.cached_merge(args, base, [(lora, 1)]), merged)
				self.assertEqual(run.call_count, 1)
			self.assertEqual(base.read_bytes(), b'original checkpoint')
			self.assertEqual(module.selected_loras(args, folder, {'strict_lora_merge': json.dumps(marker)}), [])

	def test_cached_merge_keeps_the_three_newest_models(self):
		import os
		from unittest.mock import patch
		module = self.module()
		with tempfile.TemporaryDirectory() as temporary:
			folder = Path(temporary)
			old = [folder / f'merged-stochastic-v1-{index}.safetensors' for index in range(4)]
			for index, path in enumerate(old):
				path.touch()
				os.utime(path, (index, index))
			(folder / 'base.safetensors').touch()
			args = module.arguments(['--cache', str(folder)])
			with patch.object(module.subprocess, 'run', side_effect=lambda command, **kwargs: Path(command[3]).touch()):
				merged = module.cached_merge(args, folder / 'base.safetensors', [(folder / 'base.safetensors', 1)])
			self.assertEqual(sorted(folder.glob('merged-*')), sorted([merged, *old[2:]]))

	def test_merge_keeps_sub_quantization_updates(self):
		import torch
		from safetensors import safe_open
		from safetensors.torch import save_file
		with tempfile.TemporaryDirectory() as temporary:
			folder = Path(temporary)
			name = 'layers.0.proj.q'
			weight = torch.randint(-127, 128, (64, 256), generator=torch.Generator().manual_seed(12), dtype=torch.int8)
			weight[:, 0] = 127
			weight[:, -1] = -127
			quant = json.dumps({'convrot': True, 'convrot_groupsize': 256, 'format': 'int8_tensorwise'})
			save_file({name + '.weight': weight, name + '.weight_scale': torch.full((64, 1), 0.01),
				name + '.comfy_quant': torch.tensor(list(quant.encode()), dtype=torch.uint8)}, str(folder / 'base.safetensors'))
			prefix = 'diffusion_model.' + name + '.'
			save_file({prefix + 'lora_down.weight': torch.ones((1, 256)),
				prefix + 'lora_up.weight': torch.full((64, 1), 1e-4)}, str(folder / 'lora.safetensors'))
			result = subprocess.run([sys.executable, str(ROOT / 'merge.py'), str(folder / 'base.safetensors'),
				str(folder / 'merged.safetensors'), str(folder / 'lora.safetensors')], capture_output=True, text=True)
			self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
			merged = self.tensors(folder / 'merged.safetensors')
			self.assertGreater((merged[name + '.weight'] != weight).sum().item(), 10)
			with safe_open(folder / 'merged.safetensors', framework='pt') as file:
				self.assertIn('strict_lora_merge', file.metadata())

	def test_fp8_fused_checkpoint_converts_to_strict_layout(self):
		import torch
		from safetensors.torch import save_file
		module = self.module()
		with tempfile.TemporaryDirectory() as temporary:
			folder = Path(temporary)
			prefix = 'model.diffusion_model.layers.0.'
			thirds = [torch.randn((64, 256), generator=torch.Generator().manual_seed(index)) * (index + 1) for index in range(3)]
			save_file({prefix + 'proj.qkv.weight': torch.cat(thirds).to(torch.float8_e4m3fn),
				prefix + 'norm1.weight': torch.ones(256).to(torch.float8_e4m3fn)}, str(folder / 'fp8.safetensors'))
			self.assertEqual(module.validate_model(folder / 'fp8.safetensors'), {})
			args = module.arguments(['--cache', str(folder)])
			converted = self.tensors(module.cached_merge(args, folder / 'fp8.safetensors', []))
			self.assertEqual(set(converted), {'layers.0.norm1.weight'} | {'layers.0.proj.' + name + '.' + part
				for name in 'qkv' for part in ('weight', 'weight_scale', 'comfy_quant')})
			self.assertEqual(converted['layers.0.norm1.weight'].dtype, torch.bfloat16)
			for name, third in zip('qkv', thirds):
				weight = converted['layers.0.proj.' + name + '.weight'].float() * converted['layers.0.proj.' + name + '.weight_scale']
				expected = third.to(torch.float8_e4m3fn).float().norm(dim=1)
				self.assertTrue(torch.allclose(weight.norm(dim=1), expected, rtol=0.02), name)

	def test_batch_launches_backend_once_with_distinct_inputs(self):
		from unittest.mock import MagicMock, patch
		import torch
		from safetensors.torch import save_file
		module = self.module()
		with tempfile.TemporaryDirectory() as temporary:
			folder = Path(temporary)
			args = module.arguments(['--width', '16', '--height', '16', '--cache', str(folder), '--spectrum', 'spectrum8_w3p3'])
			contexts = []
			for index in range(2):
				path = folder / f'context{index}.safetensors'
				save_file({'context': torch.full((1, 2, 3), float(index))}, str(path))
				contexts.append(path)
			process = MagicMock()
			process.stdout = io.StringIO('evaluation 0 at 1.234\ngeneration 0 in 20 ms\ngeneration 1 in 21 ms\n')
			process.wait.return_value = 0
			process.poll.return_value = 0
			with patch.object(module.subprocess, 'Popen', return_value=process) as launch, redirect_stdout(io.StringIO()) as output:
				results = list(module.Resident().generated(args, Path('model'), contexts, [{'seed': 42}, {'seed': 7}], folder))
			self.assertEqual(launch.call_count, 1)
			self.assertIn('step 0 at 1.234', output.getvalue())
			self.assertNotIn('evaluation ', output.getvalue())
			self.assertEqual(launch.call_args.args[0], [module.executable('moss'), '-', 'model', '', 'on'])
			self.assertEqual([result[1].name for result in results], ['strict-on-0-latent.safetensors', 'strict-on-1-latent.safetensors'])
			self.assertEqual([result[0] for result in results], [0, 1])
			self.assertEqual(self.tensors(folder / 'generation.safetensors')['jobs'].item(), 2)
			first = self.tensors(folder / 'generation-0.safetensors')
			self.assertEqual((first['warmup'].item(), first['period'].item()), (3, 3))
			second = self.tensors(folder / 'generation-1.safetensors')
			self.assertFalse(torch.equal(first['noise'], second['noise']))
			self.assertFalse(torch.equal(first['context'], second['context']))

	def test_resident_backend_keeps_one_process_per_settings(self):
		from unittest.mock import MagicMock, patch
		import torch
		from safetensors.torch import save_file
		module = self.module()
		with tempfile.TemporaryDirectory() as temporary:
			folder = Path(temporary)
			args = module.arguments(['--width', '16', '--height', '16', '--cache', str(folder)])
			save_file({'context': torch.zeros((1, 2, 3))}, str(folder / 'context.safetensors'))
			process = MagicMock()
			process.stdout = io.StringIO('loaded in 1 s\ngeneration 0 in 20 ms\ngeneration 0 in 21 ms\n')
			process.poll.return_value = None
			resident = module.Resident()
			with patch.object(module.subprocess, 'Popen', return_value=process) as launch, redirect_stdout(io.StringIO()):
				for run in range(3):
					(folder / str(run)).mkdir()
				for run in range(2):
					results = list(resident.generated(args, Path('model'), [folder / 'context.safetensors'],
						[{'seed': run}], folder / str(run)))
					self.assertEqual([(index, path.name, milliseconds) for index, path, milliseconds in results],
						[(0, 'strict-off-latent.safetensors', 20 + run)])
				self.assertEqual(launch.call_count, 1)
				self.assertEqual(launch.call_args.args[0][1], '-')
				self.assertEqual([call.args[0] for call in process.stdin.write.call_args_list],
					[str(folder / '0') + '\n', str(folder / '1') + '\n'])
				args.spectrum = 'spectrum8_w3p3'
				launch.return_value = MagicMock(stdout=io.StringIO('generation 0 in 22 ms\n'))
				launch.return_value.poll.return_value = None
				list(resident.generated(args, Path('model'), [folder / 'context.safetensors'], [{'seed': 2}], folder / '2'))
				self.assertEqual(launch.call_count, 2)
				process.terminate.assert_called_once()

	def test_comfyui_lora_patches_become_one_cached_adapter(self):
		from types import SimpleNamespace
		from unittest.mock import patch
		import torch
		module = self.module()
		up, down = torch.randn(4, 2), torch.randn(2, 3)
		other_up, other_down = torch.randn(4, 1), torch.randn(1, 3)
		lora = lambda weights: SimpleNamespace(name='lora', weights=(*weights, None, None, None))
		patches = {'diffusion_model.layers.0.proj.q.weight': [(0.5, lora((up, down, 4.0)), 1.0, None, None),
			(2.0, lora((other_up, other_down, None)), 1.0, None, None)]}
		with tempfile.TemporaryDirectory() as temporary:
			path = module.patched_adapter(Path(temporary), patches)
			with patch.object(module.hashlib, 'sha256', side_effect=AssertionError('the same patches hashed again')):
				self.assertEqual(module.patched_adapter(Path(temporary), patches), path)
			tensors = self.tensors(path)
			prefix = 'diffusion_model.layers.0.proj.q.'
			self.assertEqual(tensors[prefix + 'alpha'].item(), 3)
			expected = 0.5 * 4 / 2 * up @ down + 2 * other_up @ other_down
			torch.testing.assert_close(tensors[prefix + 'lora_up.weight'] @ tensors[prefix + 'lora_down.weight'], expected)
			patches['diffusion_model.layers.0.proj.q.weight'][0] = (1.0, lora((up, down, 4.0)), 1.0, (0, 0, 2), None)
			with self.assertRaises(ValueError):
				module.patched_adapter(Path(temporary), patches)

	def test_a_model_only_strict_reads_can_be_released(self):
		module = self.module()
		loaded = {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'h3.safetensors'}}
		lora = {'class_type': 'LoraLoader', 'inputs': {'model': ['1', 0], 'lora_name': 'turbo.safetensors'}}
		strict = {'class_type': 'StrictH3Sampler', 'inputs': {'model': ['2', 0], 'positive': ['5', 0]}}
		prompt = {'1': loaded, '2': lora, '3': strict, '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 1]}}}
		self.assertTrue(module.releasable(prompt, '3'))
		self.assertTrue(module.releasable(prompt | {'6': strict}, '3'))
		self.assertTrue(module.releasable(prompt | {'6': strict | {'class_type': 'StrictH3LatentSampler'}}, '3'))
		self.assertFalse(module.releasable(prompt | {'4': {'class_type': 'BasicGuider', 'inputs': {'model': ['2', 0]}}}, '3'))
		self.assertFalse(module.releasable(prompt | {'4': {'class_type': 'ModelSamplingSD3', 'inputs': {'model': ['1', 0]}}}, '3'))
		self.assertFalse(module.releasable({'1': loaded, '3': {'class_type': 'StrictH3Sampler', 'inputs': {}}}, '3'))

	def test_comfyui_patches_with_turbo_lora_are_detected(self):
		from types import SimpleNamespace
		import torch
		from safetensors.torch import save_file
		module = self.module()
		down = torch.randn(2, 3)
		patch = lambda weights: {'diffusion_model.layers.0.proj.ffn_down.weight':
			[(1.0, SimpleNamespace(name='lora', weights=(torch.randn(4, 2), weights, None, None, None, None)), 1.0, None, None)]}
		with tempfile.TemporaryDirectory() as temporary:
			turbo = Path(temporary) / 'dmd_turbo_lora.safetensors'
			save_file({'diffusion_model.layers.0.proj.ffn_down.lora_down.weight': down}, str(turbo))
			self.assertTrue(module.turbo_patched(patch(down.clone()), turbo))
			self.assertFalse(module.turbo_patched(patch(torch.randn(2, 3)), turbo))
			self.assertFalse(module.turbo_patched({}, turbo))
			h3 = Path(temporary) / 'h3_turbo.safetensors'
			save_file({'blocks.0.attn.out_proj.lora_A.weight': down}, str(h3))
			self.assertTrue(module.turbo_patched({'diffusion_model.blocks.0.attn.out_proj.weight':
				patch(down.clone())['diffusion_model.layers.0.proj.ffn_down.weight']}, h3))

	def test_h3_turbo_lora_is_not_applied_twice(self):
		module = self.module()
		turbo = (Path('loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors'), 1.0)
		merged = Path('Minimax_H3_FL2VA_PRUNED_Turbo_8step_v1.0_768p_comfyui_INT8_.safetensors')
		self.assertEqual(module.unapplied([turbo], Path('minimax_h3_fl2va_pruned_int8_convrot.safetensors'), None), [turbo])
		self.assertEqual(module.unapplied([turbo], merged, None), [])

	def test_wrong_checkpoint_is_rejected(self):
		import torch
		from safetensors.torch import save_file
		module = self.module()
		self.assertTrue(hasattr(module, 'validate_model'), 'Checkpoint validation is missing')
		with tempfile.TemporaryDirectory() as folder:
			path = Path(folder) / 'wrong.safetensors'
			save_file({'weight': torch.zeros((1, 1))}, str(path))
			with self.assertRaisesRegex(ValueError, 'int8 Convrot'):
				module.validate_model(path)

	def test_turbo_selected_up_to_eight_steps_without_duplication(self):
		module = self.module()
		self.assertTrue(hasattr(module, 'selected_loras'), 'Automatic runtime turbo selection is missing')
		with tempfile.TemporaryDirectory() as temporary:
			models = Path(temporary)
			path = models / 'loras/moss/dmd_turbo_lora.safetensors'
			path.parent.mkdir(parents=True)
			path.touch()
			args = module.arguments(['--steps', '8'])
			self.assertEqual(module.selected_loras(args, models), [(path.resolve(), 1)])
			args.steps = 4
			self.assertEqual(module.selected_loras(args, models), [(path.resolve(), 1)])
			for steps in (25, 50):
				args.steps = steps
				self.assertEqual(module.selected_loras(args, models), [])
			args.steps = 8
			args.lora = [str(path) + ':0.5']
			self.assertEqual(module.selected_loras(args, models), [(path.resolve(), 0.5)])

	def test_installed_four_step_lora_replaces_dmd_up_to_four_steps(self):
		module = self.module()
		with tempfile.TemporaryDirectory() as temporary:
			models = Path(temporary)
			dmd, four = (models / 'loras/moss' / name for name in ('dmd_turbo_lora.safetensors', 'moss_image_sft_4step_v2.0.safetensors'))
			dmd.parent.mkdir(parents=True)
			dmd.touch()
			four.touch()
			args = module.arguments(['--steps', '4'])
			self.assertEqual(module.selected_loras(args, models), [(four.resolve(), 1)])
			args.steps = 8
			self.assertEqual(module.selected_loras(args, models), [(dmd.resolve(), 1)])
			args.steps, args.lora = 2, [str(dmd) + ':0.5']
			self.assertEqual(module.selected_loras(args, models), [(dmd.resolve(), 0.5)])

	def test_runtime_stack_preserves_small_deltas_and_base_file(self):
		import torch
		from safetensors.torch import save_file
		module = self.module()
		self.assertTrue(hasattr(module, 'runtime_adapter'), 'Runtime LoRA stacking is missing')
		with tempfile.TemporaryDirectory() as temporary:
			folder = Path(temporary)
			base = folder / 'base.safetensors'
			save_file({'layers.0.proj.' + name + '.weight': torch.ones((2, 4), dtype=torch.int8)
				for name in ('q', 'k', 'v', 'o', 'ffn_up_gate', 'ffn_down')}, str(base))
			original = base.read_bytes()
			prefix = 'diffusion_model.layers.0.proj.q.'
			loras = []
			for index, (down, up) in enumerate((('lora_down', 'lora_up'), ('lora_A', 'lora_B'))):
				path = folder / f'adapter{index}.safetensors'
				save_file({prefix + down + '.weight': torch.ones((1, 4)),
					prefix + up + '.weight': torch.full((2, 1), (index + 1) * 1e-5)}, str(path))
				loras.append((path, (0.5, 0.25)[index]))
			args = module.arguments(['--cache', str(folder)])
			packed = Path(module.runtime_adapter(args, base, loras))
			weights = self.tensors(packed)
			self.assertTrue(torch.allclose(weights[prefix + 'lora_up.weight'] @ weights[prefix + 'lora_down.weight'],
				torch.full((2, 4), 1e-5), atol=1e-8, rtol=1e-4))
			self.assertEqual(weights[prefix + 'alpha'].item(), 2)
			self.assertEqual(base.read_bytes(), original)
			self.assertEqual(module.runtime_adapter(args, base, []), '')
			fused = 'diffusion_model.layers.0.proj.qkv.'
			save_file({fused + 'lora_A.weight': torch.ones((1, 4)),
				fused + 'lora_B.weight': torch.arange(1, 7).float().reshape(6, 1) * 1e-5}, str(loras[1][0]))
			weights = self.tensors(module.runtime_adapter(args, base, loras))
			key = 'diffusion_model.layers.0.proj.k.'
			self.assertTrue(torch.allclose(weights[key + 'lora_up.weight'] @ weights[key + 'lora_down.weight'],
				torch.tensor([[7.5e-6] * 4, [1e-5] * 4]), atol=1e-8, rtol=1e-4))

	def test_launcher_preserves_spaces(self):
		name = 'missing model with spaces.safetensors'
		result = subprocess.run([self.module().executable('moss'), '--model', name],
			cwd=tempfile.gettempdir(), capture_output=True, text=True)
		self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
		self.assertIn('FileNotFound:' + name.replace(' ', '_'), result.stdout)

	def test_linux_backend_is_the_release_build_and_names_its_build_when_missing(self):
		from unittest.mock import patch
		module = self.module()
		args = module.arguments(['--cache', tempfile.gettempdir()])
		with tempfile.TemporaryDirectory() as empty, patch.object(module.sys, 'platform', 'linux'), \
				patch.object(module, 'ROOT', Path(empty)):
			command = module.backend(args, Path('model'), '')
			self.assertEqual(command[0], str(Path(empty) / 'Release' / 'Linux' / 'MossImage'))
			with self.assertRaisesRegex(FileNotFoundError, 'reinstall StrictComfy or build it with: strict build MossImage.strict linux'):
				module.started(args, command)

	@unittest.skipIf(sys.platform == 'win32', 'executable bits exist on Linux only')
	def test_linux_backend_unpacked_without_executable_bit_still_starts(self):
		import os
		from unittest.mock import patch
		module = self.module()
		with tempfile.TemporaryDirectory() as folder:
			backend = Path(folder) / 'MossImage'
			backend.write_bytes(b'')
			backend.chmod(0o644)
			args = module.arguments(['--cache', folder])
			with patch.object(module.subprocess, 'Popen'):
				module.started(args, [str(backend), '-'])
			self.assertTrue(os.access(backend, os.X_OK))

	def test_weighted_lora_paths(self):
		parse = self.module().lora_spec
		for text, path, strength in [('style.safetensors:0.5', 'style.safetensors', 0.5),
			('C:\\models\\style.safetensors:0.31', 'C:\\models\\style.safetensors', 0.31),
			('C:\\models\\style.safetensors', 'C:\\models\\style.safetensors', 1),
			('style.safetensors:-0.5', 'style.safetensors', -0.5)]:
			self.assertEqual(parse(text), (path, strength))
		for text in ('style.safetensors:nan', 'style.safetensors:inf', 'style.safetensors:bad'):
			with self.assertRaises(ValueError):
				parse(text)

	def test_relative_weighted_stack_merges_once(self):
		import torch
		from safetensors.torch import save_file
		with tempfile.TemporaryDirectory() as folder:
			folder = Path(folder)
			name = 'layers.0.proj.q'
			quant = json.dumps({'convrot': True, 'convrot_groupsize': 256, 'format': 'int8_tensorwise'})
			save_file({name + '.weight': torch.ones((2, 256), dtype=torch.int8),
				name + '.weight_scale': torch.ones((2, 1)),
				'video_embedder.weight': torch.zeros((2, 3)),
				name + '.comfy_quant': torch.tensor(list(quant.encode()), dtype=torch.uint8)}, str(folder / 'base.safetensors'))
			prefix = 'diffusion_model.' + name + '.'
			save_file({prefix + 'lora_down.weight': torch.ones((1, 256)),
				prefix + 'lora_up.weight': torch.ones((2, 1)),
				'diffusion_model.video_embedder.lora_down.weight': torch.ones((1, 3)),
				'diffusion_model.video_embedder.lora_up.weight': torch.ones((2, 1))}, str(folder / 'style.safetensors'))
			for label, stack in [('stack', ['style.safetensors:0.5', 'style.safetensors:0.25']),
				('single', ['style.safetensors:0.75'])]:
				result = subprocess.run([sys.executable, str(ROOT / 'merge.py'), 'base.safetensors',
					label + '.safetensors', *stack], cwd=folder, capture_output=True, text=True)
				self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
			stack = self.tensors(folder / 'stack.safetensors')
			single = self.tensors(folder / 'single.safetensors')
			self.assertTrue(torch.equal(stack[name + '.weight'], single[name + '.weight']))
			self.assertTrue(torch.equal(stack[name + '.weight_scale'], single[name + '.weight_scale']))
			self.assertTrue(torch.allclose(stack[name + '.weight'].float() * stack[name + '.weight_scale'],
				torch.full((2, 256), 1.75)))
			self.assertTrue(torch.equal(stack['video_embedder.weight'], torch.full((2, 3), 0.75)))


	def test_h3_executable_help(self):
		result = subprocess.run([self.module().executable('h3'), '--help'], capture_output=True, text=True)
		self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
		for option in ('--prompt', '--lora', '--length', '--count', '--skip', '--sampler', '--scheduler'):
			self.assertIn(option, result.stdout)
		self.assertNotIn('--jpg_quality', result.stdout)

	def test_h3_defaults(self):
		module = self.module()
		args = module.arguments([], 'h3')
		self.assertEqual((args.width, args.height, args.length, args.steps, args.lora_mode), (480, 832, 124, 8, 'runtime'))
		self.assertEqual((args.model, args.lora), ('minimax_h3_fl2va_pruned_int8_convrot.safetensors',
			['minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors:1']))
		self.assertEqual(args.cache, ROOT / 'artifacts' / 'h3')
		self.assertFalse(hasattr(args, 'guidance'))
		with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
			module.arguments(['--length', '4'], 'h3')

	def test_h3_spectrum_preset(self):
		module = self.module()
		self.assertEqual(module.spectrum_config(module.arguments(['--spectrum', 'spectrum8_w1p2'], 'h3')), ('on', 1, 2))
		with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
			module.arguments(['--spectrum', 'spectrum8_w1p2', '--steps', '6'], 'h3')

	def test_moss_skipped_layers(self):
		module = self.module()
		self.assertEqual(module.skipped_blocks(module.arguments([])), (0, 0))
		self.assertEqual(module.skipped_blocks(module.arguments(['--skip', '6:36'])), (6, 36))
		with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
			module.arguments(['--skip', '6:42'])

	def test_moss_quality_presets(self):
		module = self.module()
		presets = {name: module.arguments(['--quality', name]) for name in ('best25steps', 'best50steps', 'fast', 'faster', 'fastest',
			'fastest2steps')}
		presets['best'] = module.arguments([])
		self.assertEqual({name: (args.quality, args.steps, args.guidance, args.spectrum) for name, args in presets.items()}, {
			'best': ('best', 8, 1, 'off'), 'best25steps': ('best25steps', 25, 2, 'off'), 'best50steps': ('best50steps', 50, 2, 'off'),
			'fast': ('fast', 8, 1, 'spectrum8_w3p3'), 'faster': ('faster', 8, 1, 'spectrum8_w2p3'), 'fastest': ('fastest', 4, 1, 'spectrum4_w2p2'),
			'fastest2steps': ('fastest2steps', 2, 1, 'off')})
		changed = module.arguments(['--quality', 'faster', '--steps', '10', '--guidance', '2'])
		self.assertEqual((changed.steps, changed.guidance, changed.spectrum), (10, 2, 'spectrum10_w2p3'))
		with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
			module.arguments(['--quality', 'fastest6steps'])

	def test_h3_quality_presets(self):
		module = self.module()
		best = module.arguments([], 'h3')
		self.assertEqual((best.quality, best.steps, best.routing, best.spectrum, best.lora, best.sampler, best.scheduler),
			('best', 8, 0, 'off', ['minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors:1'], 'euler', 'simple'))
		self.assertEqual(module.skipped_blocks(best), (5, 45))
		self.assertEqual(module.skipped_blocks(module.arguments(['--skip', 'off'], 'h3')), (0, 0))
		for skip in ('45:5', '5:51', '5-45'):
			with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
				module.arguments(['--skip', skip], 'h3')
		fastest = module.arguments(['--quality', 'fastest2steps'], 'h3')
		self.assertEqual((fastest.model, fastest.steps, fastest.routing, fastest.spectrum, fastest.lora),
			('minimax_h3_fl2va_pruned_int8_convrot.safetensors', 2, 1, 'off',
			['h3/minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors:1']))
		faster = module.arguments(['--quality', 'faster', '--steps', '10'], 'h3')
		self.assertEqual((faster.steps, faster.spectrum, faster.routing), (10, 'spectrum10_w1p2', 1))
		self.assertEqual(list(module.QUALITIES), ['best', 'fast', 'faster', 'fastest4steps', 'fastest2steps'])
		with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
			module.arguments(['--quality', 'ultra'], 'h3')

	def test_h3_node_settings_and_latents_follow_comfyui_latent(self):
		import torch
		module = self.module()
		video, audio = (1, 24, 37, 52, 30), (1, 32, 2, 207)
		args = module.h3_arguments('fast', video, audio, '10:40', 'res_multistep', 'sgm_uniform')
		self.assertEqual((args.width, args.height, args.length, args.steps, args.routing, args.skip, args.sampler, args.scheduler),
			(480, 832, 124, 8, 1, '10:40', 'res_multistep', 'sgm_uniform'))
		self.assertEqual(tuple(module.h3_noise(args, 7)['audio'].shape), audio[1:])
		self.assertEqual((args.decode, module.h3_arguments('fast', video, audio, decode=True).decode), (False, True))
		with self.assertRaises(ValueError):
			module.h3_arguments('fast', video, (1, 32, 2, 100))
		numbers = torch.arange(24 * 37 * 52 * 30 + 32 * 2 * 207, dtype=torch.float32)
		latent_video, latent_audio = module.h3_latents(numbers, video, audio)
		self.assertEqual((tuple(latent_video.shape), tuple(latent_audio.shape)), (video, audio))
		self.assertEqual(latent_audio.flatten()[0].item(), 24 * 37 * 52 * 30)

	def test_h3_preview_is_every_frame_at_512_pixels(self):
		import torch
		from PIL import Image
		module = self.module()
		batches = []
		def decoder(latents):
			batches.append(len(latents))
			return torch.nn.functional.interpolate(latents[:, :3], scale_factor=256)
		preview = Image.open(io.BytesIO(module.h3_preview(torch.rand(24 * 5 * 4 * 2), (1, 24, 5, 4, 2), decoder)))
		preview.load()
		self.assertEqual((preview.n_frames, preview.size, preview.info['duration'], batches), (5, (256, 512), 142, [5]))

	def test_skipping_drops_work_while_the_last_still_runs(self):
		from concurrent.futures import ThreadPoolExecutor
		module = self.module()
		released, ran = threading.Event(), []
		with ThreadPoolExecutor(1) as executor:
			submit = module.skipping(executor)
			submit(lambda: released.wait(2) and ran.append(1))
			submit(ran.append, 2)
			released.set()
			executor.submit(lambda: None).result()
			submit(ran.append, 3)
		self.assertEqual(ran, [1, 3])

	def test_h3_noise_follows_comfyui_order(self):
		import torch
		module = self.module()
		noise = module.h3_noise(module.arguments(['--width', '64', '--height', '32', '--length', '30', '--steps', '2',
			'--sampler', 'er_sde'], 'h3'), 7)
		self.assertEqual(list(noise), ['video', 'audio', 'video.noise.0', 'audio.noise.0'])
		generator = torch.manual_seed(7)
		for name, shape in zip(noise, [(1, 24, 12, 2, 4), (1, 32, 2, 65)] * 2):
			self.assertTrue(torch.equal(noise[name], torch.randn(shape, generator=generator)[0]), name)
		for sampler in ('euler', 'res_multistep', 'er_sde', 'euler_ancestral', 'dpmpp_2m_sde', 'gradient_estimation', 'lcm'):
			args = module.arguments(['--width', '64', '--height', '32', '--length', '5', '--steps', '3', '--sampler', sampler], 'h3')
			self.assertEqual(len(module.h3_noise(args, 7)), 6 if sampler in ('er_sde', 'euler_ancestral', 'dpmpp_2m_sde', 'lcm') else 2,
				sampler)

	def test_h3_checkpoint_with_model_prefix_is_accepted(self):
		import torch
		from safetensors.torch import save_file
		module = self.module()
		quant = json.dumps({'convrot': True, 'convrot_groupsize': 256, 'format': 'int8_tensorwise'})
		with tempfile.TemporaryDirectory() as folder:
			path = Path(folder) / 'h3.safetensors'
			save_file({'model.diffusion_model.blocks.0.attn.qkv_proj.comfy_quant': torch.tensor(list(quant.encode()),
				dtype=torch.uint8)}, str(path))
			self.assertEqual(module.validate_model(path, 'h3'), {})
			with self.assertRaisesRegex(ValueError, 'int8 Convrot'):
				module.validate_model(path)

	def test_h3_loras_apply_at_load_without_a_merge(self):
		import torch
		from safetensors.torch import save_file
		module = self.module()
		quant = json.dumps({'convrot': True, 'convrot_groupsize': 256, 'format': 'int8_tensorwise'})
		with tempfile.TemporaryDirectory() as folder:
			path = Path(folder) / 'h3.safetensors'
			save_file({'blocks.0.attn.qkv_proj.comfy_quant': torch.tensor(list(quant.encode()), dtype=torch.uint8)}, str(path))
			args = module.arguments(['--cache', folder], 'h3')
			self.assertEqual(module.lora_model(args, path, [(Path('a.safetensors'), 0.5)]), (path, 'a.safetensors=0.5'))

	def test_h3_batch_launches_backend_once_with_distinct_inputs(self):
		from unittest.mock import MagicMock, patch
		import torch
		from safetensors.torch import save_file
		module = self.module()
		with tempfile.TemporaryDirectory() as temporary:
			folder = Path(temporary)
			args = module.arguments(['--quality', 'faster', '--width', '32', '--height', '32', '--length', '5',
				'--cache', str(folder), '--sampler', 'res_multistep', '--scheduler', 'sgm_uniform', '--decode'], 'h3')
			contexts = []
			for index in range(2):
				path = folder / f'context{index}.safetensors'
				save_file({'context': torch.full((2, 3), float(index)), 'tags': torch.ones(2)}, str(path))
				contexts.append(path)
			process = MagicMock()
			process.stdout = io.StringIO('step 0 3500 ms, forecasts 0\ngeneration 0 in 20 ms\ngeneration 1 in 21 ms\n'
				'decoded 0 in 30 ms\ndecoded 1 in 31 ms\n')
			process.wait.return_value = 0
			process.poll.return_value = 0
			with patch.object(module.subprocess, 'Popen', return_value=process) as launch, redirect_stdout(io.StringIO()):
				results = list(module.Resident().generated(args, Path('model'), contexts, [{'seed': 42}, {'seed': 7}], folder,
					'style.safetensors=0.5'))
			self.assertEqual(launch.call_count, 1)
			command = launch.call_args.args[0]
			self.assertEqual(command[0], module.executable('h3'))
			self.assertEqual(command[1:], ['-', 'model', 'strict', 'style.safetensors=0.5', '8', '12', '2', '1', '1', '1', '5',
				'45', '1', '1', 'minimax_h3_video_vae_int8_convrot.safetensors', 'minimax_h3_audio_vae_fp32.safetensors'])
			self.assertEqual([result[1].name for result in results], ['strict-0-decoded.rgb', 'strict-1-decoded.rgb'])
			self.assertEqual(self.tensors(folder / 'generation.safetensors')['jobs'].item(), 2)
			first = self.tensors(folder / 'generation-0.safetensors')
			second = self.tensors(folder / 'generation-1.safetensors')
			self.assertEqual(set(first), {'context', 'tags', 'video', 'audio', 'decode'})
			self.assertEqual((tuple(first['video'].shape), tuple(first['audio'].shape)), ((24, 2, 2, 2), (32, 2, 8)))
			self.assertFalse(torch.equal(first['video'], second['video']))

	def test_h3_runtime_loras_become_one_backend_argument(self):
		self.assertEqual(self.module().runtime_spec([(Path('a.safetensors'), 1.0), (Path('b c.safetensors'), 0.5)]),
			'a.safetensors=1.0;b c.safetensors=0.5')

	def test_merge_finds_kohya_peft_and_bare_lora_names(self):
		import torch
		from safetensors.torch import save_file
		with tempfile.TemporaryDirectory() as folder:
			folder = Path(folder)
			name = 'blocks.0.attn.qkv_proj'
			quant = json.dumps({'convrot': True, 'convrot_groupsize': 256, 'format': 'int8_tensorwise'})
			save_file({'model.diffusion_model.' + name + '.weight': torch.ones((2, 256), dtype=torch.int8),
				'model.diffusion_model.' + name + '.weight_scale': torch.ones((2, 1)),
				'model.diffusion_model.' + name + '.comfy_quant': torch.tensor(list(quant.encode()), dtype=torch.uint8)},
				str(folder / 'base.safetensors'))
			down, up = torch.ones((1, 256)), torch.full((2, 1), 0.5)
			for label, prefix, names in [('peft', 'diffusion_model.' + name + '.', ('lora_A', 'lora_B')),
				('kohya', 'lora_unet_blocks_0_attn_qkv_proj.', ('lora_down', 'lora_up')), ('bare', name + '.', ('lora_down', 'lora_up'))]:
				save_file({prefix + names[0] + '.weight': down, prefix + names[1] + '.weight': up}, str(folder / (label + '.safetensors')))
				result = subprocess.run([sys.executable, str(ROOT / 'merge.py'), 'base.safetensors', label + '-merged.safetensors',
					label + '.safetensors'], cwd=folder, capture_output=True, text=True)
				self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
				merged = self.tensors(folder / (label + '-merged.safetensors'))
				self.assertTrue(torch.allclose(merged['model.diffusion_model.' + name + '.weight'].float() *
					merged['model.diffusion_model.' + name + '.weight_scale'], torch.full((2, 256), 1.5)))

	def test_h3_image_rows_restart_the_seeded_noise_per_keyframe(self):
		import torch
		module = self.module()
		latent = torch.arange(24 * 4 * 6, dtype=torch.float32).reshape(24, 1, 4, 6)
		rows = module.h3_rows([latent, latent], 42)
		self.assertEqual(tuple(rows.shape), (12, 96))
		self.assertTrue(torch.equal(rows[:6], rows[6:]))
		self.assertTrue(torch.allclose(rows[0, :4], 0.999 * torch.tensor([0.0, 1, 6, 7]) +
			0.001 * torch.randn((6, 96), generator=torch.Generator().manual_seed(42))[0, :4]))

	def test_h3_image_jobs_carry_rows_instead_of_keyframe_latents(self):
		from unittest.mock import MagicMock, patch
		import torch
		from safetensors.torch import save_file
		module = self.module()
		with tempfile.TemporaryDirectory() as temporary:
			folder = Path(temporary)
			args = module.arguments(['--width', '32', '--height', '32', '--length', '5', '--cache', str(folder)], 'h3')
			context = folder / 'context.safetensors'
			latent = torch.ones((24, 1, 2, 2))
			save_file({'context': torch.zeros((2, 3)), 'tags': torch.ones(2), 'keyframe.0': latent,
				'keyframes': torch.zeros(1)}, str(context))
			process = MagicMock()
			process.stdout = io.StringIO('generation 0 in 20 ms\n')
			process.wait.return_value = 0
			process.poll.return_value = 0
			with patch.object(module.subprocess, 'Popen', return_value=process), redirect_stdout(io.StringIO()):
				results = list(module.Resident().generated(args, Path('model'), [context], [{'seed': 42}], folder))
			self.assertEqual([result[1].name for result in results], ['strict-0-latent.safetensors'])
			job = self.tensors(folder / 'generation-0.safetensors')
			self.assertEqual(set(job), {'context', 'tags', 'keyframes', 'rows', 'video', 'audio'})
			self.assertTrue(torch.equal(job['rows'], module.h3_rows([latent], 42)))

if __name__ == '__main__':
	unittest.main()
