"""Local CLI checks; run with ComfyUI's Python. No full model required."""
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

	def test_executable_help(self):
		result = subprocess.run([str(ROOT / 'MossImage.exe'), '--help'], capture_output=True, text=True)
		self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
		self.assertIn('--prompt', result.stdout)
		self.assertIn('--lora', result.stdout)
		self.assertIn('--jpg_quality', result.stdout)
		self.assertNotIn('-h,', result.stdout)
		self.assertIn('--quality', result.stdout)

	def test_defaults(self):
		args = self.module().arguments(['--model', 'moss.safetensors'])
		self.assertEqual((args.prompt, args.steps, args.output), (Path('prompt.txt'), 8, Path('MossImage.jpg')))
		self.assertTrue(hasattr(args, 'json_output'), 'JSON sidecar option is missing')
		self.assertFalse(args.json_output)
		self.assertTrue(self.module().arguments(['--json_output']).json_output)
		self.assertTrue(hasattr(args, 'lora_mode'), 'Cached merged mode is missing')
		self.assertEqual(args.lora_mode, 'merged')
		self.assertFalse(args.merge_only)
		self.assertEqual(args.spectrum, 'off')

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
			with patch.object(module, 'validate_model', return_value={}), \
				patch.object(module, 'cached_merge', return_value=merged), \
				patch.object(module, 'conditioning') as encode:
				self.assertEqual(module.main(['moss', '--model', str(base), '--lora', str(lora),
					'--cache', str(folder), '--merge_only', '--prompt', str(folder / 'missing.txt'),
					'--te', 'missing-encoder', '--vae', 'missing-vae']), 0)
				encode.assert_not_called()

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

	def test_visible_defaults_and_long_help_only(self):
		module = self.module()
		with redirect_stderr(io.StringIO()):
			try:
				args = module.arguments(['--comfyui', 'D:/ComfyUI_windows_portable/ComfyUI'])
			except SystemExit:
				self.fail('Default model is missing')
			self.assertEqual((args.width, args.height, args.steps, args.guidance, args.jpg_quality), (720, 1280, 8, 1, 85))
			self.assertEqual(Path(args.model), Path('D:/ComfyUI_windows_portable/ComfyUI/models/diffusion_models/moss-image-sft-dit-int8-convrot.safetensors'))
			self.assertEqual(Path(args.vae).name, 'flux2-vae.safetensors')
			self.assertEqual(Path(args.te).name, 'qwen3.5_9b_int8_convrot.safetensors')
			for option in ('-h', '--quality'):
				with self.assertRaises(SystemExit):
					module.arguments([option])

	def test_count_and_distinct_reproducible_jobs(self):
		module = self.module()
		self.assertTrue(hasattr(module, 'jobs'), 'Resident batch jobs are missing')
		args = module.arguments(['--model', 'moss', '--count', '10', '--seed', '42'])
		first = module.jobs(args, '{a woman|a man|}')
		self.assertEqual(first, module.jobs(args, '{a woman|a man|}'))
		self.assertEqual(len(first), 10)
		self.assertEqual(len({job['seed'] for job in first}), 10)
		self.assertEqual(first[0]['seed'], 42)
		self.assertGreater(len({job['prompt'] for job in first}), 1)
		self.assertEqual(module.arguments(['--model', 'moss']).count, 1)
		with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
			module.arguments(['--model', 'moss', '--count', '0'])

	def test_batch_launches_backend_once_with_distinct_inputs(self):
		from unittest.mock import MagicMock, patch
		import torch
		from safetensors.torch import save_file
		module = self.module()
		self.assertTrue(hasattr(module, 'jobs'), 'Resident batch jobs are missing')
		with tempfile.TemporaryDirectory() as temporary:
			folder = Path(temporary)
			args = module.arguments(['--model', 'moss', '--count', '2', '--seed', '42',
				'--width', '16', '--height', '16', '--cache', str(folder), '--spectrum', 'spectrum8_w3p3'])
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
				results = list(module.generated(args, Path('model'), Path('vae'), contexts,
					module.jobs(args, '{a|b}'), folder))
			self.assertEqual(launch.call_count, 1)
			self.assertIn('step 0 at 1.234', output.getvalue())
			self.assertNotIn('evaluation ', output.getvalue())
			self.assertEqual(launch.call_args.args[0][5], 'on')
			self.assertEqual([result[1].name for result in results], ['strict-on-0.jpg', 'strict-on-1.jpg'])
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
					results = list(resident.generated(args, Path('model'), '', [folder / 'context.safetensors'],
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
				list(resident.generated(args, Path('model'), '', [folder / 'context.safetensors'], [{'seed': 2}], folder / '2'))
				self.assertEqual(launch.call_count, 2)
				process.terminate.assert_called_once()

	def test_backend_without_vae_returns_latents(self):
		from unittest.mock import MagicMock, patch
		import torch
		from safetensors.torch import save_file
		module = self.module()
		with tempfile.TemporaryDirectory() as temporary:
			folder = Path(temporary)
			args = module.arguments(['--width', '16', '--height', '16', '--cache', str(folder)])
			context = folder / 'context.safetensors'
			save_file({'context': torch.zeros((1, 2, 3))}, str(context))
			process = MagicMock()
			process.stdout = io.StringIO('generation 0 in 20 ms\n')
			process.wait.return_value = 0
			with patch.object(module.subprocess, 'Popen', return_value=process) as launch, redirect_stdout(io.StringIO()):
				results = list(module.generated(args, Path('model'), '', [context], [{'seed': 1}], folder))
			self.assertEqual(launch.call_args.args[0][4], '')
			self.assertEqual([result[1].name for result in results], ['strict-off-latent.safetensors'])

	def test_comfyui_lora_patches_become_one_cached_adapter(self):
		from types import SimpleNamespace
		import torch
		module = self.module()
		up, down = torch.randn(4, 2), torch.randn(2, 3)
		other_up, other_down = torch.randn(4, 1), torch.randn(1, 3)
		lora = lambda weights: SimpleNamespace(name='lora', weights=(*weights, None, None, None))
		patches = {'diffusion_model.layers.0.proj.q.weight': [(0.5, lora((up, down, 4.0)), 1.0, None, None),
			(2.0, lora((other_up, other_down, None)), 1.0, None, None)]}
		with tempfile.TemporaryDirectory() as temporary:
			path = module.patched_adapter(Path(temporary), patches)
			self.assertEqual(module.patched_adapter(Path(temporary), patches), path)
			tensors = self.tensors(path)
			prefix = 'diffusion_model.layers.0.proj.q.'
			self.assertEqual(tensors[prefix + 'alpha'].item(), 3)
			expected = 0.5 * 4 / 2 * up @ down + 2 * other_up @ other_down
			torch.testing.assert_close(tensors[prefix + 'lora_up.weight'] @ tensors[prefix + 'lora_down.weight'], expected)
			patches['diffusion_model.layers.0.proj.q.weight'][0] = (1.0, lora((up, down, 4.0)), 1.0, (0, 0, 2), None)
			with self.assertRaises(ValueError):
				module.patched_adapter(Path(temporary), patches)

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

	def test_batch_encodes_unique_missing_prompts_in_one_worker(self):
		from unittest.mock import patch
		import torch
		from safetensors.torch import save_file
		module = self.module()
		with tempfile.TemporaryDirectory() as temporary:
			folder = Path(temporary)
			(folder / 'comfy').mkdir()
			(folder / 'comfy/sd.py').write_text('fixture')
			encoder = folder / 'encoder.safetensors'
			encoder.write_bytes(b'identity')
			args = module.arguments(['--model', 'moss', '--cache', str(folder), '--comfyui', str(folder)])
			def worker(command, **kwargs):
				request = json.loads(Path(command[3]).read_text(encoding='utf-8'))
				self.assertEqual(request['prompts'], ['first', 'second'])
				for index in range(len(request['prompts'])):
					save_file({'context': torch.zeros((1, 2, 3))}, str(Path(command[4]) / f'{index}.safetensors'))
			with patch.object(module.subprocess, 'run', side_effect=worker) as launch:
				paths = module.conditioning(args, ['first', 'second', 'first'], encoder)
				self.assertEqual(paths[0], paths[2])
				self.assertNotEqual(paths[0], paths[1])
				self.assertTrue(all(path.is_file() for path in paths))
				self.assertEqual(module.conditioning(args, ['first', 'second'], encoder), paths[:2])
				self.assertEqual(launch.call_count, 1)

	def test_moss_prompts_encode_in_right_padded_batches(self):
		import torch
		class Clip:
			class cond_stage_model:
				clip = 'qwen'
				qwen = type('Model', (), {'special_tokens': {'pad': 0}})
			batches = []
			def tokenize(self, prompt):
				return {'qwen': [[(len(word), 1.0) for word in prompt.split()]]}
			def encode_from_tokens(self, tokens):
				self.batches.append(tokens['qwen'])
				return torch.tensor([[[token for token, weight in part] for part in tokens['qwen']]], dtype=torch.float).reshape(1, -1, 1)
		clip = Clip()
		contexts = [context.flatten().tolist() for context in self.module().contexts(clip, ['a bb ccc', 'dddd', 'ee f'], 2)]
		self.assertEqual(contexts, [[1, 2, 3], [4], [2, 1]])
		self.assertEqual(clip.batches, [[[(1, 1.0), (2, 1.0), (3, 1.0)], [(4, 1.0), (0, 1.0), (0, 1.0)]], [[(2, 1.0), (1, 1.0)]]])

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

	def test_json_sidecar_is_opt_in(self):
		from contextlib import redirect_stdout
		from unittest.mock import patch
		module = self.module()
		with tempfile.TemporaryDirectory() as temporary:
			folder = Path(temporary)
			model = folder / 'model.safetensors'
			model.touch()
			prompt = folder / 'prompt.txt'
			prompt.write_text('woman', encoding='utf-8')
			source = folder / 'native.jpg'
			source.write_bytes(b'native image bytes')
			def batch(*args, **kwargs):
				yield 0, source, 10
			for enabled in (False, True):
				output = folder / ('with.jpg' if enabled else 'without.jpg')
				command = ['moss', '--model', str(model), '--vae', str(model), '--te', str(model), '--steps', '25',
					'--prompt', str(prompt), '--output', str(output), '--seed', '42', '--cache', str(folder)]
				if enabled:
					command.append('--json_output')
				with patch.object(module, 'validate_model'), patch.object(module, 'native', return_value=True), \
					patch.object(module, 'conditioning', return_value=[model]), patch.object(module, 'generated', side_effect=batch), \
					redirect_stdout(io.StringIO()):
					self.assertEqual(module.main(command), 0)
				image = output.with_name(output.stem + '0001.jpg')
				self.assertEqual(image.read_bytes(), source.read_bytes())
				self.assertEqual(image.with_suffix('.jpg.json').exists(), enabled)

	def test_launcher_preserves_spaces(self):
		name = 'missing model with spaces.safetensors'
		result = subprocess.run([str(ROOT / 'MossImage.exe'), '--model', name],
			cwd=tempfile.gettempdir(), capture_output=True, text=True)
		self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
		self.assertIn('File not found: ' + name, result.stderr)

	def test_prompt_choices_and_empty_alternative(self):
		expand = self.module().expand_prompt
		results = {expand('Portrait of {a woman|a man|}.', random.Random(seed)) for seed in range(100)}
		self.assertEqual(results, {'Portrait of a woman.', 'Portrait of a man.', 'Portrait of .'})
		self.assertEqual(expand('{a|{b|c}} and {d|}', random.Random(7)),
			expand('{a|{b|c}} and {d|}', random.Random(7)))
		self.assertEqual(expand('plain | text', random.Random(0)), 'plain | text')
		for text in ('{unclosed', 'unopened}'):
			with self.assertRaises(ValueError):
				expand(text, random.Random(0))

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

	def test_numbered_outputs_never_overwrite(self):
		reserve = self.module().reserve_output
		with tempfile.TemporaryDirectory() as folder:
			base = Path(folder) / 'MossImage.jpg'
			old = base.with_name('MossImage0009.jpg')
			old.write_bytes(b'keep')
			base.with_name('MossImage0001.jpg').write_bytes(b'first')
			path, output = reserve(base)
			with output:
				output.write(b'new')
			self.assertEqual(path.name, 'MossImage0010.jpg')
			self.assertEqual(old.read_bytes(), b'keep')
			path, output = reserve(base)
			output.close()
			self.assertEqual(path.name, 'MossImage0011.jpg')

	def test_number_reservation_is_exclusive(self):
		from concurrent.futures import ThreadPoolExecutor
		reserve = self.module().reserve_output
		with tempfile.TemporaryDirectory() as folder:
			with ThreadPoolExecutor(max_workers=4) as threads:
				results = list(threads.map(lambda _: reserve(Path(folder) / 'image.jpg'), range(8)))
			for path, output in results:
				output.close()
			self.assertEqual(len({path for path, output in results}), 8)

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
		result = subprocess.run([str(ROOT / 'H3Video.exe'), '--help'], capture_output=True, text=True)
		self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
		for option in ('--prompt', '--lora', '--length', '--count', '--spectrum'):
			self.assertIn(option, result.stdout)
		self.assertNotIn('--jpg_quality', result.stdout)

	def test_h3_defaults(self):
		module = self.module()
		args = module.arguments(['--comfyui', 'C:/ComfyUI'], 'h3')
		self.assertEqual((args.width, args.height, args.length, args.steps, args.shift, args.output),
			(480, 832, 124, 8, 12, Path('H3Video.mp4')))
		self.assertEqual(Path(args.model).name, 'Minimax_H3_FL2VA_PRUNED_Turbo_8step_v1.0_768p_comfyui_INT8_.safetensors')
		self.assertEqual(Path(args.audio_vae).name, 'minimax_h3_audio_vae_fp32.safetensors')
		self.assertEqual(args.cache, ROOT / 'artifacts' / 'h3')
		self.assertFalse(hasattr(args, 'jpg_quality'))
		with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
			module.arguments(['--output', 'video.jpg'], 'h3')

	def test_h3_spectrum_preset(self):
		module = self.module()
		self.assertEqual(module.spectrum_config(module.arguments(['--spectrum', 'spectrum8_w1p2'], 'h3')), ('on', 1, 2))
		with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
			module.arguments(['--spectrum', 'spectrum8_w1p2', '--steps', '6'], 'h3')

	def test_moss_quality_presets(self):
		module = self.module()
		presets = {name: module.arguments(['--quality', name]) for name in ('best25steps', 'best50steps', 'fast', 'faster', 'fastest')}
		presets['best'] = module.arguments([])
		self.assertEqual({name: (args.quality, args.steps, args.guidance, args.spectrum) for name, args in presets.items()}, {
			'best': ('best', 8, 1, 'off'), 'best25steps': ('best25steps', 25, 4, 'off'), 'best50steps': ('best50steps', 50, 4, 'off'),
			'fast': ('fast', 8, 1, 'spectrum8_w3p3'), 'faster': ('faster', 8, 1, 'spectrum8_w2p3'), 'fastest': ('fastest', 4, 1, 'spectrum4_w2p2')})
		changed = module.arguments(['--quality', 'faster', '--steps', '10', '--guidance', '2'])
		self.assertEqual((changed.steps, changed.guidance, changed.spectrum), (10, 2, 'spectrum10_w2p3'))
		with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
			module.arguments(['--quality', 'fastest6steps'])

	def test_h3_quality_presets(self):
		module = self.module()
		best = module.arguments(['--comfyui', 'C:/ComfyUI'], 'h3')
		self.assertEqual((best.quality, best.steps, best.routing, best.spectrum, best.lora), ('best', 8, 0, 'off', []))
		fastest = module.arguments(['--quality', 'fastest3steps', '--lora', 'style.safetensors:0.5'], 'h3')
		self.assertEqual((Path(fastest.model).name, fastest.steps, fastest.routing, fastest.lora),
			('minimax_h3_fl2va_pruned_int8_convrot.safetensors', 3, 1,
			['h3/minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors:1', 'style.safetensors:0.5']))
		faster = module.arguments(['--quality', 'faster', '--steps', '10', '--routing', '0.5'], 'h3')
		self.assertEqual((faster.steps, faster.spectrum, faster.routing), (10, 'spectrum10_w1p2', 0.5))
		self.assertEqual(Path(module.arguments(['--quality', 'best20steps'], 'h3').model).name,
			'minimax_h3_fl2va_pruned_int8_convrot.safetensors')
		with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
			module.arguments(['--quality', 'ultra'], 'h3')

	def test_h3_node_settings_and_latents_follow_comfyui_latent(self):
		import torch
		module = self.module()
		video, audio = (1, 24, 37, 52, 30), (1, 32, 2, 207)
		args = module.h3_arguments('fast', video, audio)
		self.assertEqual((args.width, args.height, args.length, args.steps, args.routing), (480, 832, 124, 8, 1))
		self.assertEqual(tuple(module.h3_noise(args, 7)[1].shape), audio[1:])
		with self.assertRaises(ValueError):
			module.h3_arguments('fast', video, (1, 32, 2, 100))
		numbers = torch.arange(24 * 37 * 52 * 30 + 32 * 2 * 207, dtype=torch.float32)
		latent_video, latent_audio = module.h3_latents(numbers, video, audio)
		self.assertEqual((tuple(latent_video.shape), tuple(latent_audio.shape)), (video, audio))
		self.assertEqual(latent_audio.flatten()[0].item(), 24 * 37 * 52 * 30)

	def test_h3_preview_is_every_other_frame_at_512_pixels(self):
		import torch
		from PIL import Image
		module = self.module()
		decoder = lambda latent: torch.nn.functional.interpolate(latent[:, :3], scale_factor=256)
		preview = Image.open(io.BytesIO(module.h3_preview(torch.rand(24 * 5 * 4 * 2), (1, 24, 5, 4, 2), decoder)))
		self.assertEqual((preview.n_frames, preview.size), (3, (256, 512)))

	def test_h3_noise_follows_comfyui_order(self):
		import torch
		module = self.module()
		video, audio = module.h3_noise(module.arguments(['--width', '64', '--height', '32', '--length', '30'], 'h3'), 7)
		self.assertEqual((tuple(video.shape), tuple(audio.shape)), ((24, 12, 2, 4), (32, 2, 65)))
		generator = torch.manual_seed(7)
		self.assertTrue(torch.equal(video, torch.randn((1, 24, 12, 2, 4), generator=generator)[0]))
		self.assertTrue(torch.equal(audio, torch.randn((1, 32, 2, 65), generator=generator)[0]))

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
			args = module.arguments(['--count', '2', '--seed', '42', '--width', '32', '--height', '32', '--length', '5',
				'--cache', str(folder), '--spectrum', 'spectrum8_w1p2', '--routing', '1'], 'h3')
			contexts = []
			for index in range(2):
				path = folder / f'context{index}.safetensors'
				save_file({'context': torch.full((2, 3), float(index)), 'tags': torch.ones(2)}, str(path))
				contexts.append(path)
			process = MagicMock()
			process.stdout = io.StringIO('step 0 3500 ms, forecasts 0\ngeneration 0 in 20 ms\ngeneration 1 in 21 ms\n')
			process.wait.return_value = 0
			process.poll.return_value = 0
			with patch.object(module.subprocess, 'Popen', return_value=process) as launch, redirect_stdout(io.StringIO()):
				results = list(module.generated(args, Path('model'), Path('vae'), contexts,
					module.jobs(args, '{a|b}'), folder, 'style.safetensors=0.5'))
			self.assertEqual(launch.call_count, 1)
			command = launch.call_args.args[0]
			self.assertEqual(Path(command[0]).name, 'H3Video.exe')
			self.assertEqual(command[1:], [str(folder), 'model', 'strict', 'style.safetensors=0.5', '8', '12', '2', '1', '1', '1'])
			self.assertEqual([result[1].name for result in results], ['strict-0-latent.safetensors', 'strict-1-latent.safetensors'])
			self.assertEqual(self.tensors(folder / 'generation.safetensors')['jobs'].item(), 2)
			first = self.tensors(folder / 'generation-0.safetensors')
			second = self.tensors(folder / 'generation-1.safetensors')
			self.assertEqual(set(first), {'context', 'tags', 'video', 'audio'})
			self.assertEqual((tuple(first['video'].shape), tuple(first['audio'].shape)), ((24, 2, 2, 2), (32, 2, 8)))
			self.assertFalse(torch.equal(first['video'], second['video']))

	def test_h3_latents_decode_in_one_worker_sampling_does_not_wait_for(self):
		from unittest.mock import MagicMock, patch
		module = self.module()
		with tempfile.TemporaryDirectory() as temporary:
			folder = Path(temporary)
			args = module.arguments(['--comfyui', str(folder)], 'h3')
			paths = [folder / f'strict-{index}-latent.safetensors' for index in range(2)]
			events, finished = [], []
			sampled, written = threading.Event(), threading.Event()
			def printed():
				sampled.wait(2)
				events.append('printed')
				yield 'decoder started\n'
			process = MagicMock()
			process.stdout = printed()
			process.wait.return_value = 0
			process.poll.return_value = None
			process.stdin.write.side_effect = lambda text: events.append(json.loads(text)) or written.set()
			def latents():
				events.append('sampling')
				sampled.set()
				yield from [(index, path, 20 + index) for index, path in enumerate(paths)]
				finished.append(written.wait(2))
			def launch(command, **kwargs):
				events.append('worker')
				return process
			with patch.object(module.subprocess, 'Popen', side_effect=launch), redirect_stdout(io.StringIO()):
				videos = module.decoded(args, latents(), folder, len(paths))
			self.assertEqual(events[:2], ['worker', 'sampling'])
			self.assertEqual(finished, [True])
			jobs = next(event for event in events if isinstance(event, list))
			self.assertEqual([job['latent'] for job in jobs], [str(path) for path in paths])
			self.assertEqual(jobs[1]['prefix'], 'strict-1-')
			self.assertEqual(videos, [(0, folder / 'strict-0-video.mp4', 20), (1, folder / 'strict-1-video.mp4', 21)])

	def test_h3_decode_worker_encodes_while_the_next_latent_decodes(self):
		from types import ModuleType
		from unittest.mock import MagicMock, patch
		import torch
		from safetensors.torch import save_file
		module = self.module()
		second, overlapped, events = threading.Event(), [], []
		def decoded(vae, path, shapes, out, prefix):
			events.append('decode ' + prefix)
			second.set() if prefix == 'strict-1-' else None
			return prefix, 0, 0
		def encoded(frames, out, prefix):
			overlapped.append(second.wait(2)) if prefix == 'strict-0-' else None
			events.append('encode ' + prefix)
		dump = ModuleType('dump')
		dump.comfy, dump.audio_decoder, dump.decoded_latent, dump.encoded_av = MagicMock(), MagicMock(), decoded, encoded
		audio = ModuleType('comfy_extras.nodes_audio')
		with tempfile.TemporaryDirectory() as folder:
			save_file({'video': torch.zeros(1), 'audio': torch.zeros(1)}, str(Path(folder) / 'generation.safetensors'))
			jobs = [{'latent': 'x', 'generation': str(Path(folder) / 'generation.safetensors'), 'prefix': f'strict-{index}-'}
				for index in range(2)]
			stdin, before_jobs = MagicMock(), []
			stdin.read.side_effect = lambda: before_jobs.extend(call.args[0] for call in imported.call_args_list) or json.dumps(jobs)
			modules = {'dump': dump, 'comfy_extras': ModuleType('comfy_extras'), 'comfy_extras.nodes_audio': audio}
			with patch.dict(sys.modules, modules), patch.object(sys, 'stdin', stdin), patch.object(sys, 'argv', ['generate.py']), \
				patch.object(module.os, 'environ', {}), patch('builtins.__import__', wraps=__import__) as imported, \
				redirect_stdout(io.StringIO()):
				module.decoding({'comfyui': folder, 'vae': 'v', 'audio_vae': 'a', 'folder': folder})
		self.assertEqual(overlapped, [True])
		self.assertEqual(sorted(events), ['decode strict-0-', 'decode strict-1-', 'encode strict-0-', 'encode strict-1-'])
		self.assertIn('comfy_extras.nodes_audio', before_jobs)

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

	def test_h3_image_options(self):
		module = self.module()
		args = module.arguments(['--image', 'first.png', '--last_image', 'last.png'], 'h3')
		self.assertEqual((args.image, args.last_image), (Path('first.png'), Path('last.png')))
		self.assertIsNone(module.arguments([], 'h3').image)
		with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
			module.arguments(['--image', 'first.png'])

	def test_h3_image_rows_restart_the_seeded_noise_per_keyframe(self):
		import torch
		module = self.module()
		latent = torch.arange(24 * 4 * 6, dtype=torch.float32).reshape(24, 1, 4, 6)
		rows = module.h3_rows([latent, latent], 42)
		self.assertEqual(tuple(rows.shape), (12, 96))
		self.assertTrue(torch.equal(rows[:6], rows[6:]))
		self.assertTrue(torch.allclose(rows[0, :4], 0.999 * torch.tensor([0.0, 1, 6, 7]) +
			0.001 * torch.randn((6, 96), generator=torch.Generator().manual_seed(42))[0, :4]))

	def test_h3_image_conditioning_is_cached_per_image_and_size(self):
		from unittest.mock import patch
		import torch
		from safetensors.torch import save_file
		module = self.module()
		with tempfile.TemporaryDirectory() as temporary:
			folder = Path(temporary)
			(folder / 'comfy').mkdir()
			(folder / 'comfy/sd.py').write_text('fixture')
			for name in ('encoder.safetensors', 'vae.safetensors', 'first.png', 'other.png'):
				(folder / name).write_bytes(name.encode())
			requests = []
			def worker(command, **kwargs):
				request = json.loads(Path(command[3]).read_text(encoding='utf-8'))
				requests.append(request)
				for index in range(len(request['prompts'])):
					save_file({'context': torch.zeros((2, 3))}, str(Path(command[4]) / f'{index}.safetensors'))
			paths = []
			for image in ('first.png', 'other.png', 'first.png'):
				args = module.arguments(['--comfyui', str(folder), '--cache', str(folder), '--image', str(folder / image),
					'--vae', str(folder / 'vae.safetensors')], 'h3')
				with patch.object(module.subprocess, 'run', side_effect=worker):
					paths += module.conditioning(args, ['prompt'], folder / 'encoder.safetensors')
			self.assertEqual(len(requests), 2)
			self.assertEqual(requests[0]['images'], [str(folder / 'first.png'), ''])
			self.assertEqual((requests[0]['width'], requests[0]['height'], requests[0]['length']), (480, 832, 124))
			self.assertNotEqual(paths[0], paths[1])
			self.assertEqual(paths[0], paths[2])

	def test_h3_image_jobs_carry_rows_instead_of_keyframe_latents(self):
		from unittest.mock import MagicMock, patch
		import torch
		from safetensors.torch import save_file
		module = self.module()
		with tempfile.TemporaryDirectory() as temporary:
			folder = Path(temporary)
			args = module.arguments(['--seed', '42', '--width', '32', '--height', '32', '--length', '5', '--cache', str(folder)], 'h3')
			context = folder / 'context.safetensors'
			latent = torch.ones((24, 1, 2, 2))
			save_file({'context': torch.zeros((2, 3)), 'tags': torch.ones(2), 'keyframe.0': latent,
				'keyframes': torch.zeros(1)}, str(context))
			process = MagicMock()
			process.stdout = io.StringIO('generation 0 in 20 ms\n')
			process.wait.return_value = 0
			process.poll.return_value = 0
			with patch.object(module.subprocess, 'Popen', return_value=process), redirect_stdout(io.StringIO()):
				list(module.generated(args, Path('model'), Path('vae'), [context], module.jobs(args, 'a'), folder, ''))
			job = self.tensors(folder / 'generation-0.safetensors')
			self.assertEqual(set(job), {'context', 'tags', 'keyframes', 'rows', 'video', 'audio'})
			self.assertTrue(torch.equal(job['rows'], module.h3_rows([latent], 42)))

if __name__ == '__main__':
	unittest.main()
