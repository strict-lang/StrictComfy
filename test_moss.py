"""Local CLI checks; run with ComfyUI's Python. No full model required."""
import importlib.util
import io
import json
from contextlib import redirect_stderr
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


class MossCommandTests(unittest.TestCase):
	def module(self):
		self.assertIsNotNone(importlib.util.find_spec('moss'), 'Local Moss command is missing')
		import moss
		return moss

	def test_executable_help(self):
		result = subprocess.run([str(ROOT / 'MossImage.exe'), '--help'], capture_output=True, text=True)
		self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
		self.assertIn('--prompt', result.stdout)
		self.assertIn('--lora', result.stdout)
		self.assertIn('--jpg_quality', result.stdout)
		self.assertNotIn('-h,', result.stdout)
		self.assertNotIn('--quality', result.stdout)

	def test_defaults(self):
		args = self.module().arguments(['--model', 'moss.safetensors'])
		self.assertEqual((args.prompt, args.steps, args.output), (Path('prompt.txt'), 8, Path('MossImage.jpg')))
		self.assertTrue(hasattr(args, 'json_output'), 'JSON sidecar option is missing')
		self.assertFalse(args.json_output)
		self.assertTrue(self.module().arguments(['--json_output']).json_output)

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
		from safetensors.torch import save_file, load_file
		module = self.module()
		self.assertTrue(hasattr(module, 'jobs'), 'Resident batch jobs are missing')
		with tempfile.TemporaryDirectory() as temporary:
			folder = Path(temporary)
			args = module.arguments(['--model', 'moss', '--count', '2', '--seed', '42',
				'--width', '16', '--height', '16', '--cache', str(folder)])
			contexts = []
			for index in range(2):
				path = folder / f'context{index}.safetensors'
				save_file({'context': torch.full((1, 2, 3), float(index))}, str(path))
				contexts.append(path)
			process = MagicMock()
			process.stdout = io.StringIO('generation 0 in 20 ms\ngeneration 1 in 21 ms\n')
			process.wait.return_value = 0
			process.poll.return_value = 0
			with patch.object(module.subprocess, 'Popen', return_value=process) as launch:
				results = list(module.generated(args, Path('model'), Path('vae'), contexts,
					module.jobs(args, '{a|b}'), folder))
			self.assertEqual(launch.call_count, 1)
			self.assertEqual([result[0] for result in results], [0, 1])
			self.assertEqual(load_file(str(folder / 'generation.safetensors'))['jobs'].item(), 2)
			first = load_file(str(folder / 'generation-0.safetensors'))
			second = load_file(str(folder / 'generation-1.safetensors'))
			self.assertFalse(torch.equal(first['noise'], second['noise']))
			self.assertFalse(torch.equal(first['context'], second['context']))

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

	def test_turbo_selected_only_for_eight_steps_without_duplication(self):
		module = self.module()
		self.assertTrue(hasattr(module, 'selected_loras'), 'Automatic runtime turbo selection is missing')
		with tempfile.TemporaryDirectory() as temporary:
			models = Path(temporary)
			path = models / 'loras/moss/dmd_turbo_lora.safetensors'
			path.parent.mkdir(parents=True)
			path.touch()
			args = module.arguments(['--steps', '8'])
			self.assertEqual(module.selected_loras(args, models), [(path.resolve(), 1)])
			for steps in (25, 50):
				args.steps = steps
				self.assertEqual(module.selected_loras(args, models), [])
			args.steps = 8
			args.lora = [str(path) + ':0.5']
			self.assertEqual(module.selected_loras(args, models), [(path.resolve(), 0.5)])

	def test_runtime_stack_preserves_small_deltas_and_base_file(self):
		import torch
		from safetensors.torch import save_file, load_file
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
			weights = load_file(str(packed))
			self.assertTrue(torch.allclose(weights[prefix + 'lora_up.weight'] @ weights[prefix + 'lora_down.weight'],
				torch.full((2, 4), 1e-5), atol=1e-8, rtol=1e-4))
			self.assertEqual(weights[prefix + 'alpha'].item(), 2)
			self.assertEqual(base.read_bytes(), original)
			self.assertEqual(module.runtime_adapter(args, base, []), '')
			fused = 'diffusion_model.layers.0.proj.qkv.'
			save_file({fused + 'lora_A.weight': torch.ones((1, 4)),
				fused + 'lora_B.weight': torch.arange(1, 7).float().reshape(6, 1) * 1e-5}, str(loras[1][0]))
			weights = load_file(module.runtime_adapter(args, base, loras))
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
				command = ['--model', str(model), '--vae', str(model), '--te', str(model), '--steps', '25',
					'--prompt', str(prompt), '--output', str(output), '--seed', '42', '--cache', str(folder)]
				if enabled:
					command.append('--json_output')
				with patch.object(module, 'validate_model'), patch.object(module, 'conditioning', return_value=[model]), \
					patch.object(module, 'generated', side_effect=batch), redirect_stdout(io.StringIO()):
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
		from safetensors.torch import save_file, load_file
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
			stack = load_file(str(folder / 'stack.safetensors'))
			single = load_file(str(folder / 'single.safetensors'))
			self.assertTrue(torch.equal(stack[name + '.weight'], single[name + '.weight']))
			self.assertTrue(torch.equal(stack[name + '.weight_scale'], single[name + '.weight_scale']))
			self.assertTrue(torch.allclose(stack[name + '.weight'].float() * stack[name + '.weight_scale'],
				torch.full((2, 256), 1.75)))
			self.assertTrue(torch.equal(stack['video_embedder.weight'], torch.full((2, 3), 0.75)))


if __name__ == '__main__':
	unittest.main()
