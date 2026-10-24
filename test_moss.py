"""Local CLI checks; run with ComfyUI's Python. No full model required."""
import importlib.util
import json
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

	def test_defaults(self):
		args = self.module().arguments(['--model', 'moss.safetensors'])
		self.assertEqual((args.prompt, args.steps, args.output), (Path('prompt.txt'), 8, Path('MossImage.jpg')))

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
