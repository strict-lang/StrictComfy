# Pending work
Only what is still broken or not implemented: remove an item once it is done (numbers go into commit messages or the Readme).

## Nodes
- The Moss and H3 nodes call `unload_all_models` for every prompt: on the 4090 ComfyUI's 9.3 GB Moss text encoder leaves the GPU and loads again (9.4-10 s per image against ComfyUI's 3.2 s). Free only the memory Strict needs (`comfy.model_management.free_memory`), the whole unload only on small cards.
- `released` moves the cached Load Diffusion Model weights to meta: a later prompt with ComfyUI's KSampler on the same loader fails ("Tensor.item() cannot be called on meta tensors").
