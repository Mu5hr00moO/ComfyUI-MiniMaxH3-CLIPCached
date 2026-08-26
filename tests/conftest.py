"""Shared pytest setup.

comfy.nested_tensor is a standalone leaf module (only `import torch`, no
ComfyUI startup, no model loading) reused by caching.comparison for the AV
latent's "samples" field. Adding ComfyUI's root to sys.path is enough to
make `import comfy...` resolve here without launching ComfyUI itself.
"""

import sys

COMFYUI_ROOT = "/home/kamil/ComfyUI"
if COMFYUI_ROOT not in sys.path:
    sys.path.insert(0, COMFYUI_ROOT)
