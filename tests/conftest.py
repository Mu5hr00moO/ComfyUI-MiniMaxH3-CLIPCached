"""Shared pytest setup.

comfy.nested_tensor is a standalone leaf module (only `import torch`, no
ComfyUI startup, no model loading) reused by minimaxh3_clipcache.comparison
for the AV latent's "samples" field. Putting ComfyUI's root on sys.path is
enough to make `import comfy...` resolve here without launching ComfyUI
itself.

This repo lives at <ComfyUI_root>/custom_nodes/<this_repo>/tests/conftest.py
under a normal install, so the ComfyUI root is four levels up from this
file. That is derived rather than hard-coded so the tests also run in a
fork checked out elsewhere; COMFYUI_ROOT in the environment overrides it if
the layout differs.
"""

import os
import sys

_here = os.path.abspath(__file__)
_derived_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_here))))
COMFYUI_ROOT = os.environ.get("COMFYUI_ROOT", _derived_root)
if COMFYUI_ROOT not in sys.path:
    sys.path.insert(0, COMFYUI_ROOT)
