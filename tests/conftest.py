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

Phase 5: minimaxh3_clipcache.routes does `from server import PromptServer`
and, at import time, `routes = PromptServer.instance.routes` followed by
@routes.get / @routes.post handler decorators. There is no ComfyUI server
under pytest, so `server` is stubbed here with a pass-through route table:
the decorators return the handler unchanged, leaving each handler as a
plain module function the tests call directly. Mirrors the stub in
MiniMaxH3-Prompt-Writer/tests/test_routes_stability.py.

CPU-only portability: `comfy.model_management` calls
`torch.cuda.current_device()` at import time (via `get_torch_device()` ->
`total_vram`), which raises "No CUDA GPUs are available" on a host with no
usable NVIDIA driver and aborts `import comfy...` during collection. This
suite is CPU-only, so when CUDA is unavailable we opt into ComfyUI's own
`--cpu` code path (`args.cpu`) before `model_management` is imported. On a
normal GPU host `torch.cuda.is_available()` is True and `args` is left
untouched, so GPU behavior is unchanged.
"""

import os
import sys
import types

_here = os.path.abspath(__file__)
_derived_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_here))))
COMFYUI_ROOT = os.environ.get("COMFYUI_ROOT", _derived_root)
if COMFYUI_ROOT not in sys.path:
    sys.path.insert(0, COMFYUI_ROOT)


try:
    import torch
    import comfy.cli_args

    if not torch.cuda.is_available():
        comfy.cli_args.args.cpu = True
except Exception:
    pass


if "server" not in sys.modules:
    class _PassThroughRoutes:
        def get(self, _path):
            return lambda handler: handler

        post = get
        delete = get

    sys.modules["server"] = types.SimpleNamespace(
        PromptServer=types.SimpleNamespace(
            instance=types.SimpleNamespace(routes=_PassThroughRoutes())
        )
    )
