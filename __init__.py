"""ComfyUI V1 custom-node entry point: NODE_CLASS_MAPPINGS /
NODE_DISPLAY_NAME_MAPPINGS, confirmed as the working registration style in
this ComfyUI version (see CLAUDE.md phase 18 investigation).

nodes.py is loaded via an explicit file path with a private module name
instead of a package-relative `from .nodes import ...`. A relative import
requires a resolvable parent package, which this __init__.py does not
reliably have: pytest's own Package collector (created because this
directory contains __init__.py) always executes this file directly to
support package-level fixtures, and cannot give it package context because
this repo's directory name ("ComfyUI-MiniMaxH3-Cached", matching the
GitHub repo / ComfyUI custom_nodes folder convention) contains hyphens and
is not a valid Python identifier. Loading nodes.py this way works
identically under ComfyUI's real loader and under pytest.
"""

import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("minimaxh3cached_nodes", os.path.join(_here, "nodes.py"))
_nodes_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nodes_module)

MiniMaxH3CachedImageToVideo = _nodes_module.MiniMaxH3CachedImageToVideo

NODE_CLASS_MAPPINGS = {"MiniMaxH3CachedImageToVideo": MiniMaxH3CachedImageToVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3CachedImageToVideo": "MiniMax H3 Cached Images to Video"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
