"""ComfyUI V1 custom-node entry point: NODE_CLASS_MAPPINGS /
NODE_DISPLAY_NAME_MAPPINGS, confirmed as the working registration style in
this ComfyUI version (see CLAUDE.md phase 18 investigation).

nodes.py is loaded via an explicit file path with a private module name
instead of a package-relative `from .nodes import ...`. A relative import
requires a resolvable parent package, which this __init__.py does not
reliably have: pytest's own Package collector (created because this
directory contains __init__.py) always executes this file directly to
support package-level fixtures, and cannot give it package context because
this repo's directory name ("ComfyUI-MiniMaxH3-CLIPCached", matching the
GitHub repo / ComfyUI custom_nodes folder convention) contains hyphens and
is not a valid Python identifier. Loading nodes.py this way works
identically under ComfyUI's real loader and under pytest.
"""

import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))

# nodes.py does `from caching.loader import ...`, an absolute import of our
# own top-level caching/ package. ComfyUI's real loader never puts this
# repo's own directory on sys.path (confirmed locally: the only sys.path
# mutation in ComfyUI's init_external_custom_nodes() adds <ComfyUI_root>/comfy,
# nothing custom-node-specific), so without this, caching.loader resolves
# fine under pytest (which puts cwd on sys.path) but fails with
# ModuleNotFoundError under real ComfyUI. append, NOT insert(0): confirmed
# locally that ComfyUI's own top-level "nodes" module is already fully
# loaded into sys.modules by the time load_custom_node() reaches this
# package (main.py does `import nodes` long before it calls
# nodes.init_extra_nodes()), so no future `import nodes` anywhere in the
# process will ever re-search sys.path for it -- but append instead of
# insert(0) is one more layer of caution regardless, so this repo's
# directory is never given priority over any existing ComfyUI path entry.
if _here not in sys.path:
    sys.path.append(_here)

_spec = importlib.util.spec_from_file_location("minimaxh3clipcached_nodes", os.path.join(_here, "nodes.py"))
_nodes_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nodes_module)

MiniMaxH3CLIPCachedImageToVideo = _nodes_module.MiniMaxH3CLIPCachedImageToVideo

NODE_CLASS_MAPPINGS = {"MiniMaxH3CLIPCachedImageToVideo": MiniMaxH3CLIPCachedImageToVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3CLIPCachedImageToVideo": "MiniMax H3 CLIP-Cached Images to Video"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
