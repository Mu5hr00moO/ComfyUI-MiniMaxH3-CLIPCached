"""Flatten/unflatten a CONDITIONING-shaped structure for pickle-free storage.

CONDITIONING is an arbitrary nesting of list/tuple/dict/None/str/int/float/
bool/torch.Tensor (e.g. [[tensor, {"pooled_output": None,
"minimax_token_tags": tensor, "minimax_keyframes": [{"latent": tensor, ...}]}]]).
flatten_tensors() pulls every tensor out into a flat dict keyed by its path
so the remaining "skeleton" is plain JSON, and every tensor can be handed to
safetensors.torch.save_file (which only accepts a flat str->Tensor mapping).

Known format limitation: tensor paths join nested components with ".", so the
mapping is not injective if a dict key itself contains "." (see the explicit
ValueError guard in flatten_tensors). Real MiniMax H3 conditioning keys are all
dot-free, so this only matters as defense against an unexpected structure.
"""

import torch

_TENSOR_REF_KEY = "__tensor_ref__"
_TYPE_KEY = "__type__"


def flatten_tensors(obj):
    tensors = {}

    def walk(node, path):
        if isinstance(node, torch.Tensor):
            tensors[path] = node
            return {_TENSOR_REF_KEY: path}
        if isinstance(node, dict):
            for key in node:
                # Flat tensor keys are built by joining path components with
                # ".", so a "." inside a dict key makes the resulting path
                # ambiguous: {"a.b": t} and {"a": {"b": t}} would both flatten
                # to the key "a.b". This format is therefore NOT injective in
                # the general case -- it is only safe for dot-free dict keys.
                # Every key the MiniMax H3 conditioning actually contains
                # (pooled_output, minimax_token_tags, minimax_keyframes,
                # latent, resolved_frame_index, ...) is dot-free, so this is a
                # purely defensive guard against an unexpected structure, not
                # a live bug. Fail loudly here rather than silently write a
                # colliding path.
                if "." in str(key):
                    raise ValueError(
                        "minimaxh3_clipcache.serialize cannot serialize a dict "
                        "key containing '.': {!r} (at path {!r}). The cache "
                        "format joins nested path components with '.', so a "
                        "dotted key would produce an ambiguous, possibly "
                        "colliding tensor path. Keys must be dot-free.".format(
                            key, path or "<root>"))
            return {key: walk(value, "{}.{}".format(path, key) if path else str(key))
                    for key, value in node.items()}
        if isinstance(node, (list, tuple)):
            items = [walk(item, "{}.{}".format(path, i) if path else str(i))
                     for i, item in enumerate(node)]
            if isinstance(node, tuple):
                return {_TYPE_KEY: "tuple", "items": items}
            return items
        # None/str/int/float/bool: already JSON-serializable as-is.
        return node

    skeleton = walk(obj, "")
    return skeleton, tensors


def unflatten_tensors(skeleton, tensors):
    def walk(node):
        if isinstance(node, dict):
            if _TENSOR_REF_KEY in node:
                return tensors[node[_TENSOR_REF_KEY]]
            if node.get(_TYPE_KEY) == "tuple":
                return tuple(walk(item) for item in node["items"])
            return {key: walk(value) for key, value in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(skeleton)
