"""Flatten/unflatten a CONDITIONING-shaped structure for pickle-free storage.

CONDITIONING is an arbitrary nesting of list/tuple/dict/None/str/int/float/
bool/torch.Tensor (e.g. [[tensor, {"pooled_output": None,
"minimax_token_tags": tensor, "minimax_keyframes": [{"latent": tensor, ...}]}]]).
flatten_tensors() pulls every tensor out into a flat dict keyed by its path
so the remaining "skeleton" is plain JSON, and every tensor can be handed to
safetensors.torch.save_file (which only accepts a flat str->Tensor mapping).
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
