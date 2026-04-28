"""Veldt metadata marker nodes for GLB exports.

Embeds key-value metadata as empty glTF nodes named ``__veldt_{key}_{value}``.
The Unity client reads these via ``VeldtMeta.FromGameObject()`` after loading.

Example::

    add_veldt_meta_node(gltf, "type", "dicom2d")
    # → creates node named "__veldt_type_dicom2d"
"""

from __future__ import annotations

import pygltflib


def add_veldt_meta_node(gltf: pygltflib.GLTF2, key: str, value: str) -> None:
    """Add a ``__veldt_{key}_{value}`` marker node to the first scene."""
    node_idx = len(gltf.nodes)
    gltf.nodes.append(pygltflib.Node(name=f"__veldt_{key}_{value}"))
    if gltf.scenes and gltf.scenes[0].nodes is not None:
        gltf.scenes[0].nodes.append(node_idx)
