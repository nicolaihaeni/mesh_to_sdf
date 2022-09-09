import os
import json
import numpy as np
import trimesh

import torch
from pytorch3d.datasets import ShapeNetCore
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import TexturesAtlas, TexturesVertex


modes = ["train", "test"]
split_file_path = "/home/nicolai/phd/code/DIF-Net/split/"
SHAPENET_PATH = "/home/nicolai/phd/data/ShapeNetCore.v2"
shapenet_dataset = ShapeNetCore(SHAPENET_PATH, version=2)
model_ids = shapenet_dataset.model_ids

categories = {
    "car": "02958343",
    "chair": "03001627",
    "plane": "02691156",
    "table": "04379243",
}

for category in categories.keys():
    with open(os.path.join(split_file_path, category + ".json"), "r") as infile:
        data = json.load(infile)

    for mode in modes:
        split_data = data[mode]
        for filename in split_data:
            idx = model_ids.index(filename)
            dictionary = shapenet_dataset[idx]
            verts, faces = (
                dictionary["verts"].numpy(),
                dictionary["faces"].numpy(),
            )

            if "textures" in dictionary:
                textures = TexturesAtlas(atlas=dictionary["textures"].unsqueeze(0))
                mesh = Meshes(
                    verts=[dictionary["verts"]],
                    faces=[dictionary["faces"]],
                    textures=textures,
                )
                verts_colors_packed = torch.ones_like(mesh.verts_packed())
                verts_colors_packed[
                    mesh.faces_packed()
                ] = mesh.textures.faces_verts_textures_packed()

                verts_colors = verts_colors_packed.squeeze(0).numpy()
                mesh = trimesh.Trimesh(
                    vertices=verts, faces=faces, vertex_colors=verts_colors
                )
            else:
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)

            # Trimesh save a copy of the mesh to file
            out_path = os.path.join(
                SHAPENET_PATH,
                categories[category],
                filename,
                "models",
                "model_trimesh.obj",
            )

            mesh.export(out_path)
