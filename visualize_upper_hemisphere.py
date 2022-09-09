import os
import numpy as np
import open3d as o3d
import trimesh


def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)


def sample_spherical(n, radius=1.0):
    xyz = np.random.normal(size=(n, 3))
    xyz = normalize(xyz) * radius
    xyz[:, 1] = abs(xyz[:, 1])
    return xyz


path = "/home/nicolai/phd/data/ShapeNetCore.v2/02958343/100715345ee54d7ae38b52b4ee9d36a3/models/model_normalized.obj"
mesh = trimesh.load(path, force="mesh")
vertices, faces = np.array(mesh.vertices), np.array(mesh.faces)

camera_views = sample_spherical(1, radius=2.0)

points = trimesh.sample.sample_surface(mesh, 10000)[0]
points = points[np.where(points[:, 0] > 0.0)]

points_reflected = points.copy()
points_reflected[:, 0] = -points_reflected[:, 0]

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
pcd_reflected = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_reflected))
pcd.paint_uniform_color(np.array([1, 0, 0]))
pcd_reflected.paint_uniform_color(np.array([0, 1, 0]))

o3d_mesh = o3d.geometry.TriangleMesh()
o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
o3d.visualization.draw_geometries([pcd, pcd_reflected, axis])
