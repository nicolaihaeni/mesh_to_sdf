import os

# os.environ["PYOPENGL_PLATFORM"] = "egl"
import json
import argparse
import h5py
import numpy as np
import trimesh
import open3d as o3d

from mesh_to_sdf.utils import scale_to_unit_sphere
from mesh_to_sdf import get_surface_point_cloud, surface_point_cloud


categories = {
    "car": "02958343",
    "chair": "03001627",
    "plane": "02691156",
    "table": "04379243",
}
split_dir = "/home/isleri/haeni001/code/DIF-Net/split/"


def main(args):
    # Load all the split files
    filenames = []
    for cat in categories:
        with open(os.path.join(split_dir, f"{cat}.json"), "r") as f:
            data = json.load(f)
            for mode in ["test", "train"]:
                for filename in data[mode][categories[cat]]:
                    filenames.append(f"{categories[cat]}/{filename}")

    chunk = np.array_split(filenames, args.num_procs)[args.rank]

    for filename in filenames:
        mesh = trimesh.load(
            os.path.join(args.mesh_dir, filename, "models", "model_normalized.obj")
        )

        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("The mesh parameter must be a trimesh mesh.")

        # Normalize the mesh in sphere of radius 1/1.03
        mesh = scale_to_unit_sphere(mesh)
        bounding_radius = np.max(np.linalg.norm(mesh.vertices, axis=1)) * 1.1

        # Get the surface point cloud
        point_cloud = get_surface_point_cloud(mesh, "scan", bounding_radius)
        surface_points, surface_normals = point_cloud.get_random_surface_points(500000)

        # Free points
        free_points = np.random.uniform(-1, 1, size=(500000, 3))
        free_points_sdf = point_cloud.get_sdf_in_batches(
            free_points, use_depth_buffer=True, sample_count=10000000
        )

        # Visualize
        pcd, free_pcd = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(surface_points)
        pcd.normals = o3d.utility.Vector3dVector(surface_normals)
        colors = np.zeros_like(surface_points)
        colors[:, 0] = 1
        pcd.colors = o3d.utility.Vector3dVector(colors)

        free_pcd.points = o3d.utility.Vector3dVector(free_points)
        colors = np.zeros_like(free_points)
        colors[:, 1] = 1
        free_pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd, free_pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_procs", default=1, type=int, help="number of parallel processes started"
    )
    parser.add_argument("--rank", default=0, type=int, help="rank of current process")
    parser.add_argument("--out_dir", default="./out", type=str, help="output directory")
    parser.add_argument(
        "--mesh_dir",
        default="/home/isleri/haeni001/data/ShapeNetCore.v2/",
        type=str,
        help="mesh directory",
    )
    args = parser.parse_args()
    main(args)
