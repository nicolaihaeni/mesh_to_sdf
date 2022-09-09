import os

# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
# os.environ["PYOPENGL_PLATFORM"] = "egl"
import json
import h5py
import random
import argparse
import numpy as np
import trimesh

import utils
from mesh_to_sdf.utils import scale_to_unit_sphere
from mesh_to_sdf import get_surface_point_cloud, surface_point_cloud
from pytorch3d.datasets import ShapeNetCore
import pyrender

# import matplotlib.pyplot as plt

np.random.seed(0)

SHAPENET_PATH = "/home/isleri/haeni001/data/ShapeNetCore.v2"
# SHAPENET_PATH = "/home/nicolai/phd/data/ShapeNetCore.v2"
categories = {
    "car": "02958343",
    "chair": "03001627",
    "plane": "02691156",
    "table": "04379243",
}
key_list = list(categories.keys())
val_list = list(categories.values())


def main(args):
    # Load all the split files
    shapenet_dataset = ShapeNetCore(SHAPENET_PATH, version=2)
    filenames = []
    for cat in categories:
        with open(os.path.join(args.split_dir, f"{cat}.json"), "r") as f:
            data = json.load(f)
            for mode in ["test", "train"]:
                for filename in data[mode]:
                    filenames.append(f"{categories[cat]}/{filename}")

    chunk = np.array_split(filenames, args.num_procs)[args.rank]

    for ii, filename in enumerate(chunk):
        # Check if we can skip this instance
        category, fname = filename.split("/")
        category = key_list[val_list.index(category)]
        out_path = os.path.join(args.out_dir, category, fname)
        out_filename = os.path.join(out_path, f"{fname}.h5")
        out_rgbd_path = os.path.join(out_path, f"{fname}_rgbd.h5")

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        try:
            mesh = utils.load_mesh_with_color(shapenet_dataset, fname)
        except Exception as e:
            print("Error in loading mesh: {fname}")

        # Normalize the mesh in sphere of radius 1/1.03
        mesh = scale_to_unit_sphere(mesh)
        mesh.export("./mesh.obj")

        # Create SDF data
        if not os.path.exists(out_filename):
            print(f"Processing {ii}/{len(chunk)} files")
            bounding_radius = np.max(np.linalg.norm(mesh.vertices, axis=1)) * 1.1

            # Get the surface point cloud
            point_cloud = get_surface_point_cloud(mesh, "scan", bounding_radius)
            surface_points, surface_normals = point_cloud.get_random_surface_points(
                500000
            )

            # Free points
            free_points = np.random.uniform(-1, 1, size=(500000, 3))
            free_points_sdf = point_cloud.get_sdf_in_batches(
                free_points, use_depth_buffer=True, sample_count=10000000
            )

            surface_data = np.concatenate([surface_points, surface_normals], axis=-1)
            free_data = np.concatenate([free_points, free_points_sdf[:, None]], axis=-1)

            h5file = h5py.File(out_filename, "w")
            h5file.create_dataset(
                "free_pts",
                data=free_data,
                compression="gzip",
            )
            h5file.create_dataset(
                "surface_pts",
                data=surface_data,
                compression="gzip",
            )
            h5file.close()
        else:
            print(f"Skipping: {out_filename}")
        # Render RGBD images
        if not os.path.exists(out_rgbd_path):
            mesh = trimesh.load("./mesh.obj")
            cam_locations = utils.sample_spherical(30, 3.0)
            obj_location = np.zeros((1, 3))
            cv_poses = utils.look_at(cam_locations, obj_location)

            cv_poses = [
                f @ utils.sample_roll_matrix(np.random.uniform(-1, 1, 1) * np.pi)
                for f in cv_poses
            ]

            cam_locations = [utils.cv_cam2world_to_bcam2world(m) for m in cv_poses]
            image_size = (256, 256)
            K = np.array([[262.5, 0.0, 128.0], [0.0, 262.5, 128.0], [0.0, 0.0, 1.0]])
            camera = pyrender.IntrinsicsCamera(
                fx=262.5, fy=262.5, cx=128.0, cy=128.0, znear=0.01, zfar=100
            )

            scene = pyrender.Scene.from_trimesh_scene(
                trimesh.Scene(mesh), ambient_light=(1, 1, 1)
            )

            rgbs = []
            depths = []
            masks = []
            for ii, w2c in enumerate(cam_locations):
                # Add camera roll
                if ii == 0:
                    cam_node = scene.add(camera, pose=np.array(w2c))
                else:
                    scene.set_pose(cam_node, pose=np.array(w2c))

                r = pyrender.OffscreenRenderer(*image_size)
                color, depth = r.render(
                    scene, flags=pyrender.constants.RenderFlags.FLAT
                )

                mask = depth != 0
                depth[mask == 0.0] = 10

                rgbs.append(color)
                depths.append(depth)
                masks.append(mask)
                r.delete()

            # fig, ax = plt.subplots(1, 3)
            # ax[0].imshow(rgbs[0])
            # ax[1].imshow(depths[0], cmap="plasma")
            # ax[2].imshow(masks[0], cmap="plasma")
            # plt.show()

            images = np.stack([r for r in rgbs])
            depths = np.stack([r for r in depths])
            masks = np.stack([r for r in masks])
            cam2world = np.stack([r for r in cv_poses])

            hf = h5py.File(out_rgbd_path, "w")
            hf.create_dataset("rgb", data=rgbs, compression="gzip", dtype="f")
            hf.create_dataset("depth", data=depths, compression="gzip", dtype="f")
            hf.create_dataset("mask", data=masks, compression="gzip", dtype="f")
            hf.create_dataset(
                "cam2world", data=cam2world, compression="gzip", dtype="f"
            )
            hf.create_dataset("K", data=K, dtype="f")
            hf.close()

            # Visualize
            # import open3d as o3d

            # with h5py.File(out_filename, "r") as hf:
            # surface_points = hf["surface_pts"][:, :3]

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(surface_points)
            # pcd.paint_uniform_color(np.array([1, 0, 0]))

            # depth = depths[0]
            # depth[depth == 10] = np.inf

            # u, v = np.where(depth != np.inf)
            # y = depth[u, v] * ((u - 128.0) / 262.5)
            # x = depth[u, v] * ((v - 128.0) / 262.5)
            # z = depth[u, v]
            # pts = np.stack([x, y, z], axis=-1)

            # pcd_part = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            # pcd_part.paint_uniform_color(np.array([0, 1, 0]))
            # pcd_part.transform(cv_poses[0])

            # axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
            # camera = o3d.geometry.TriangleMesh.create_coordinate_frame()
            # camera.transform(cv_poses[0])
            # o3d.visualization.draw_geometries([pcd, axis, pcd_part, camera])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_procs", default=1, type=int, help="number of parallel processes started"
    )
    parser.add_argument("--rank", default=0, type=int, help="rank of current process")
    parser.add_argument(
        "--out_dir",
        default="/home/isleri/haeni001/data/dif",
        type=str,
        help="output directory",
    )
    parser.add_argument(
        "--mesh_dir",
        default="/home/isleri/haeni001/data/ShapeNetCore.v2/",
        type=str,
        help="mesh directory",
    )
    parser.add_argument(
        "--split_dir",
        default="/home/isleri/haeni001/code/DIF-Net/split/",
        type=str,
        help="split directory",
    )
    args = parser.parse_args()

    print(f"Running process {args.rank} of {args.num_procs}")
    main(args)
    os.remove("./model.obj")
