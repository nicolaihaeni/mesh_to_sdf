import os

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
import pyrender

import matplotlib.pyplot as plt


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
    filenames = []
    for cat in categories:
        with open(os.path.join(args.split_dir, f"{cat}.json"), "r") as f:
            data = json.load(f)
            for mode in ["test", "train"]:
                for filename in data[mode][cat]:
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

        mesh = trimesh.load(
            os.path.join(args.mesh_dir, filename, "models", "model_normalized.obj"),
            force="mesh",
        )

        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("The mesh parameter must be a trimesh mesh.")

        # Normalize the mesh in sphere of radius 1/1.03
        mesh = scale_to_unit_sphere(mesh)

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
            cam_locations = utils.sample_spherical(30, 3.0)
            obj_location = np.zeros((1, 3))
            cv_poses = utils.look_at(cam_locations, obj_location)
            cam_locations = [utils.cv_cam2world_to_bcam2world(m) for m in cv_poses]
            image_size = (256, 256)
            camera = pyrender.IntrinsicsCamera(
                fx=262.5, fy=262.5, cx=128.0, cy=128.0, znear=0.01, zfar=100
            )

            mesh = trimesh.load_mesh(
                os.path.join(args.mesh_dir, filename, "models", "model_normalized.obj"),
            )
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump().sum()
            if not isinstance(mesh, trimesh.Trimesh):
                raise TypeError("The mesh parameter must be a trimesh mesh.")

            mesh = pyrender.Mesh.from_trimesh(mesh)
            scene = pyrender.Scene()
            scene.add(mesh)
            pyrender.Viewer(scene, use_raymond_lighting=True)

            rgbs = []
            depths = []
            masks = []
            for ii, w2c in enumerate(cam_locations):
                # Add camera roll
                roll_matrix = utils.sample_roll_matrix(
                    np.random.uniform(-1, 1, 1) * np.pi
                )
                w2c = roll_matrix @ w2c

                if ii == 0:
                    w2c0 = np.array(utils.get_world2cam_from_blender_cam(w2c))

                if ii == 0:
                    cam_node = scene.add(camera, pose=np.array(w2c))
                else:
                    scene.set_pose(cam_node, pose=np.array(w2c))

                r = pyrender.OffscreenRenderer(*image_size)
                color, depth = r.render(
                    scene, flags=pyrender.constants.RenderFlags.FLAT
                )

                mask = depth != 0
                depth[mask == 0.0] = 100

                rgbs.append(color)
                depths.append(depth)
                masks.append(mask)
                r.delete()

            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(rgbs[0])
            ax[1].imshow(depths[0], cmap="plasma")
            ax[2].imshow(masks[0], cmap="plasma")
            plt.show()

            rgbs = np.stack([r for r in rgbs])
            depths = np.stack([r for r in depths])
            masks = np.stack([r for r in masks])
            hf = h5py.File(out_rgbd_path, "w")
            hf.create_dataset("rgb", data=rgbs, compression="gzip", dtype="f")
            hf.create_dataset("depth", data=depths, compression="gzip", dtype="f")
            hf.create_dataset("mask", data=masks, compression="gzip", dtype="f")
            hf.close()

            # # Visualize
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(surface_points)
            # pcd.normals = o3d.utility.Vector3dVector(surface_normals)
            # colors = np.zeros_like(surface_points)
            # colors[:, 0] = 1
            # pcd.colors = o3d.utility.Vector3dVector(colors)

            # depth = depths[0]
            # depth[depth == 100] = np.inf
            # u, v = np.where(depth != np.inf)
            # x = (u - 128) / 262.5 * depth[u, v]
            # y = (v - 128) / 262.5 * depth[u, v]
            # z = depth[u, v]
            # points = np.stack([x, y, z], axis=-1)

            # pcd_part = o3d.geometry.PointCloud()
            # pcd_part.points = o3d.utility.Vector3dVector(points)
            # colors = np.zeros_like(points)
            # colors[:, 1] = 1
            # pcd_part.colors = o3d.utility.Vector3dVector(colors)

            # axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
            # pcd_part.transform(np.linalg.inv(w2c0))
            # o3d.visualization.draw_geometries([pcd, pcd_part, axis])


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
