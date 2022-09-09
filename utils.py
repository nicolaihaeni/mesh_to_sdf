import numpy as np
from mathutils import Matrix, Vector


def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)


def sample_roll_matrix(theta):
    return Matrix(
        (
            (np.cos(theta), -np.sin(theta), 0, 0),
            (np.sin(theta), np.cos(theta), 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
    )


def sample_spherical(n, radius=1.0):
    xyz = np.random.normal(size=(n, 3))
    xyz = normalize(xyz) * radius
    xyz[:, 1] = abs(xyz[:, 1])
    return xyz


# All the following functions follow the opencv convention for camera coordinates.
def look_at(cam_location, point):
    # Cam points in positive z direction
    forward = point - cam_location
    forward = normalize(forward)

    tmp = np.array([0.0, -1.0, 0.0])

    right = np.cross(tmp, forward)
    right = normalize(right)

    up = np.cross(forward, right)
    up = normalize(up)

    mat = np.stack((right, up, forward, cam_location), axis=-1)

    hom_vec = np.array([[0.0, 0.0, 0.0, 1.0]])

    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])

    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat


def cv_cam2world_to_bcam2world(cv_cam2world):
    """
    :cv_cam2world: numpy array.
    :return:
    """
    R_bcam2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))

    cam_location = Vector(cv_cam2world[:3, -1].tolist())
    cv_cam2world_rot = Matrix(cv_cam2world[:3, :3].tolist())

    cv_world2cam_rot = cv_cam2world_rot.transposed()
    cv_translation = -1.0 * cv_world2cam_rot @ cam_location

    blender_world2cam_rot = R_bcam2cv @ cv_world2cam_rot
    blender_translation = R_bcam2cv @ cv_translation

    blender_cam2world_rot = blender_world2cam_rot.transposed()
    blender_cam_location = -1.0 * blender_cam2world_rot @ blender_translation

    blender_matrix_world = Matrix(
        (
            blender_cam2world_rot[0][:] + (blender_cam_location[0],),
            blender_cam2world_rot[1][:] + (blender_cam_location[1],),
            blender_cam2world_rot[2][:] + (blender_cam_location[2],),
            (0, 0, 0, 1),
        )
    )

    return blender_matrix_world


# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_world2cam_from_blender_cam(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.decompose()[
        0:2
    ]  # Matrix_world returns the cam2world matrix.
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2cv[0][:] + (T_world2cv[0],),
            R_world2cv[1][:] + (T_world2cv[1],),
            R_world2cv[2][:] + (T_world2cv[2],),
            (0, 0, 0, 1),
        )
    )
    return RT
