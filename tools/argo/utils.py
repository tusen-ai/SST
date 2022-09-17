import torch
from torch import Tensor
from SO3 import quat_to_mat

LABEL_ATTR = (
    "tx_m",
    "ty_m",
    "tz_m",
    "length_m",
    "width_m",
    "height_m",
    "qw",
    "qx",
    "qy",
    "qz",
)

def cuboid_to_vertices(cuboid):
    r"""Return the cuboid vertices in the destination reference frame.

        5------4
        |\\    |\\
        | \\   | \\
        6--\\--7  \\
        \\  \\  \\ \\
    l    \\  1-------0    h
        e    \\ ||   \\ ||   e
        n    \\||    \\||   i
        g    \\2------3    g
        t      width.     h
            h.               t.

    Returns:
        (8,3) array of cuboid vertices.
    """
    dims_lwh_m = cuboid[:, 3:6]
    xyz_m = cuboid[:, :3]
    quat_wxyz = cuboid[:, 6:10]
    unit_vertices_obj_xyz_m: Tensor = torch.as_tensor(
        [
            [+1, +1, +1],  # 0
            [+1, -1, +1],  # 1
            [+1, -1, -1],  # 2
            [+1, +1, -1],  # 3
            [-1, +1, +1],  # 4
            [-1, -1, +1],  # 5
            [-1, -1, -1],  # 6
            [-1, +1, -1],  # 7
        ],
        device=dims_lwh_m.device,
        dtype=dims_lwh_m.dtype,
    )
    # Transform unit polygons.
    vertices_ego: Tensor = (
        dims_lwh_m[:, None] / 2.0
    ) * unit_vertices_obj_xyz_m[None]
    R = quat_to_mat(quat_wxyz)
    t = xyz_m
    vertices_ego = vertices_ego @ R.transpose(-2, -1) + t[:, None]
    return vertices_ego