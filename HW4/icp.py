"""
Initially written by Ming Hsiao in MATLAB
Redesigned and rewritten by Wei Dong (weidong@andrew.cmu.edu)
"""

import json
import os

import numpy as np
import rerun as rr
import transforms
import tyro
import utils
from PIL import Image


def find_projective_correspondence(
    source_points,
    source_normals,
    target_vertex_map,
    target_normal_map,
    intrinsic,
    T_init,
    dist_diff=0.07,
):
    """
    Args:
        source_points: Source point cloud locations, (N, 3)
        source_normals: Source point cloud normals, (N, 3)
        target_vertex_map: Target vertex map, (H, W, 3)
        target_normal_map: Target normal map, (H, W, 3)
        intrinsic: Intrinsic matrix, (3, 3)
        T_init: Initial transformation from source to target, (4, 4)
        dist_diff: Distance difference threshold to filter correspondences

    Returns:
        source_indices: indices of points in the source point cloud with a valid projective correspondence in the target map, (M, 1)
        target_us: associated u coordinate of points in the target map, (M, 1)
        target_vs: associated v coordinate of points in the target map, (M, 1)
    """
    h, w, _ = target_vertex_map.shape

    R = T_init[:3, :3]
    t = T_init[:3, 3:]

    # Transform source points from the source coordinate system to the target coordinate system
    T_source_points = (R @ source_points.T + t).T

    # Set up initial correspondences from source to target
    source_indices = np.arange(len(source_points)).astype(int)
    target_us, target_vs, target_ds = transforms.project(T_source_points, intrinsic)
    target_us = np.round(target_us).astype(int)
    target_vs = np.round(target_vs).astype(int)

    # TODO: first filter: valid projection
    #mask = np.zeros_like(target_us).astype(bool)
    mask = (target_us >= 0) & (target_us < w) & (target_vs >= 0) & (target_vs < h) & (target_ds > 0)
    # End of TODO

    source_indices = source_indices[mask]
    target_us = target_us[mask]
    target_vs = target_vs[mask]
    T_source_points = T_source_points[mask]

    # TODO: second filter: apply distance threshold
    #mask = np.zeros_like(target_us).astype(bool)
    target_points = target_vertex_map[target_vs, target_us]
    dist = np.linalg.norm(T_source_points - target_points, axis=1)
    mask = dist < dist_diff
    # End of TODO

    source_indices = source_indices[mask]
    target_us = target_us[mask]
    target_vs = target_vs[mask]

    return source_indices, target_us, target_vs


def build_linear_system(source_points, target_points, target_normals, T):
    M = len(source_points)
    assert len(target_points) == M and len(target_normals) == M

    R = T[:3, :3]
    t = T[:3, 3:]

    p_prime = (R @ source_points.T + t).T
    q = target_points
    n_q = target_normals

    A = np.zeros((M, 6))
    b = np.zeros((M,))

    # TODO: build the linear system
    cross = np.cross(p_prime, n_q)   # (M, 3): p'i x n{q_i}
    A[:, :3] = cross
    A[:, 3:] = n_q
    b = -np.sum(n_q * (p_prime - q), axis=1)
    # End of TODO

    return A, b


def pose2transformation(delta):
    w = delta[:3]
    u = np.expand_dims(delta[3:], axis=1)

    T = np.eye(4)

    R = np.array([
        [
            np.cos(w[2]) * np.cos(w[1]),
            -np.sin(w[2]) * np.cos(w[0]) + np.cos(w[2]) * np.sin(w[1]) * np.sin(w[0]),
            np.sin(w[2]) * np.sin(w[0]) + np.cos(w[2]) * np.sin(w[1]) * np.cos(w[1])  # ← fix: cos(w[0])
        ],
        [
            np.sin(w[2]) * np.cos(w[1]),
            np.cos(w[2]) * np.cos(w[0]) + np.sin(w[2]) * np.sin(w[1]) * np.sin(w[0]),
            -np.cos(w[2]) * np.sin(w[0]) + np.sin(w[2]) * np.sin(w[1]) * np.cos(w[0])
        ],
        [
            -np.sin(w[1]),
            np.cos(w[1]) * np.sin(w[0]),
            np.cos(w[1]) * np.cos(w[0])
        ]
    ])

    T[:3, :3] = R
    T[:3, 3:] = u

    return T

def solve(A, b):
    """
    Args:
        A: (6, 6) matrix in the LU formulation, or (N, 6) in the QR formulation
        b: (6, 1) vector in the LU formulation, or (N, 1) in the QR formulation

    Returns:
        delta: (6, 1) vector by solving the linear system. You may directly use dense solvers from numpy.
    """
    # TODO: write your relevant solver
    #return np.zeros((6,))
    delta, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return delta


def icp(
    source_points,
    source_normals,
    target_vertex_map,
    target_normal_map,
    intrinsic,
    T_init=None,
    debug_association=False,
):
    """
    Args:
        source_points: Source point cloud locations, (N, 3)
        source_normals: Source point cloud normals, (N, 3)
        target_vertex_map: Target vertex map, (H, W, 3)
        target_normal_map: Target normal map, (H, W, 3)
        intrinsic: Intrinsic matrix, (3, 3)
        T_init: Initial transformation from source to target, (4, 4)
        debug_association: Visualize association between sources and targets for debug

    Returns:
        T: (4, 4) transformation from source to target
    """

    T = np.eye(4) if T_init is None else T_init

    for i in range(20):
        # TODO: fill in find_projective_correspondences
        source_indices, target_us, target_vs = find_projective_correspondence(
            source_points,
            source_normals,
            target_vertex_map,
            target_normal_map,
            intrinsic,
            T,
        )

        # Select associated source and target points
        corres_source_points = source_points[source_indices]
        corres_target_points = target_vertex_map[target_vs, target_us]
        corres_target_normals = target_normal_map[target_vs, target_us]

        # Debug, if necessary
        if debug_association:
            rr.set_time("icp_iter", sequence=i)
            utils.visualize_correspondences(
                corres_source_points, corres_target_points, T
            )

        # TODO: fill in build_linear_system and solve
        A, b = build_linear_system(
            corres_source_points, corres_target_points, corres_target_normals, T
        )
        delta = solve(A, b)

        # Update and output
        T = pose2transformation(delta) @ T
        loss = np.mean(b**2)
        print(
            "iter {}: avg loss = {:.4e}, inlier count = {}".format(
                i, loss, len(corres_source_points)
            )
        )

    return T


def main(
    path: str,
    source_idx: int = 10,
    target_idx: int = 50,
):
    """
    Args:
        path: path to the dataset folder containing rgb/ and depth/
        source_idx: index of the source frame
        target_idx: index of the target frame
    """

    with open("intrinsics.json") as f:
        intrinsic = np.array(json.load(f)["intrinsic_matrix"]).reshape(3, 3, order="F")

    depth_path = os.path.join(path, "depth")
    normal_path = os.path.join(path, "normal")

    # TUM convention -- uint16 value to float meters
    depth_scale = 5000.0

    # Source: load depth and rescale to meters
    source_depth = (
        np.asarray(Image.open("{}/{}.png".format(depth_path, source_idx))) / depth_scale
    )

    # Unproject depth to vertex map (H, W, 3) and reshape to a point cloud (H*W, 3)
    source_vertex_map = transforms.unproject(source_depth, intrinsic)
    source_points = source_vertex_map.reshape((-1, 3))

    # Load normal map (H, W, 3) and reshape to point cloud normals (H*W, 3)
    source_normal_map = np.load("{}/{}.npy".format(normal_path, source_idx))
    source_normals = source_normal_map.reshape((-1, 3))

    # Similar preparation for target, but keep the image format for projective association
    target_depth = (
        np.asarray(Image.open("{}/{}.png".format(depth_path, target_idx))) / depth_scale
    )
    target_vertex_map = transforms.unproject(target_depth, intrinsic)
    target_normal_map = np.load("{}/{}.npy".format(normal_path, target_idx))

    rr.init("icp", spawn=True)

    # Visualize before ICP
    rr.set_time("step", sequence=0)
    utils.visualize_icp(source_points, target_vertex_map.reshape((-1, 3)), np.eye(4))

    # TODO: fill-in components in ICP
    T = icp(
        source_points,
        source_normals,
        target_vertex_map,
        target_normal_map,
        intrinsic,
        np.eye(4),
        debug_association=False,
    )

    # Visualize after ICP
    rr.set_time("step", sequence=1)
    utils.visualize_icp(source_points, target_vertex_map.reshape((-1, 3)), T)


if __name__ == "__main__":
    tyro.cli(main)