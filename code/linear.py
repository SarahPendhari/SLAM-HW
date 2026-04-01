"""
Initially written by Ming Hsiao in MATLAB
Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
"""

import time
import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
import tyro
from dataclasses import dataclass, field
from typing import Literal
import matplotlib.pyplot as plt
from solvers import *
from utils import *

plt.ion() 


def create_linear_system(
    odoms, observations, sigma_odom, sigma_observation, n_poses, n_landmarks
):
    r"""
    \param odoms Odometry measurements between i and i+1 in the global coordinate system. Shape: (n_odom, 2).
    \param observations Landmark measurements between pose i and landmark j in the global coordinate system. Shape: (n_obs, 4).
    \param sigma_odom Shared covariance matrix of odometry measurements. Shape: (2, 2).
    \param sigma_observation Shared covariance matrix of landmark measurements. Shape: (2, 2).

    \return A (M, N) Jacobian matrix.
    \return b (M, ) Residual vector.
    where M = (n_odom + 1) * 2 + n_obs * 2, total rows of measurements.
          N = n_poses * 2 + n_landmarks * 2, length of the state vector.
    """

    n_odom = len(odoms)
    n_obs = len(observations)

    M = (n_odom + 1) * 2 + n_obs * 2
    N = n_poses * 2 + n_landmarks * 2

    A = np.zeros((M, N))
    b = np.zeros((M,))

    # Prepare Sigma^{-1/2}.
    sqrt_inv_odom = np.linalg.inv(scipy.linalg.sqrtm(sigma_odom))
    sqrt_inv_obs = np.linalg.inv(scipy.linalg.sqrtm(sigma_observation))

    # TODO: First fill in the prior to anchor the 1st pose at (0, 0)
    row = 0
    A[row:row+2, 0:2] = np.eye(2)
    b[row:row+2] = np.array([0.0, 0.0])    
    
    # TODO: Then fill in odometry measurements
    row = 2

    for i in range(n_odom):
        z = odoms[i]  # shape (2,) -> [dx, dy]

        # indices in state vector
        idx_i = 2 * i
        idx_j = 2 * (i + 1)

        # Fill Jacobian
        # -I for r^t
        A[row:row+2, idx_i:idx_i+2] = -sqrt_inv_odom @ np.eye(2)

        # +I for r^{t+1}
        A[row:row+2, idx_j:idx_j+2] = sqrt_inv_odom @ np.eye(2)

        # RHS = z (since we write A x = b)
        b[row:row+2] = sqrt_inv_odom @ z

        row += 2

    # TODO: Then fill in landmark measurements
    for obs in observations:
        i = int(obs[0])   # pose index
        k = int(obs[1])   # landmark index
        z = obs[2:4]      # [dx, dy]

        # indices
        pose_idx = 2 * i
        landmark_idx = 2 * n_poses + 2 * k

        # -I for pose
        A[row:row+2, pose_idx:pose_idx+2] = -sqrt_inv_obs @ np.eye(2)

        # +I for landmark
        A[row:row+2, landmark_idx:landmark_idx+2] = sqrt_inv_obs @ np.eye(2)

        # RHS
        b[row:row+2] = sqrt_inv_obs @ z

        row += 2

    return csr_matrix(A), b


@dataclass
class Args:
    data: str = "../data/2d_linear.npz"
    """Path to npz file."""
    method: list[Literal["default", "pinv", "qr", "lu", "qr_colamd", "lu_colamd"]] = (
        field(default_factory=lambda: ["default"])
    )
    repeats: int = 1
    """Number of repeats in evaluation efficiency. Increase to ensure stability."""


if __name__ == "__main__":
    args = tyro.cli(Args)

    data = np.load(args.data)

    # Plot gt trajectory and landmarks for a sanity check.
    gt_traj = data["gt_traj"]
    gt_landmarks = data["gt_landmarks"]
    plt.figure()
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], "b-", label="gt trajectory")
    plt.scatter(
        gt_landmarks[:, 0], gt_landmarks[:, 1], c="b", marker="+", label="gt landmarks"
    )
    plt.legend()
    plt.show(block=False)  

    n_poses = len(gt_traj)
    n_landmarks = len(gt_landmarks)

    odoms = data["odom"]
    observations = data["observations"]
    sigma_odom = data["sigma_odom"]
    sigma_landmark = data["sigma_landmark"]

    # Build a linear system
    A, b = create_linear_system(
        odoms, observations, sigma_odom, sigma_landmark, n_poses, n_landmarks
    )

    # Solve with the selected method
    for method in args.method:
        print(f"Applying {method}")

        total_time = 0
        total_iters = args.repeats
        for i in range(total_iters):
            start = time.time()
            x, R = solve(A, b, method)
            residual = np.linalg.norm(A @ x - b)
            print(f"  Residual ||Ax - b|| = {residual:.6f}")
            print(f"  x range: [{x.min():.3f}, {x.max():.3f}]")
            print(f"  x[:6] = {x[:6]}")  # first 3 poses
            end = time.time()
            total_time += end - start
        print(f"{method} takes {total_time / total_iters}s on average")

        if R is not None:
            fig, ax = plt.subplots()
            ax.spy(R)
            ax.set_title(f"Sparsity pattern ({method})")
            plt.show(block=False)

        traj, landmarks = devectorize_state(x, n_poses)

        plt.figure()
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)
        plt.title(f"Trajectory and landmarks ({method})")
        plt.show(block=False)
        plt.pause(0.1)

    input("Press Enter to close all plots...")
    plt.close('all')
