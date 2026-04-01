"""
Initially written by Ming Hsiao in MATLAB
Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
"""

import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import tyro
from dataclasses import dataclass, field
from typing import Literal
import matplotlib.pyplot as plt
from solvers import *
from utils import *


plt.ion()

def warp2pi(angle_rad):
    r"""
    Warps an angle in [-pi, pi]. Used in the update step.
    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    angle_rad = angle_rad - 2 * np.pi * np.floor((angle_rad + np.pi) / (2 * np.pi))
    return angle_rad


def init_states(odoms, observations, n_poses, n_landmarks):
    """
    Initialize the state vector given odometry and observations.
    """
    traj = np.zeros((n_poses, 2))
    landmarks = np.zeros((n_landmarks, 2))
    landmarks_mask = np.zeros((n_landmarks), dtype=np.bool)

    for i in range(len(odoms)):
        traj[i + 1, :] = traj[i, :] + odoms[i, :]

    for i in range(len(observations)):
        pose_idx = int(observations[i, 0])
        landmark_idx = int(observations[i, 1])

        if not landmarks_mask[landmark_idx]:
            landmarks_mask[landmark_idx] = True

            pose = traj[pose_idx, :]
            theta, d = observations[i, 2:]

            landmarks[landmark_idx, 0] = pose[0] + d * np.cos(theta)
            landmarks[landmark_idx, 1] = pose[1] + d * np.sin(theta)

    return traj, landmarks


def odometry_estimation(x, i):
    r"""
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from (odometry between pose i and i+1)
    \return odom Odometry (\Delta x, \Delta) in the shape (2, )
    """
    # TODO: return odometry estimation
    odom = np.zeros((2,))

    xi = x[2*i:2*i+2]
    xj = x[2*(i+1):2*(i+1)+2]

    odom = xj - xi
    return odom



def bearing_range_estimation(x, i, j, n_poses):
    r"""
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from
    \param j Index of the landmark to be measured
    \param n_poses Number of poses
    \return obs Observation from pose i to landmark j (theta, d) in the shape (2, )
    """
    # TODO: return bearing range estimations
    obs = np.zeros((2,))
    # pose
    rx, ry = x[2*i], x[2*i+1]

    # landmark
    lx = x[2*n_poses + 2*j]
    ly = x[2*n_poses + 2*j + 1]

    dx = lx - rx
    dy = ly - ry

    theta = np.arctan2(dy, dx)
    d = np.sqrt(dx**2 + dy**2)

    return np.array([theta, d])


def compute_meas_obs_jacobian(x, i, j, n_poses):
    r"""
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from
    \param j Index of the landmark to be measured
    \param n_poses Number of poses
    \return jacobian Derived Jacobian matrix in the shape (2, 4)
    """
    # TODO: return jacobian matrix
    jacobian = np.zeros((2, 4))

    rx, ry = x[2*i], x[2*i+1]
    lx = x[2*n_poses + 2*j]
    ly = x[2*n_poses + 2*j + 1]

    dx = lx - rx
    dy = ly - ry

    q = dx**2 + dy**2
    sqrt_q = np.sqrt(q)

    # θ row
    jacobian[0, 0] = dy / q        # dθ/drx
    jacobian[0, 1] = -dx / q       # dθ/dry
    jacobian[0, 2] = -dy / q       # dθ/dlx
    jacobian[0, 3] = dx / q        # dθ/dly

    # d row
    jacobian[1, 0] = -dx / sqrt_q  # dd/drx
    jacobian[1, 1] = -dy / sqrt_q  # dd/dry
    jacobian[1, 2] = dx / sqrt_q   # dd/dlx
    jacobian[1, 3] = dy / sqrt_q   # dd/dly

    return jacobian


def create_linear_system(
    x, odoms, observations, sigma_odom, sigma_observation, n_poses, n_landmarks
):
    r"""
    \param x State vector x at which we linearize the system.
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

    sqrt_inv_odom = np.linalg.inv(scipy.linalg.sqrtm(sigma_odom))
    sqrt_inv_obs = np.linalg.inv(scipy.linalg.sqrtm(sigma_observation))

    # TODO: First fill in the prior to anchor the 1st pose at (0, 0)
    row = 0
    A[row:row+2, 0:2] = np.eye(2)
    b[row:row+2] = -x[0:2]   # current estimate - desired (0,0)
    row += 2

    # TODO: Then fill in odometry measurements
    for i in range(n_odom):
        z = odoms[i]  # measurement

        # predicted
        h = odometry_estimation(x, i)

        # residual
        r = z - h

        # indices
        idx_i = 2 * i
        idx_j = 2 * (i + 1)

        # Jacobian (same as linear case)
        J_i = -np.eye(2)
        J_j = np.eye(2)

        # fill A
        A[row:row+2, idx_i:idx_i+2] = sqrt_inv_odom @ J_i
        A[row:row+2, idx_j:idx_j+2] = sqrt_inv_odom @ J_j

        # fill b
        b[row:row+2] = sqrt_inv_odom @ r

        row += 2

    # TODO: Then fill in landmark measurements
    for obs in observations:
        i = int(obs[0])
        j = int(obs[1])
        z = obs[2:4]  # (theta, d)

        # predicted
        h = bearing_range_estimation(x, i, j, n_poses)

        # residual
        r = z - h
        r[0] = warp2pi(r[0])  # normalize angle

        # Jacobian (2x4)
        H = compute_meas_obs_jacobian(x, i, j, n_poses)

        # indices
        pose_idx = 2 * i
        landmark_idx = 2 * n_poses + 2 * j

        # split Jacobian
        H_pose = H[:, 0:2]
        H_land = H[:, 2:4]

        # fill A
        A[row:row+2, pose_idx:pose_idx+2] = sqrt_inv_obs @ H_pose
        A[row:row+2, landmark_idx:landmark_idx+2] = sqrt_inv_obs @ H_land

        # fill b
        b[row:row+2] = sqrt_inv_obs @ r

        row += 2

    return csr_matrix(A), b


@dataclass
class Args:
    data: str = "../data/2d_nonlinear.npz"
    method: list[Literal["default", "pinv", "qr", "lu", "qr_colamd", "lu_colamd"]] = (
        field(default_factory=lambda: ["default"])
    )


if __name__ == "__main__":
    args = tyro.cli(Args)

    data = np.load(args.data)

    # Plot gt trajectory and landmarks for a sanity check.
    gt_traj = data["gt_traj"]
    gt_landmarks = data["gt_landmarks"]
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], "b-")
    plt.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], c="b", marker="+")
    plt.show(block=False)

    n_poses = len(gt_traj)
    n_landmarks = len(gt_landmarks)

    odom = data["odom"]
    observations = data["observations"]
    sigma_odom = data["sigma_odom"]
    sigma_landmark = data["sigma_landmark"]

    # Initialize: non-linear optimization requires a good init.
    for method in args.method:
        print(f"Applying {method}")

        traj, landmarks = init_states(odom, observations, n_poses, n_landmarks)

        print("Before optimization")
        plt.figure()
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)
        plt.pause(0.1)
        input("Press Enter to continue...")

        # Iterative optimization
        x = vectorize_state(traj, landmarks)
        for i in range(10):
            A, b = create_linear_system(
                x, odom, observations, sigma_odom, sigma_landmark, n_poses, n_landmarks
            )
            dx, _ = solve(A, b, method)
            x = x + dx

        traj, landmarks = devectorize_state(x, n_poses)

        print("After optimization")
        plt.figure()
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)
        plt.pause(0.1)
        input("Press Enter to continue...")
