import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import transforms
import tyro
from fusion import Map
from icp import icp
from PIL import Image
from preprocess import load_gt_poses


def main(
    path: str,
    start_idx: int = 1,
    end_idx: int = 200,
    downsample_factor: int = 2,
):
    with open('intrinsics.json') as f:
        intrinsic = np.array(json.load(f)['intrinsic_matrix']).reshape(3, 3, order='F')
    indices, gt_poses = load_gt_poses(os.path.join(path, 'livingRoom2.gt.freiburg'))

    rgb_path = os.path.join(path, 'rgb')
    depth_path = os.path.join(path, 'depth')
    normal_path = os.path.join(path, 'normal')
    depth_scale = 5000.0

    m = Map()
    down_factor = downsample_factor
    intrinsic /= down_factor
    intrinsic[2, 2] = 1

    T_cam_to_world = gt_poses[0]

    traj_gt = []
    traj_est = []

    for i in range(start_idx, end_idx + 1):
        print('processing frame {}'.format(i))

        depth = np.asarray(Image.open('{}/{}.png'.format(depth_path, i))) / depth_scale
        depth = depth[::down_factor, ::down_factor]
        vertex_map = transforms.unproject(depth, intrinsic)

        color_map = np.asarray(Image.open('{}/{}.png'.format(rgb_path, i))).astype(float) / 255.0
        color_map = color_map[::down_factor, ::down_factor]

        normal_map = np.load('{}/{}.npy'.format(normal_path, i))
        normal_map = normal_map[::down_factor, ::down_factor]

        if i > 1:
            T_world_to_cam = np.linalg.inv(T_cam_to_world)
            T_world_to_cam = icp(m.points[::down_factor],
                                 m.normals[::down_factor],
                                 vertex_map,
                                 normal_map,
                                 intrinsic,
                                 T_world_to_cam)
            T_cam_to_world = np.linalg.inv(T_world_to_cam)

        m.fuse(vertex_map, normal_map, color_map, intrinsic, T_cam_to_world)

        traj_gt.append(gt_poses[i - 1][:3, 3].copy())
        traj_est.append(T_cam_to_world[:3, 3].copy())

    traj_gt = np.array(traj_gt)
    traj_est = np.array(traj_est)

    np.save('traj_gt.npy', traj_gt)
    np.save('traj_est.npy', traj_est)
    print('saved traj_gt.npy and traj_est.npy')

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(traj_gt[:, 0], traj_gt[:, 1], traj_gt[:, 2],
            'g-', linewidth=2, label='Ground Truth')
    ax.plot(traj_est[:, 0], traj_est[:, 1], traj_est[:, 2],
            'r-', linewidth=2, label='Estimated')

    ax.scatter(*traj_gt[0], c='green', s=100, zorder=5, marker='o', label='Start')
    ax.scatter(*traj_gt[-1], c='black', s=100, zorder=5, marker='x', label='End')

    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)
    ax.set_title('Camera Trajectory: Estimated vs Ground Truth', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('trajectory_3d.png', dpi=150, bbox_inches='tight')
    print('saved trajectory_3d.png')


if __name__ == '__main__':
    tyro.cli(main)