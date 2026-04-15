'''
    Initially written by Ming Hsiao in MATLAB
    Redesigned and rewritten by Wei Dong (weidong@andrew.cmu.edu)
'''

import json
import os

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
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
    """
    Args:
        path: path to the dataset folder containing rgb/ and depth/
        start_idx: start frame index
        end_idx: end frame index
        downsample_factor: spatial downsample factor
    """
    with open('intrinsics.json') as f:
        intrinsic = np.array(json.load(f)['intrinsic_matrix']).reshape(3, 3, order='F')
    indices, gt_poses = load_gt_poses(
        os.path.join(path, 'livingRoom2.gt.freiburg'))

    rgb_path = os.path.join(path, 'rgb')
    depth_path = os.path.join(path, 'depth')
    normal_path = os.path.join(path, 'normal')

    # TUM convention
    depth_scale = 5000.0

    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(
            origin='/',
            overrides={'world/map/normals': rrb.EntityBehavior(visible=False)},
        )
    )
    rr.init('icp_fusion', spawn=True, default_blueprint=blueprint)

    # Rotate from Y-down/Z-forward (camera convention) to Z-up for visualization
    R_z_up = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)

    m = Map()

    down_factor = downsample_factor
    intrinsic /= down_factor
    intrinsic[2, 2] = 1

    # Only use pose 0 for 1-th frame for alignment.
    # DO NOT use other gt poses here
    T_cam_to_world = gt_poses[0]

    H, W = int(480 / down_factor), int(680 / down_factor)
    rr.log('world/camera/image', rr.Pinhole(
        image_from_camera=intrinsic,
        width=W,
        height=H,
        camera_xyz=rr.ViewCoordinates.RDF,
    ), static=True)

    traj_gt = []
    traj_est = []
    for i in range(start_idx, end_idx + 1):
        print('loading frame {}'.format(i))
        rr.set_time('frame', sequence=i)

        depth = np.asarray(Image.open('{}/{}.png'.format(depth_path, i))) / depth_scale
        depth = depth[::down_factor, ::down_factor]
        vertex_map = transforms.unproject(depth, intrinsic)

        color_map = np.asarray(Image.open('{}/{}.png'.format(rgb_path, i))).astype(float) / 255.0
        color_map = color_map[::down_factor, ::down_factor]

        normal_map = np.load('{}/{}.npy'.format(normal_path, i))
        normal_map = normal_map[::down_factor, ::down_factor]

        if i > 1:
            print('Frame-to-model icp')
            T_world_to_cam = np.linalg.inv(T_cam_to_world)
            T_world_to_cam = icp(m.points[::down_factor],
                                 m.normals[::down_factor],
                                 vertex_map,
                                 normal_map,
                                 intrinsic,
                                 T_world_to_cam,
                                 debug_association=False)
            T_cam_to_world = np.linalg.inv(T_world_to_cam)
        print('Point-based fusion')
        m.fuse(vertex_map, normal_map, color_map, intrinsic, T_cam_to_world)

        traj_gt.append(R_z_up @ gt_poses[i - 1][:3, 3])
        traj_est.append(R_z_up @ T_cam_to_world[:3, 3])
        if len(traj_gt) > 1:
            rr.log('world/trajectory/gt', rr.LineStrips3D([traj_gt], colors=[0, 255, 0]))
            rr.log('world/trajectory/est', rr.LineStrips3D([traj_est], colors=[255, 0, 0]))

        points_viz = (R_z_up @ m.points.T).T
        normals_viz = (R_z_up @ m.normals.T).T
        rr.log('world/map', rr.Points3D(
            points_viz,
            colors=(m.colors * 255).astype(np.uint8),
        ))
        rr.log('world/map/normals', rr.Arrows3D(
            vectors=normals_viz * 0.02,
            origins=points_viz,
        ))
        rr.log('world/camera', rr.Transform3D(
            translation=R_z_up @ T_cam_to_world[:3, 3],
            mat3x3=R_z_up @ T_cam_to_world[:3, :3],
        ))



if __name__ == '__main__':
    tyro.cli(main)
