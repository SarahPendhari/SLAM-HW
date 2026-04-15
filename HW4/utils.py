'''
    Initially written by Ming Hsiao in MATLAB
    Redesigned and rewritten by Wei Dong (weidong@andrew.cmu.edu)
'''

import rerun as rr
import numpy as np


def visualize_icp(source_points, target_points, T):
    R = T[:3, :3]
    t = T[:3, 3:]
    transformed_source = (R @ source_points.T + t).T

    rr.log('icp/source', rr.Points3D(transformed_source, colors=[255, 0, 0]))
    rr.log('icp/target', rr.Points3D(target_points, colors=[0, 255, 0]))


def visualize_correspondences(source_points, target_points, T):
    if len(source_points) != len(target_points):
        print(
            'Error! source points and target points has different length {} vs {}'
            .format(len(source_points), len(target_points)))
        return

    R = T[:3, :3]
    t = T[:3, 3:]
    transformed_source = (R @ source_points.T + t).T

    rr.log('icp/source', rr.Points3D(transformed_source, colors=[255, 0, 0]))
    rr.log('icp/target', rr.Points3D(target_points, colors=[0, 255, 0]))

    lines = np.stack([transformed_source, target_points], axis=1)
    rr.log('icp/correspondences', rr.LineStrips3D(lines))
