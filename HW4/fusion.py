"""
Initially written by Ming Hsiao in MATLAB
Redesigned and rewritten by Wei Dong (weidong@andrew.cmu.edu)
"""

import json
import os

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import transforms
import tyro
from PIL import Image
from preprocess import load_gt_poses


class Map:
    def __init__(self):
        self.points = np.empty((0, 3))
        self.normals = np.empty((0, 3))
        self.colors = np.empty((0, 3))
        self.weights = np.empty((0, 1))
        self.initialized = False

    def merge(self, indices, points, normals, colors, R, t):
        """TODO: implement the merge function

        Args:
            indices: Indices of selected points. Used for IN PLACE modification.
            points: Input associated points, (N, 3)
            normals: Input associated normals, (N, 3)
            colors: Input associated colors, (N, 3)
            R: rotation from camera (input) to world (map), (3, 3)
            t: translation from camera (input) to world (map), (3, )

        Returns:
            None, update map properties IN PLACE
        """
        w = self.weights[indices, 0]  # (N,)
        # Transform input points/normals from camera to world
        world_points = (R @ points.T + t).T
        world_normals = (R @ normals.T).T

        # Weighted average
        self.points[indices] = (w[:, None] * self.points[indices] + world_points) / (w[:, None] + 1)
        self.normals[indices] = (w[:, None] * self.normals[indices] + world_normals) / (w[:, None] + 1)
        # Normalize normals
        norms = np.linalg.norm(self.normals[indices], axis=1, keepdims=True)
        self.normals[indices] = self.normals[indices] / np.where(norms == 0, 1, norms)
        self.colors[indices] = (w[:, None] * self.colors[indices] + colors) / (w[:, None] + 1)
        self.weights[indices, 0] += 1

    def add(self, points, normals, colors, R, t):
        """TODO: implement the add function

        Args:
            points: Input associated points, (N, 3)
            normals: Input associated normals, (N, 3)
            colors: Input associated colors, (N, 3)
            R: rotation from camera (input) to world (map), (3, 3)
            t: translation from camera (input) to world (map), (3, )

        Returns:
            None, update map properties by concatenation
        """
        # Filter out zero/invalid points
        valid = np.linalg.norm(points, axis=1) > 0
        points = points[valid]
        normals = normals[valid]
        colors = colors[valid]

        # Transform from camera to world
        world_points = (R @ points.T + t).T
        world_normals = (R @ normals.T).T

        n = len(world_points)
        self.points = np.concatenate([self.points, world_points], axis=0)
        self.normals = np.concatenate([self.normals, world_normals], axis=0)
        self.colors = np.concatenate([self.colors, colors], axis=0)
        self.weights = np.concatenate([self.weights, np.ones((n, 1))], axis=0)

    def filter_pass1(self, us, vs, ds, h, w):
        """TODO: implement the filter function

        Args:
            us: Putative corresponding u coordinates on an image, (N, 1)
            vs: Putative corresponding v coordinates on an image, (N, 1)
            ds: Putative corresponding depth on an image, (N, 1)
            h: Height of the image projected to
            w: Width of the image projected to

        Returns:
            mask: (N, 1) in bool indicating the valid coordinates
        """
        #return np.zeros_like(us)
        mask = (us >= 0) & (us < w) & (vs >= 0) & (vs < h) & (ds > 0)
        return mask

    def filter_pass2(
        self, points, normals, input_points, input_normals, dist_diff, angle_diff
    ):
        """TODO: implement the filter function

        Args:
            points: Maintained associated points, (M, 3)
            normals: Maintained associated normals, (M, 3)
            input_points: Input associated points, (M, 3)
            input_normals: Input associated normals, (M, 3)
            dist_diff: Distance difference threshold to filter correspondences by positions
            angle_diff: Angle difference threshold to filter correspondences by normals

        Returns:
            mask: (N, 1) in bool indicating the valid correspondences
        """
        #return np.zeros((len(points)))
        dist = np.linalg.norm(points - input_points, axis=1)
        cos_angle = np.sum(normals * input_normals, axis=1)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        mask = (dist < dist_diff) & (angle < angle_diff)
        return mask

    def fuse(
        self,
        vertex_map,
        normal_map,
        color_map,
        intrinsic,
        T,
        dist_diff=0.03,
        angle_diff=np.deg2rad(5),
    ):
        """
        Args:
            vertex_map: Input vertex map, (H, W, 3)
            normal_map: Input normal map, (H, W, 3)
            color_map: Input color map, (H, W, 3)
            intrinsic: Intrinsic matrix, (3, 3)
            T: transformation from camera (input) to world (map), (4, 4)

        Returns:
            None, update map properties on demand
        """
        # Camera to world
        R = T[:3, :3]
        t = T[:3, 3:]

        # World to camera
        T_inv = np.linalg.inv(T)
        R_inv = T_inv[:3, :3]
        t_inv = T_inv[:3, 3:]

        if not self.initialized:
            points = vertex_map.reshape((-1, 3))
            normals = normal_map.reshape((-1, 3))
            colors = color_map.reshape((-1, 3))

            # TODO: add step
            self.add(points, normals, colors, R, t)
            self.initialized = True

        else:
            h, w, _ = vertex_map.shape

            # Transform from world to camera for projective association
            indices = np.arange(len(self.points)).astype(int)
            T_points = (R_inv @ self.points.T + t_inv).T
            R_normals = (R_inv @ self.normals.T).T

            # Projective association
            us, vs, ds = transforms.project(T_points, intrinsic)
            us = np.round(us).astype(int)
            vs = np.round(vs).astype(int)

            # TODO: first filter: valid projection
            mask = self.filter_pass1(us, vs, ds, h, w)
            # Should not happen -- placeholder before implementation
            if mask.sum() == 0:
                return
            # End of TODO

            indices = indices[mask]
            us = us[mask]
            vs = vs[mask]

            T_points = T_points[indices]
            R_normals = R_normals[indices]
            valid_points = vertex_map[vs, us]
            valid_normals = normal_map[vs, us]

            # TODO: second filter: apply thresholds
            mask = self.filter_pass2(
                T_points, R_normals, valid_points, valid_normals, dist_diff, angle_diff
            )
            # Should not happen -- placeholder before implementation
            if mask.sum() == 0:
                return
            # End of TODO

            indices = indices[mask]
            us = us[mask]
            vs = vs[mask]

            updated_entries = len(indices)

            merged_points = vertex_map[vs, us]
            merged_normals = normal_map[vs, us]
            merged_colors = color_map[vs, us]

            # TODO: Merge step - compute weight average after transformation
            self.merge(indices, merged_points, merged_normals, merged_colors, R, t)
            # End of TODO

            associated_mask = np.zeros((h, w)).astype(bool)
            associated_mask[vs, us] = True
            new_points = vertex_map[~associated_mask]
            new_normals = normal_map[~associated_mask]
            new_colors = color_map[~associated_mask]

            # TODO: Add step
            self.add(new_points, new_normals, new_colors, R, t)
            # End of TODO

            added_entries = len(new_points)
            print(
                "updated: {}, added: {}, total: {}".format(
                    updated_entries, added_entries, len(self.points)
                )
            )


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
    with open("intrinsics.json") as f:
        intrinsic = np.array(json.load(f)["intrinsic_matrix"]).reshape(3, 3, order="F")
    indices, gt_poses = load_gt_poses(os.path.join(path, "livingRoom2.gt.freiburg"))
    # TUM convention
    depth_scale = 5000.0

    rgb_path = os.path.join(path, "rgb")
    depth_path = os.path.join(path, "depth")
    normal_path = os.path.join(path, "normal")

    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(
            origin="/",
            overrides={"world/map/normals": rrb.EntityBehavior(visible=False)},
        )
    )
    rr.init("fusion", spawn=True, default_blueprint=blueprint)

    # Rotate from Y-down/Z-forward (camera convention) to Z-up for visualization
    R_z_up = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)

    m = Map()

    down_factor = downsample_factor
    intrinsic /= down_factor
    intrinsic[2, 2] = 1

    H, W = int(480 / down_factor), int(680 / down_factor)
    rr.log("world/camera/image", rr.Pinhole(
        image_from_camera=intrinsic,
        width=W,
        height=H,
        camera_xyz=rr.ViewCoordinates.RDF,
    ), static=True)

    for i in range(start_idx, end_idx + 1):
        print("Fusing frame {:03d}".format(i))
        rr.set_time("frame", sequence=i)

        source_depth = (
            np.asarray(Image.open("{}/{}.png".format(depth_path, i))) / depth_scale
        )
        source_depth = source_depth[::down_factor, ::down_factor]
        source_vertex_map = transforms.unproject(source_depth, intrinsic)

        source_color_map = (
            np.asarray(Image.open("{}/{}.png".format(rgb_path, i))).astype(float)
            / 255.0
        )
        source_color_map = source_color_map[::down_factor, ::down_factor]

        source_normal_map = np.load("{}/{}.npy".format(normal_path, i))
        source_normal_map = source_normal_map[::down_factor, ::down_factor]

        m.fuse(
            source_vertex_map,
            source_normal_map,
            source_color_map,
            intrinsic,
            gt_poses[i],
        )

        points_viz = (R_z_up @ m.points.T).T
        normals_viz = (R_z_up @ m.normals.T).T
        rr.log(
            "world/map",
            rr.Points3D(
                points_viz,
                colors=(m.colors * 255).astype(np.uint8),
            ),
        )
        rr.log(
            "world/map/normals",
            rr.Arrows3D(
                vectors=normals_viz * 0.05,
                origins=points_viz,
            ),
        )
        rr.log(
            "world/camera",
            rr.Transform3D(
                translation=R_z_up @ gt_poses[i][:3, 3],
                mat3x3=R_z_up @ gt_poses[i][:3, :3],
            ),
        )


if __name__ == "__main__":
    tyro.cli(main)