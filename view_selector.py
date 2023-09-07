# the input is a point cloud and multiple viewpoints, given intrinsic and extrinsics
# I need to find a subset of extrinsics that can cover the most points

import numpy as np
import utils
import os
import open3d as o3d
# import gen_image
DATA_PATH = "/datax/matterport/17DRP5sb8fy_region0/"
INTRINSICS_DIR = os.path.join(DATA_PATH, "camera_intrinsics")
POINTCLOUD_PATH = os.path.join(DATA_PATH, "segmentations", "region0.ply")
EXTRINSICS_DIR = os.path.join(DATA_PATH, "camera_poses")



def coverage_solver(boolean_vecs, target_coverage=0.99, max_vecs=10000):
    """
        boolean_vecs: n boolean vectors of length m, shape (n, m)
        target_coverage: float
        max_vecs: int
        return: a list of indices of the selected vectors
            the logical or of the selected vectors should cover at least target_coverage of the coverable points
    """
    assert target_coverage <= 1.0
    num_vecs, vec_len = boolean_vecs.shape
    can_be_covered = np.sum(boolean_vecs, axis=0) > 0
    already_covered = np.zeros(vec_len, dtype=bool)
    num_can_be_covered = np.sum(can_be_covered)
    print("can be covered: {}/{}".format(num_can_be_covered, vec_len))
    sol = []
    while np.sum(already_covered) < target_coverage * num_can_be_covered and len(sol) < max_vecs:
        # greedy: find the camera that can cover the most 
        num_covered = np.sum(boolean_vecs[:, can_be_covered], axis=1)
        best_cam_idx = np.argmax(num_covered)
        sol.append(best_cam_idx)
        already_covered = np.logical_or(already_covered, boolean_vecs[best_cam_idx])
        can_be_covered = np.logical_and(can_be_covered, np.logical_not(already_covered))
        print("already covered: {}/{}".format(np.sum(already_covered), num_can_be_covered))
    print("solution: {}".format(sol))
    return sol


def view_selector(pointcloud, intrinsics, extrinsics, target_coverage=0.99, max_views=10000):
    """
        pointcloud: open3d.geometry.PointCloud
        intrinsics: a list of dictionary of intrinsics, containing height, width, fx, fy, cx, cy, k1, k2, p1, p2, k3
        extrinsics: a list of extrinsics, each is a 4x4 numpy array, world coordinate -> camera coordinate
        target_coverage: float, the target coverage of the selected views
        max_views: int, the maximum number of views to select
        return: a list of indices of the selected views
    """
    assert target_coverage <= 1.0
    num_views = len(intrinsics)
    num_points = len(pointcloud.points)
    boolean_vecs = []
    for i in range(num_views):
        boolean_vec = utils.is_in_cone_by_camera_params(pointcloud, intrinsics[i], extrinsics[i])
        boolean_vecs.append(boolean_vec)
        print("view {}: {}/{}".format(i, np.sum(boolean_vec), num_points))
    boolean_vecs = np.array(boolean_vecs)
    selected_view_indices = coverage_solver(boolean_vecs, target_coverage, max_views)
    return selected_view_indices



if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(POINTCLOUD_PATH)
    # 201726 points
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    # 43220 points
    points = np.asarray(pcd.points)
    extrinsics = []
    intrinsics = []
    boolean_vecs = []
    for file in os.listdir(EXTRINSICS_DIR):
        # naming rule: 5b9b2794954e4694a45fc424a8643081_pose_0_0.txt
        if file.endswith(".txt"):
            intrinsic_id, pose_text, layer, view = file[:-4].split("_")
            # layer: looking up, middle, down
            # view: one for every 60 degrees
            assert pose_text == "pose"
            extrinsic = utils.read_extrinsic(os.path.join(EXTRINSICS_DIR, file))
            extrinsics.append(extrinsic)
            # print(extrinsic)
            intrinsic = utils.read_intrinsic(os.path.join(INTRINSICS_DIR, intrinsic_id + "_intrinsics_" + layer + ".txt"))
            intrinsics.append(intrinsic)
    selected_indices = view_selector(pcd, intrinsics, extrinsics, target_coverage=0.99, max_views=10000)
    selected_camera_extrinsics = [extrinsics[i] for i in selected_indices]
    utils.visualize_camera_extrinsics(pcd, selected_camera_extrinsics, add_coordinate_frame=False)











