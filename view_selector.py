# the input is a point cloud and multiple viewpoints, given intrinsic and extrinsics
# I need to find a subset of extrinsics that can cover the most points

import numpy as np
import utils
import os
import open3d as o3d
import gen_image
DATA_PATH = "/datax/matterport/17DRP5sb8fy_region0/"
INTRINSICS_DIR = os.path.join(DATA_PATH, "camera_intrinsics")
POINTCLOUD_PATH = os.path.join(DATA_PATH, "segmentations", "region0.ply")
# POINTCLOUD_PATH = "/datax/3rscan/0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca/labels.instances.annotated.v2.ply"
EXTRINSICS_DIR = os.path.join(DATA_PATH, "camera_poses")


def get_intrinsic(pcd):
    """
        pcd: open3d.geometry.PointCloud
        return: a dictionary of intrinsics, containing height, width, fx, fy, cx, cy
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    vis.destroy_window()
    intrinsic = param.intrinsic
    intrinsic_dict = {
        "height": intrinsic.height,
        "width": intrinsic.width,
        "fx": intrinsic.intrinsic_matrix[0, 0],
        "fy": intrinsic.intrinsic_matrix[1, 1],
        "cx": intrinsic.intrinsic_matrix[0, 2],
        "cy": intrinsic.intrinsic_matrix[1, 2],
    }
    print("intrinsic: {}".format(intrinsic_dict))
    return intrinsic_dict
    

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


def select_views(pointcloud, intrinsics, extrinsics, target_coverage=0.99, max_views=10000, hidden_point_removal=True):
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
        boolean_vec = utils.is_in_cone_by_camera_params(pointcloud, intrinsics[i], extrinsics[i], hidden_point_removal=hidden_point_removal)
        boolean_vecs.append(boolean_vec)
        # print("view {}: {}/{}".format(i, np.sum(boolean_vec), num_points))
    print("num_views to select: {}".format(num_views))
    boolean_vecs = np.array(boolean_vecs)
    selected_view_indices = coverage_solver(boolean_vecs, target_coverage, max_views)
    return selected_view_indices


def generate_seed_candidates(pointcloud, percentile=98):
    """
        pointcloud: open3d.geometry.PointCloud
        return: a list of 3d points, the seed candidates
    """
    world_center = np.average(np.array(pointcloud.points), axis=0)
    # the chosen should be 98 percentile and 2 percentile, to avoid outliers
    max_x, max_y, max_z = np.percentile(np.array(pointcloud.points), percentile, axis=0)
    min_x, min_y, min_z = np.percentile(np.array(pointcloud.points), 100 - percentile, axis=0)
    x_candidates = np.linspace(min_x, max_x, 5)
    y_candidates = np.linspace(min_y, max_y, 5)
    z_candidate = np.mean([min_z, max_z])
    seed_candidates = []
    for x in x_candidates[1:-1]:
        for y in y_candidates[1:-1]:
            candidate = np.array([x, y, z_candidate])
            candidate_refined = utils.move_away_from_neighbours(candidate, np.array(pointcloud.points), target_distance=0.5)
            if np.any(np.abs(candidate_refined - candidate) > 1e-6):
                print("candidate: {} -> {}".format(candidate, candidate_refined))
            seed_candidates.append(np.array(candidate_refined))
    return seed_candidates


def generate_views_from_seed(seed, views_per_layer=8):
    """
        seed: 3d point
        the output views are represented by camera extrinsics
        return: a list of camera extrinsics, each is a 4x4 numpy array, world coordinate -> camera coordinate
    """
    # generate 18 views, 3 layers, 6 views per layer
    x, y, z = seed
    start_pitch, delta_pitch = np.pi/6, -np.pi/6
    start_yaw, delta_yaw = 0, 2*np.pi/views_per_layer
    extrinsics = []
    for layer_idx in range(3):
        for view_idx in range(views_per_layer):
            pitch = start_pitch + layer_idx * delta_pitch
            yaw = start_yaw + view_idx * delta_yaw
            look_at_x = x + np.cos(pitch) * np.cos(yaw)
            look_at_y = y + np.cos(pitch) * np.sin(yaw)
            look_at_z = z + np.sin(pitch)
            look_at = np.array([look_at_x, look_at_y, look_at_z])
            extrinsic = utils.compute_extrinsic_matrix(look_at, seed)
            extrinsics.append(extrinsic)
    return extrinsics


def generate_views(pointcloud, percentile=98):
    """
        pointcloud: open3d.geometry.PointCloud
        return: a list of camera extrinsics, each is a 4x4 numpy array, world coordinate -> camera coordinate
    """
    seed_candidates = generate_seed_candidates(pointcloud, percentile)
    extrinsics = []
    for seed in seed_candidates:
        extrinsics += generate_views_from_seed(seed)
    return extrinsics
            


if __name__ == "__main__":
    original_mesh = o3d.io.read_triangle_mesh(POINTCLOUD_PATH)
    pcd = o3d.io.read_point_cloud(POINTCLOUD_PATH)
    # 201726 points
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    # 43220 points
    points = np.asarray(pcd.points)

    # extrinsics = []
    # intrinsics = []
    # for file in os.listdir(EXTRINSICS_DIR):
    #     if file.endswith(".txt"):
    #         # naming rule: 5b9b2794954e4694a45fc424a8643081_pose_0_0.txt
    #         # layer: looking up, middle, down
    #         # view: one for every 60 degrees
    #         intrinsic_id, pose_text, layer, view = file[:-4].split("_")
    #         assert pose_text == "pose"
    #         extrinsic = utils.read_extrinsic(os.path.join(EXTRINSICS_DIR, file))
    #         extrinsics.append(extrinsic)
    #         # print(extrinsic)
    #         intrinsic = utils.read_intrinsic(os.path.join(INTRINSICS_DIR, intrinsic_id + "_intrinsics_" + layer + ".txt"))
    #         intrinsics.append(intrinsic)
    extrinsics = generate_views(pcd)
    intrinsic = get_intrinsic(pcd)
    intrinsics = [intrinsic] * len(extrinsics)

    selected_indices = select_views(pcd, intrinsics, extrinsics, target_coverage=0.999, max_views=10000, hidden_point_removal=False)
    selected_camera_extrinsics = [extrinsics[i] for i in selected_indices]
    utils.visualize_camera_extrinsics(pcd, selected_camera_extrinsics, add_coordinate_frame=False)
    gen_image.generate_rendered_pictures(original_mesh, selected_camera_extrinsics, visible_window=True, wait_time=0)











