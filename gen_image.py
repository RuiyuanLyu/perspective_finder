import os
import sys
import open3d as o3d
import numpy as np
import json
from queue import Queue
import pickle
import cv2
import matplotlib.pyplot as plt
import utils
from path_finder import find_path
import shutil
import time

O3D_VERSION = utils.O3D_VERSION
DATA_DIR = "/datax/3rscan/0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca/"
# data_dir = "/datax/3rscan/0ad2d3a1-79e2-2212-9b99-a96495d9f7fe"
render_output_dir = 'render_output/'
"""
mesh.refined.v2.obj
    Reconstructed mesh
mesh.refined.mtl
    Corresponding material file 
mesh.refined_0.png
    Corresponding mesh texture
sequence.zip
    Calibrated RGB-D sensor stream with color and depth frames, camera poses
labels.instances.annotated.v2.ply
    Visualization of semantic segmentation
mesh.refined.0.010000.segs.v2.json
    Over-segmentation of annotation mesh
semseg.v2.json
    Instance segmentation of the mesh (contains the labels)
"""
MESH_FILE = "mesh.refined.v2.obj"
MATERIAL_FILE = "mesh.refined.mtl"
TEXTURE_FILE = "mesh.refined_0.png"
SEMANTIC_VISUAL_FILE = "labels.instances.annotated.v2.ply"
OVER_SEG_FILE = "mesh.refined.0.010000.segs.v2.json"
INSTANCE_SEG_FILE = "semseg.v2.json"
instance_labels_to_remove = ["wall", "floor", "ceiling"]
# remove the rendered output and create a new directory
if os.path.exists(render_output_dir):
    shutil.rmtree(render_output_dir, ignore_errors=True)
os.mkdir(render_output_dir)
##########################################################################################
# preparation complete
##########################################################################################
# now load the data


def get_obb_frame(obb_idx):
    assert obb_idx < len(seg_groups)
    obb = o3d.geometry.OrientedBoundingBox()
    obb.center = seg_groups[obb_idx]['obb']['centroid']
    # trans the normalized axes to the 3*3 numpy.ndarray
    # 转置归一化的轴到3*3的numpy.ndarray
    obb.R = np.array(seg_groups[obb_idx]['obb']
                     ['normalizedAxes']).reshape(3, 3).T

    obb.extent = seg_groups[obb_idx]['obb']['axesLengths']
    obb_frame = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    obb_frame.paint_uniform_color([1, 0, 0])  # 设置边框颜色
    return obb_frame


def show_obb(mesh, obb_idx):
    obb_frame = get_obb_frame(obb_idx)
    o3d.visualization.draw_geometries([mesh, obb_frame])


def generate_rendered_pictures(mesh, extrinsic_trajectory=None, visible_window=True, wait_time=0.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=visible_window)
    # 创建一个摄像机并设置参数
    ctr = vis.get_view_control()
    vis.add_geometry(mesh)
    vis.add_geometry(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
    # x: red, y: green, z: blue
    # vis.run()
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(os.path.join(render_output_dir, "screenshot.png"))
    # 将图像保存到文件
    if extrinsic_trajectory is not None:
        num_views = len(extrinsic_trajectory)
    else:
        num_views = 20
    print("initial view extrinsic: \n{}".format(
        ctr.convert_to_pinhole_camera_parameters().extrinsic))
    for i in range(num_views):
        if extrinsic_trajectory is not None:
            vis.clear_geometries()
            vis.add_geometry(mesh)
            # vis.add_geometry(get_obb_frame(i))
            # # add the center of the frame
            # # vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=seg_groups[i]['obb']['centroid']))
            # vis.poll_events()
            # vis.update_renderer()
            # vis.capture_screen_image(os.path.join(render_output_dir, "screenshot_{}_bev.png".format(i)))
            param = ctr.convert_to_pinhole_camera_parameters()
            param.extrinsic = extrinsic_trajectory[i]
            # print("input view extrinsic: \n{}".format(param.extrinsic))
        else:
            param = ctr.convert_to_pinhole_camera_parameters()
            extrinsic = np.array(param.extrinsic)
            print("input view extrinsic: \n{}".format(extrinsic))
            extrinsic[:3, 3] += np.array([0.1, 0.1, 0.1])
            param.extrinsic = extrinsic
        if O3D_VERSION == "9":
            ctr.convert_from_pinhole_camera_parameters(param)
        else:
            ctr.convert_from_pinhole_camera_parameters(param, True)
        print("result view extrinsic: \n{}".format(
            ctr.convert_to_pinhole_camera_parameters().extrinsic))
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join(
            render_output_dir, "screenshot_{}.png".format(i)), True)
        if wait_time > 0:
            time.sleep(wait_time)
    print("finished rendering %d images" % num_views)
    vis.destroy_window()


def make_camera_trajectory(point_cloud, height, pillar_resolution=0.1, num_views=20):
    output_trajectory = []
    x_min, y_min = np.min(np.array(point_cloud.points), axis=0)[:2]
    pillarized_point_cloud, pillarized_count_map = utils.pillarize_point_cloud(
        point_cloud, pillar_resolution)
    count_normalizer = np.percentile(pillarized_count_map, 90)
    pillarized_count_map = pillarized_count_map / count_normalizer
    dense_map = np.clip(pillarized_count_map, 0, 1)
    edge_map = utils.double_thresholding(dense_map, 0.25, 1)
    touchable_map = (dense_map > 0) - edge_map
    utils.remove_small_islands(touchable_map, 15)
    path = find_path(touchable_map)
    utils.visualize_path(path, edge_map + touchable_map*0.3, show=False)
    control_points = []
    for node in path:
        x, y, theta = node.x, node.y, node.angle
        x = x * pillar_resolution + pillar_resolution / 2 + x_min
        y = y * pillar_resolution + pillar_resolution / 2 + y_min
        control_points.append([x, y])
    bezier_curve = utils.bezier_curve(control_points, 1/num_views)
    for i in range(len(bezier_curve)-1):
        x1, y1 = bezier_curve[i]
        x2, y2 = bezier_curve[i+1]
        camera_pos = np.array([x1, y1, height])
        lookat_pos = np.array([x2, y2, height])
        extrinsic = utils.compute_extrinsic_matrix(lookat_pos, camera_pos)
        output_trajectory.append(extrinsic)
    return output_trajectory


if __name__ == "__main__":
    if O3D_VERSION == "9":
        mesh = o3d.io.read_triangle_mesh(os.path.join(DATA_DIR, MESH_FILE))
    else:
        mesh = o3d.io.read_triangle_mesh(
            os.path.join(DATA_DIR, MESH_FILE), True)
    mesh_over = o3d.io.read_triangle_mesh(
        os.path.join(DATA_DIR, OVER_SEG_FILE))
    semseg = json.load(open(os.path.join(DATA_DIR, INSTANCE_SEG_FILE)))
    seg_groups = semseg["segGroups"]
    seg_groups = [seg_group for seg_group in seg_groups if seg_group["label"]
                  not in instance_labels_to_remove]
    for seg_group in seg_groups:
        print(seg_group["label"])
    point_cloud = o3d.io.read_point_cloud(
        os.path.join(DATA_DIR, SEMANTIC_VISUAL_FILE))

    main_info = []
    for seg_group in seg_groups:
        main_info.append(utils.convert_seg_group(seg_group))

    world_center = (np.max(np.array(mesh.vertices), axis=0) +
                    np.min(np.array(mesh.vertices), axis=0))/2
    extrinsic_trajectory = make_camera_trajectory(
        point_cloud, height=0.2, pillar_resolution=0.1, num_views=20)
    generate_rendered_pictures(mesh, extrinsic_trajectory)
    # generate_rendered_pictures(mesh)
    utils.visualize_camera_extrinsics(mesh, extrinsic_trajectory)

