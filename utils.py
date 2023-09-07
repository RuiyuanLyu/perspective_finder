import os, sys
import open3d as o3d
import numpy as np
import json
from queue import Queue
import pickle
import cv2
import matplotlib.pyplot as plt
import copy 

VISUALIZATION_FOLDER = "visualization"

####################################################################################
# visualization utils
####################################################################################
def visualize_2d_projection(mesh):
    """
        mesh: o3d.geometry.TriangleMesh
        project the mesh to the 2D plane z=0
    """
    vertices = np.asarray(mesh.vertices)
    z_axis_noised = np.random.normal(0, 0.02, size=vertices.shape[0])
    projected_vertices = np.concatenate((vertices[:, :2], z_axis_noised.reshape(-1, 1)), axis=1)
    projected_point_cloud = o3d.geometry.PointCloud()
    projected_point_cloud.points = o3d.utility.Vector3dVector(projected_vertices)
    # projected_point_cloud.points = o3d.utility.Vector3dVector(vertices)
    projected_mesh, _ = projected_point_cloud.compute_convex_hull()
    projected_mesh.paint_uniform_color([1, 0.0, 0.0])
    o3d.visualization.draw_geometries([projected_mesh, mesh, projected_point_cloud])


def visualize_freqency(data, bins=100, title=None, xlabel=None, ylabel=None):
    plt.hist(data, bins=bins)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
        plt.savefig(os.path.join(VISUALIZATION_FOLDER, title + ".png"))
    else:
        plt.savefig(os.path.join(VISUALIZATION_FOLDER, "frequency.png"))
    plt.close()


def visualize_2d_map(data, title=None, xlabel=None, ylabel=None, show=False):
    plt.imshow(data.T, cmap="gray", origin="lower")
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
        plt.savefig(os.path.join(VISUALIZATION_FOLDER, title + ".png"))
    else:
        plt.savefig(os.path.join(VISUALIZATION_FOLDER, "2d_map.png"))
    if show:
        plt.show()
    plt.close()


def visualize_path(path, background, path_alpha=0.2, show=True, save=True):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    fig, ax = plt.subplots()
    def update(frame):
        if frame == 0:
            ax.clear()
        ax.imshow(background.T, origin="lower")
        ax.imshow(path[frame].covered_map.T, origin="lower", alpha=path_alpha)
        ax.scatter(path[frame].x, path[frame].y, c="r")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if frame > 0:
            x1, y1, x2, y2 = path[frame-1].x, path[frame-1].y, path[frame].x, path[frame].y
            # print(x1, y1, "->", x2, y2)
            ax.plot([x1, x2], [y1, y2], c="r")
        ax.set_title("step: %d" % frame)
    ani = FuncAnimation(fig, update, frames=len(path), interval=500)
    if show:
        plt.show()
    if save:
        ani.save(os.path.join(VISUALIZATION_FOLDER, "path.gif"), writer="pillow")
        # also save the last frame
        update(len(path)-1)
        plt.savefig(os.path.join(VISUALIZATION_FOLDER, "path.png"))
    plt.close()


def visualize_camera_extrinsics(background, camera_extrinsics, add_coordinate_frame=True):
    """
        background: o3d.geometry.TriangleMesh / o3d.geometry.PointCloud
        camera_extrinsics: list of 4*4 numpy.ndarray, world coordinate -> camera coordinate
        add_coordinate_frame: whether to add the coordinate frame
        return: None, just visualize
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    vis.add_geometry(background)
    for i in range(len(camera_extrinsics)):
        extrinsic = camera_extrinsics[i]
        coord, look_at = extrinsic_to_coord_and_lookat(extrinsic)
        _, mesh_arrow, _, _ = get_arrow(coord, look_at-coord)
        vis.add_geometry(mesh_arrow)
    if add_coordinate_frame:
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
    vis.run()


def visualize_camera_seen_points(background, camera_extrinsic, points, point_colors=None):
    """
        background: o3d.geometry.TriangleMesh / o3d.geometry.PointCloud
        camera_extrinsics: a 4*4 numpy.ndarray, world coordinate -> camera coordinate
        points: boolean numpy.ndarray, length: N
        point_colors: the color of the points, length: 3
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    # draw the camera
    coord, look_at = extrinsic_to_coord_and_lookat(camera_extrinsic)
    _, mesh_arrow, _, _ = get_arrow(coord, look_at-coord)
    vis.add_geometry(mesh_arrow)
    
    # draw the background
    modified_background = copy.deepcopy(background)
    modified_background.paint_uniform_color([0.5, 0.5, 0.5])
    if point_colors is None:
        point_colors = [0, 0, 1]
    individual_colors = np.array([point_colors if p else [0.5, 0.5, 0.5] for p in points])
    modified_background.colors = o3d.utility.Vector3dVector(individual_colors)
    vis.add_geometry(modified_background)
    # visualize
    vis.run()


####################################################################################
# object utils (thick line, and arrows)
####################################################################################
def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=True)
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat
 
 
def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)
 
    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)
 
    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                            z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))
 
    qTrans_Mat *= scale
    return qTrans_Mat
 

def get_arrow(begin=[0,0,0],vec=[0,0,1]):
    z_unit_Arr = np.array([0, 0, 1])
    begin = begin
    end = np.add(begin,vec)
    vec_Arr = np.array(end) - np.array(begin)
    vec_len = np.linalg.norm(vec_Arr)
    sphere_size = vec_len / 10
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60, origin=[0, 0, 0])
 
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.2 * 1 ,
        cone_radius=0.06 * 1,
        cylinder_height=0.8 * 1,
        cylinder_radius=0.04 * 1
    )
    mesh_arrow.paint_uniform_color([0, 1, 0])
    mesh_arrow.compute_vertex_normals()
 
    mesh_sphere_begin = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_size, resolution=20)
    mesh_sphere_begin.translate(begin)
    mesh_sphere_begin.paint_uniform_color([0, 1, 1])
    mesh_sphere_begin.compute_vertex_normals()
    mesh_sphere_end = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_size, resolution=20)
    mesh_sphere_end.translate(end)
    mesh_sphere_end.paint_uniform_color([0, 1, 1])
    mesh_sphere_end.compute_vertex_normals()
 
    rot_mat = caculate_align_mat(vec_Arr)
    mesh_arrow.rotate(rot_mat, center=False)
    mesh_arrow.translate(np.array(begin))  # 0.5*(np.array(end) - np.array(begin))
    return mesh_frame, mesh_arrow, mesh_sphere_begin, mesh_sphere_end


####################################################################################
# input/output utils
####################################################################################
def read_intrinsic(file_path, convention='matterport'):
    output_dict = {}
    if convention == 'matterport':
        # read a txt file, values spaced by space
        with open(file_path, 'r') as f:
            line = f.readline()
            splited = line.split(' ')
            assert len(splited) == 11
            output_dict['width'] = int(splited[0])
            output_dict['height'] = int(splited[1])
            output_dict['fx'] = float(splited[2])
            output_dict['fy'] = float(splited[3])
            output_dict['cx'] = float(splited[4])
            output_dict['cy'] = float(splited[5])
            output_dict['k1'] = float(splited[6])
            output_dict['k2'] = float(splited[7])
            output_dict['p1'] = float(splited[8])
            output_dict['p2'] = float(splited[9])
            output_dict['k3'] = float(splited[10])
    else:
        raise NotImplementedError
    return output_dict


def read_extrinsic(file_path, convention='matterport'):
    """
        convention: 'matterport' only
        matterport: camera coordinate -> world coordinate, need to np.linalg.inv
        read a txt file, values spaced by space.
        output: 4 x 4 extrinsic matrix, world corrdinate -> camera coordinate

    """
    output = np.zeros([4, 4])
    if convention == 'matterport':
        with open(file_path, 'r') as f:
            for i in range(4):
                line = f.readline().strip()
                splited = line.split(' ')
                assert len(splited) == 4, splited
                output[i, :] = np.array([float(s) for s in splited])
        output = np.linalg.inv(output)
    else:
        raise NotImplementedError
    return output


####################################################################################
# 2D map utils
####################################################################################
def pillarize_point_cloud(point_cloud, pillar_size):
    """
        point_cloud: o3d.geometry.PointCloud
        pillar_size: float, the size of the pillar
        return: pillarized point cloud, count map
    """
    points = np.array(point_cloud.points)
    grid_indices = ((points[:,:2] - np.min(points[:,:2], axis=0)) / pillar_size).astype(np.int32)
    
    pillar_dict = {}
    for i, point in enumerate(points):
        grid_index = tuple(grid_indices[i])
        if grid_index not in pillar_dict:
            pillar_dict[grid_index] = []
        pillar_dict[grid_index].append(i)

    pillar_point_counts = {}
    for grid_idx, point_indices in pillar_dict.items():
        pillar_point_counts[grid_idx] = len(point_indices)
    # visualize_count_freq(list(pillar_point_counts.values()))

    pillared_point_colors = np.zeros((len(points), 3))
    import matplotlib.cm as cm
    count_normalizer = np.percentile(list(pillar_point_counts.values()), 75)
    # print("count_normalizer: {}".format(count_normalizer))

    dense_map = sparse_to_dense(pillar_point_counts)
    dense_map /= count_normalizer
    dense_map = np.clip(dense_map, 0, 1)
    visualize_2d_map(dense_map, title="dense_map")
    edge_map = double_thresholding(dense_map, 0.25, 1)
    visualize_2d_map(edge_map, title="edge_map")
    touchable_map = (dense_map > 0) - edge_map
    remove_small_islands(touchable_map, 15)
    np.save(os.path.join(VISUALIZATION_FOLDER, "touchable_map.npy"), touchable_map)
    np.save(os.path.join(VISUALIZATION_FOLDER, "edge_map.npy"), edge_map)
    visualize_2d_map(touchable_map*0.5 + edge_map, title="touchable_map")
    
    edge_indices = np.array(np.where(edge_map == 1)).T
    for grid_idx, point_count in pillar_point_counts.items():
        # color = cm.hot(point_count / count_normalizer)
        color = [0.5, 0.5, 0.5]
        pillared_point_colors[pillar_dict[grid_idx]] = color[:3]
    for grid_idx in edge_indices:
        pillared_point_colors[pillar_dict[tuple(grid_idx)]] = [0, 0, 1]

    pillar_point_cloud = o3d.geometry.PointCloud()
    pillar_point_cloud.points = point_cloud.points
    pillar_point_cloud.colors = o3d.utility.Vector3dVector(np.array(pillared_point_colors))
    return pillar_point_cloud, dense_map


def get_island_coordinates(grid, x, y):
    if (x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0]) or grid[x][y] != 1):
        return []
    
    grid[x][y] = -1
    coordinates = [(x, y)]
    coordinates += get_island_coordinates(grid, x + 1, y)
    coordinates += get_island_coordinates(grid, x - 1, y)
    coordinates += get_island_coordinates(grid, x, y + 1)
    coordinates += get_island_coordinates(grid, x, y - 1)
    
    return coordinates


def get_all_island_coordinates(map):
    grid = np.array(map).copy()
    sizes = []
    coordinates_list = []
    
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            if grid[x][y] == 1:
                coordinates = get_island_coordinates(grid, x, y)
                size = len(coordinates)
                sizes.append(size)
                coordinates_list.append(coordinates)
    
    return sizes, coordinates_list


def remove_small_islands(map, threshold):
    """
        removes the islands whose size is strictly smaller than the threshold
        map: 2D numpy array, 1: island, 0: water
        threshold: int, the threshold of the island size
    """
    sizes, coordinates_list = get_all_island_coordinates(map)
    for i in range(len(sizes)):
        if sizes[i] < threshold:
            for x, y in coordinates_list[i]:
                map[x][y] = 0
    return map


def sparse_to_dense(sparse_dict):
    """
        sparse_dict: {grid_idx: value}
    """
    x, y = np.max(np.array(list(sparse_dict.keys())), axis=0) + 1
    dense_map = np.zeros((x, y))
    for grid_idx, value in sparse_dict.items():
        dense_map[grid_idx] = value
    return dense_map


def sparse_to_dense_with_default(sparse_list, map_size, default_value):
    """
        sparse_list: [(x1, y1), (x2, y2), ...]
    """
    dense_map = np.zeros(map_size)
    dense_map[sparse_list] = default_value
    return dense_map


def double_thresholding(data, weak, strong):
    """
        data: 2D numpy array
        weak: float
        strong: float
        return: a 2D numpy array with values 0, 1
    """
    assert weak < strong
    map = np.zeros_like(data)
    strong_mask = data >= strong
    weak_mask = np.logical_and(data >= weak, data < strong)
    map[strong_mask] = 2
    map[weak_mask] = 1
    
    def dfs(x, y):
        if x < 0 or x >= map.shape[0] or y < 0 or y >= map.shape[1]:
            return
        if map[x, y] != 1:
            return
        map[x, y] = 2
        for i in range(-1, 2):
            for j in range(-1, 2):
                dfs(x + i, y + j)
    
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i, j] == 2:
                dfs(i, j)
    map[map == 1] = 0
    map[map == 2] = 1
    return map


def find_outlier_indices(points, threshold):
    from scipy.spatial import distance
    # 计算每个点与其周围点的距离
    distances = distance.cdist(points, points, 'euclidean')
    
    # 计算每个点的平均距离
    avg_distances = np.mean(distances, axis=1)
    
    # 计算每个点的标准差
    std_distances = np.std(distances, axis=1)
    
    # 根据平均距离和标准差判断是否为outlier
    is_outlier = np.logical_or(distances > threshold * avg_distances[:, np.newaxis],
                               distances > threshold * std_distances[:, np.newaxis])

    indices = np.where(np.sum(is_outlier, axis=1) > 0)[0]
    return indices


def get_dist_map_and_angle_map(x: int, y: int, input_map: np.ndarray, angle: float = 0):
    """
        x, y: the position of the hero
        angle: the towards angle of the hero
        map: the map of the scene, only its shape is used
    """
    x_map, y_map = np.meshgrid(np.arange(input_map.shape[0]), np.arange(input_map.shape[1]))
    x_map = x_map.T - x
    y_map = y_map.T - y
    dist_map = np.sqrt(x_map**2 + y_map**2)
    angle_map = np.arctan2(y_map, x_map) # in the range [-pi, pi]
    angle_map = subtract_angle(angle_map, angle)
    return dist_map, angle_map


def compute_no_collision_map(input_map, radius):
    print("computing non collision map...")
    result = np.zeros_like(input_map)
    one_indices = np.argwhere(input_map == 1)
    print("num lands to search: %d" % len(one_indices))
    for one_idx in one_indices:
        dist_map, _ = get_dist_map_and_angle_map(one_idx[0], one_idx[1], input_map)
        occupied_map = dist_map <= radius
        if not np.logical_and(occupied_map, 1-input_map).any():
            # not any void in the circle
            result[one_idx[0], one_idx[1]] = 1
    print("non collision lands num: %d" % np.sum(result))
    print("non collision land percentage: %.3f" % (np.sum(result)/np.sum(input_map) * 100))
    return result


####################################################################################
# bezier curve utils
####################################################################################
def bezier_curve_point(control_points, t):
    # NOTE: 当t以匀速移动时，Q(t)在沿曲线方向上并不是匀速移动。要做到这一点，显然曲线长度是需要考虑的。
    control_points = np.array(control_points, dtype=np.float64)
    n = len(control_points) - 1
    result = np.zeros_like(control_points[0])
    for i in range(n + 1):
        result += control_points[i] * binomial_coefficient(n, i) * (t**i) * (1 - t)**(n - i)
    return result


def bezier_curve(control_points, interval=0.01):
    t_values = np.arange(0, 1 + interval, interval)
    curve_points = np.array([bezier_curve_point(control_points, t) for t in t_values])
    return curve_points


def binomial_coefficient(n, k):
    return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))


def draw_bezier():
    control_points = np.random.rand(100, 2)
    # 计算贝塞尔曲线上的点
    t_values = np.linspace(0, 1, 100)
    curve_points = np.array([bezier_curve_point(control_points, t) for t in t_values])

    # 绘制贝塞尔曲线
    plt.plot(control_points[:, 0], control_points[:, 1], 'ro-', label='Control Points')
    plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='Bezier Curve')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bezier Curve')
    plt.grid(True)
    plt.show()


def subtract_angle(a, b):
    """
        return the angle between a and b, in the range [-pi, pi]
    """
    return np.mod(a - b + np.pi, 2*np.pi) - np.pi


####################################################################################
# 3d math/data utils
####################################################################################
def axes_to_rotation(obb):
    R = np.array(obb["normalizedAxes"]).reshape(3, 3)
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return [roll, pitch, yaw]


def convert_seg_group(seg_group):
    obb = seg_group["obb"]
    roll, pitch, yaw = axes_to_rotation(obb)
    main_info = {
        "obj_id": str(seg_group["objectId"]),
        "obj_type": str(seg_group["label"]),
        "psr": {
            "position": {
                "x": obb["centroid"][0],
                "y": obb["centroid"][1],
                "z": obb["centroid"][2]
            },
            "scale": {
                "x": obb["axesLengths"][0],
                "y": obb["axesLengths"][1],
                "z": obb["axesLengths"][2]
            },
            "rotation": {
                "x": roll,
                "y": pitch,
                "z": yaw
            }
        }
    }

    return main_info


def compute_extrinsic_matrix(lookat_point, camera_coords):
    """
        lookat_point: 3D point in world coordinate
        camera_coords: 3D point in world coordinate
        NOTE: the camera convention is xyz:RDF
    """
    camera_direction = lookat_point - camera_coords
    camera_direction_normalized = camera_direction / np.linalg.norm(camera_direction)
    up_vector = np.array([0, 0, -1])  
    right_vector = np.cross(up_vector, camera_direction_normalized)
    right_vector_normalized = right_vector / np.linalg.norm(right_vector)
    true_up_vector = np.cross(camera_direction_normalized, right_vector_normalized)
    view_direction_matrix = np.vstack((right_vector_normalized, true_up_vector, camera_direction_normalized))
    extrinsic = np.zeros((4, 4))
    extrinsic[:3, :3] = view_direction_matrix
    extrinsic[:3, 3] = - view_direction_matrix @ camera_coords
    extrinsic[3, 3] = 1
    return extrinsic


def extrinsic_to_coord_and_lookat(extrinsic):
    """
        extrinsic: 4*4 numpy array, world coordinate to camera coordinate
        return: camera_coords, lookat_point
    """
    camera_direction_normalized = extrinsic[2, :3]
    camera_coords = - extrinsic[:3, :3].T @ extrinsic[:3, 3]
    lookat_point = camera_coords + camera_direction_normalized
    return camera_coords, lookat_point


def draw_camera(camera_pose):
    """
        camera_pose: 4*4 numpy array, camera coordinate to world coordinate
    """
    zhui = np.array([[0,0,0], [0, 0, 1]]) # [[0,0,0], [-0.5, -0.5, 1], [0.5, -0.5, 1], [-0.5, 0.5, 1], [0.5, 0.5, 1]])
    aa = o3d.geometry.PointCloud(points = o3d.utility.Vector3dVector(zhui))
    aa.transform(camera_pose)

    posi = np.array([[camera_pose[0][3],camera_pose[1][3],camera_pose[2][3]]])
    dire = np.array([[camera_pose[0][2],camera_pose[1][2],camera_pose[2][2]]])
    dire = dire + posi
    camera = o3d.geometry.PointCloud()
    camera.points = o3d.utility.Vector3dVector(posi)

    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector([[0, 1]])
    lines_pcd.points = aa.points
    lines_pcd.paint_uniform_color((0, 0, 1))

    return lines_pcd


def compute_camera_extrinsic_obb(camera_pos, obb):
    # compute the view point according to the bounding box
    # obb contains position, scale, rotation
    item_centroid = np.array(obb["centroid"])
    item_scales = np.array(obb["axesLengths"])
    item_rotation = axes_to_rotation(obb)    
    lookat_point = np.array((item_centroid[0], item_centroid[1], camera_pos[2]))
    camera_coords = np.array(camera_pos)
    extrinsic = compute_extrinsic_matrix(lookat_point, camera_coords)
    return extrinsic


def normalize_camera2d(x, y, fx, fy, cx, cy):
    x_normalized = (x - cx) / fx
    y_normalized = (y - cy) / fy
    return np.array([x_normalized, y_normalized])


def camera2d_to_world3d(points, extrinsic):
    """
        convert 2d/3d points in camera coordinate to a 3d point in world coordinate
        points: n x 2 or n x 3
        depth is 1
    """
    assert len(points.shape) == 2
    assert points.shape[1] == 2 or points.shape[1] == 3
    if points.shape[1] == 2:
        points = np.hstack([points, np.ones([points.shape[0], 1])])
    translation = extrinsic[:3, 3]
    rotation = extrinsic[:3, :3]
    points_in_world = np.matmul(points, rotation.T) - translation
    return points_in_world


def world_coord_to_camera_coord(points, extrinsic):
    """
        points: n x 3 in world coordinate
        extrinsic: 4 x 4
        returns: n x 3 in camera coordinate
    """
    assert len(points.shape) == 2
    assert points.shape[1] == 3
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3]
    points_in_camera = rotation @ points.T + translation.reshape([-1, 1])
    return points_in_camera.T


def camera_coord_to_2d_image(points, intrinsic):
    """
        points: n x 3 in camera coordinate
        intrinsic: dict
        returns: n x 2 in 2d image
    """
    assert len(points.shape) == 2
    assert points.shape[1] == 3
    fx, fy = intrinsic['fx'], intrinsic['fy']
    cx, cy = intrinsic['cx'], intrinsic['cy']
    new_x = points[:, 0] * fx / points[:, 2] + cx
    new_y = points[:, 1] * fy / points[:, 2] + cy
    points_2d = np.hstack([new_x.reshape([-1, 1]), new_y.reshape([-1, 1])])
    return points_2d


def get_vision_cone(intrinsic, extrinsic):
    """
        get the vision cone of a camera
        returns the 4 corners of the vision cone, and the camera itself's coordinate in the world
        returns: corners, coord
    """
    coord, look_at = extrinsic_to_coord_and_lookat(extrinsic)
    width, height = intrinsic['width'], intrinsic['height']
    fx, fy = intrinsic['fx'], intrinsic['fy']
    cx, cy = intrinsic['cx'], intrinsic['cy']
    # get the 4 corners of the vision cone
    # the 4 corners are in the order of top-left, top-right, bottom-left, bottom-right
    # first, get the 4 corners in the camera coordinate
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]])
    corners = normalize_camera2d(corners[:, 0], corners[:, 1], fx, fy, cx, cy)
    # second, send the corners to the world coordinate
    corners = camera2d_to_world3d(corners, extrinsic)
    return corners, coord


# @profile
def is_in_cone_by_camera_params(pointcloud, intrinsic, extrinsic, hidden_point_removal=False, cone_depth=10000):
    """
        pointcloud: o3d.geometry.PointCloud
        intrinsic: dict
        extrinsic: 4 x 4
        returns: n x 1 bool array
    """
    points = np.array(pointcloud.points)
    assert len(points.shape) == 2
    assert points.shape[1] == 3
    corners, coord = extrinsic_to_coord_and_lookat(extrinsic)
    if hidden_point_removal:
        _, visible_indices = pointcloud.hidden_point_removal(coord, cone_depth)
        output_indices = np.zeros(points.shape[0], dtype=np.bool)
        output_indices[visible_indices] = True
    else:
        output_indices = np.ones(points.shape[0], dtype=np.bool)
    points_in_camera = world_coord_to_camera_coord(points, extrinsic)
    # eliminate those points behind the camera
    output_indices = np.logical_and(output_indices, points_in_camera[:, 2] > 0)
    points_2d = camera_coord_to_2d_image(points_in_camera, intrinsic)
    # eliminate those points outside the image
    xmax, ymax = intrinsic['width'], intrinsic['height']
    output_indices = np.logical_and(output_indices, points_2d[:, 0] >= 0)
    output_indices = np.logical_and(output_indices, points_2d[:, 0] < xmax)
    output_indices = np.logical_and(output_indices, points_2d[:, 1] >= 0)
    output_indices = np.logical_and(output_indices, points_2d[:, 1] < ymax)

    return output_indices

