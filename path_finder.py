# i need to find a trajectory for the camera to cover most part of the scene
# we have a 2d camera, the scope is a sector.

# the imput is 2d array, a input_map of the scene, 0: void, 1: ground
import numpy as np
import utils
from tqdm import tqdm

HERO_RADIUS = 1.5
MOVE_SPEED = 8
ANGLE_SPEED = np.pi/4
VISION_RANGE = 15
VISION_ANGLE = np.pi/2  # twice of half angle
VISION_AREA = VISION_RANGE**2 * VISION_ANGLE
GOAL_PERCENTAGE = 0.6




class Node:
    def __init__(self, x: int, y: int, move_angle: float, look_angle: float, input_map: np.ndarray, parent: 'Node' = None):
        assert isinstance(input_map, np.ndarray) and input_map.ndim == 2
        self.x = x
        self.y = y
        self.move_angle = move_angle
        self.look_angle = look_angle
        self.parent = parent
        prev_map = parent.covered_map if parent else np.zeros_like(input_map)
        self.covered_map = get_covered_map(input_map, prev_map, x, y, look_angle)
        self.g = 0  # 抵达这个点已经花掉的实际代价
        self.h = np.inf  # 抵达终点还需要的估计代价

    def f(self):
        return self.g + self.h

    def is_goal(self, input_map, interest_points_to_look):
        percentage = np.sum(self.covered_map) / np.sum(input_map)
        interest_looked = np.all(self.covered_map[interest_points_to_look[:, 0], interest_points_to_look[:, 1]])
        return percentage >= GOAL_PERCENTAGE and interest_looked

    def __eq__(self, other):
        ans = self.x == other.x and self.y == other.y and self.move_angle == other.move_angle and self.look_angle == other.look_angle
        ans = ans and np.all(self.covered_map == other.covered_map)
        # if ans:
        #     print("an equal node found")
        return ans

    def __lt__(self, other):
        return self.f() < other.f()

    def __str__(self):
        return "(x: %d, y: %d, move_angle: %.2f, look_angle: %.2f)" % (self.x, self.y, self.move_angle, self.look_angle)
    
    def __hash__(self) -> int:
        return hash((self.x, self.y, self.move_angle, self.look_angle, tuple(self.covered_map.reshape(-1))))

@profile
def astar_search(start_points, input_map, no_collision_map=None, interest_points_to_look=None, add_new_seeds=False, max_iter=10000, patience=1000):
    if no_collision_map is None:
        no_collision_map = utils.compute_no_collision_map(
            input_map, HERO_RADIUS)
    interest_map = np.zeros_like(input_map)
    if interest_points_to_look is not None:
        interest_map = utils.sparse_to_dense_with_default(
            interest_points_to_look, interest_map.shape, 1)
        input_map = np.logical_or(input_map, interest_map)
    else:
        interest_points_to_look = np.empty((0, 2), dtype=np.int32)

    open_list = utils.QuickList()
    closed_list = utils.QuickList()
    open_list.extend(start_points)
    best_coverage = 0
    best_num_viewed_interests = 0
    iter_count = 0
    patience_count = 0

    pbar = tqdm(total=max_iter)
    while open_list:
        iter_count += 1
        patience_count += 1
        open_list.sort()
        while len(open_list) > max_iter:
            open_list.remove(open_list[-1])
        current_node = open_list[0]
        # print("current node f: %.2f, g: %.2f, h: %.2f" % (current_node.f(), current_node.g, current_node.h))
        # print("current node: " + str(current_node))
        current_coverage = np.sum(current_node.covered_map) / np.sum(input_map)
        current_num_viewed_interests = np.sum(current_node.covered_map[interest_points_to_look[:, 0], interest_points_to_look[:, 1]])
        if current_coverage > best_coverage:
            if current_coverage > best_coverage:
                best_coverage = current_coverage
                patience_count = 0
            best_node = current_node
            best_num_viewed_interests = current_num_viewed_interests
        pbar.update(1)
        pbar.set_description("open list size: %d, current best coverage: %.2f%%, num of interest points viewed: %d/%d" % (
            len(open_list), best_coverage*100, best_num_viewed_interests, len(interest_points_to_look)))
        open_list.remove(current_node)
        closed_list.append(current_node)

        if current_node.is_goal(input_map, interest_points_to_look) or iter_count > max_iter or patience_count > patience:
            break

        neighbors = get_neighbors(current_node, input_map, no_collision_map)
        for neighbor in neighbors:
            if neighbor in closed_list:
                continue

            # dist = np.sqrt((neighbor.x - current_node.x)**2 + (neighbor.y - current_node.y)**2)
            # rotation = np.abs(subtract_angle(neighbor.angle, current_node.angle))
            g = current_node.g + 1
            if neighbor not in open_list or g < neighbor.g:
                neighbor.g = g
                neighbor.h = heuristic(neighbor, input_map)
                neighbor.parent = current_node

                if neighbor not in open_list:
                    open_list.append(neighbor)
        if add_new_seeds:
            new_seeds = get_start_nodes(
                input_map, num_seeds=1, no_collision_map=no_collision_map)
            open_list.extend(new_seeds)

    print("search finished, best coverage: %.2f%%" % (best_coverage * 100))
    current_node = best_node
    path = []
    while current_node:
        path.append(current_node)
        current_node = current_node.parent
    return path[::-1]


def get_covered_map(input_map: np.ndarray, prev_map: np.ndarray, x: int, y: int, theta: float, vision_range=VISION_RANGE, vision_angle=VISION_ANGLE):
    """
        prev_map: the covered input_map before the move
        x, y: the position after the move
        theta: the towards angle after the move
    """
    dist_map, angle_map = utils.get_dist_map_and_angle_map(
        x, y, input_map, theta)
    new_map = np.logical_and(dist_map <= vision_range,
                             np.abs(angle_map) <= vision_angle/2)
    new_map = np.logical_or(new_map, prev_map)
    new_map = np.logical_and(new_map, input_map)
    return new_map


def get_neighbors(node, input_map, no_collision_map, move_speed=MOVE_SPEED, angle_speed=ANGLE_SPEED):
    # 返回当前节点的相邻节点
    dist_map, angle_map = utils.get_dist_map_and_angle_map(
        node.x, node.y, input_map, node.move_angle)
    # 限制移动范围和方向
    move_map = np.logical_and(dist_map <= move_speed,
                              np.abs(angle_map) <= angle_speed)
    # 限制不会撞到障碍物或者出界
    feasible_map = np.logical_and(move_map, no_collision_map)
    input_map_to_show = move_map * 0.4 + feasible_map * 0.4 + input_map * 0.2
    # utils.visualize_2d_map(map_to_show, show=True)
    # 生成相邻节点
    neighbors = []
    # utils.visualize_2d_map(input_map_to_show, save=False, show=True)
    for x in range(feasible_map.shape[0]):
        for y in range(feasible_map.shape[1]):
            if feasible_map[x, y]:
                new_move_angle = utils.subtract_angle(angle_map[x, y], -node.move_angle)
                # get the new look angle
                for k in range(5):
                    new_look_angle = utils.subtract_angle(node.look_angle, (k-2)*angle_speed)
                    neighbors.append(Node(x, y, new_move_angle, new_look_angle, input_map, node))
                    # print("new neighbor:" + str(neighbors[-1]))

    return neighbors


def heuristic(node, input_map, interest_points_to_look=None):
    # 估计从当前节点到看见整个地图的代价
    overall_num = np.sum(input_map)
    covered_num = np.sum(node.covered_map)
    cover_cost = (overall_num - covered_num) / VISION_AREA * 4 # 1 is a hyperparameter to adjust
    # cover_cost = np.ceil(cover_cost)
    # 估计从当前节点到看见所有感兴趣点的移动和转角代价
    if interest_points_to_look is None:
        return cover_cost
    interest_map = utils.sparse_to_dense_with_default(
        interest_points_to_look, np.zeros_like(input_map), 1)
    not_covered_interest_map = np.logical_and(
        interest_map, np.logical_not(node.covered_map))
    not_covered_indices = np.array(np.where(not_covered_interest_map)).T # shape (n, 2)
    dist_cost, angle_cost = 0, 0
    if not_covered_indices.shape[0] == 0:
        dist_cost_map, angle_cost_map = utils.get_dist_map_and_angle_map(
            node.x, node.y, input_map, node.look_angle)
        dist_costs = dist_cost_map[interest_points_to_look[:, 0], interest_points_to_look[:, 1]]
        angle_costs = np.abs(angle_cost_map[interest_points_to_look[:, 0], interest_points_to_look[:, 1]])
        # dist_costs, angle_costs = np.ceil(dist_costs), np.ceil(angle_costs)
        dist_cost = np.sum(dist_costs) / MOVE_SPEED / 2
        angle_cost = np.sum(angle_costs) / ANGLE_SPEED / 2 

    return cover_cost 


def get_start_nodes(input_map, no_collision_map=None):
    max_x, max_y = input_map.shape
    if no_collision_map is None:
        no_collision_map = utils.compute_no_collision_map(
            input_map, HERO_RADIUS)
    output_nodes = []
    directions = np.array(
        [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]])
    x_map, y_map = np.meshgrid(
        np.arange(input_map.shape[0]), np.arange(input_map.shape[1]))
    x_map, y_map = x_map.T, y_map.T
    for direction in directions:
        angle = np.arctan2(-direction[1], -direction[0])  # point to the center
        value_map = x_map * direction[0] + y_map * direction[1]
        value_map = np.float32(value_map) * no_collision_map - \
            1000000000 * (1 - no_collision_map)
        max_value = np.max(value_map)
        max_points = np.argwhere(value_map >= max_value - 1)
        # now select a random point
        seed = np.random.choice(len(max_points))
        x, y = max_points[seed]
        new_node = Node(x, y, angle, angle, input_map)
        new_node.h = heuristic(new_node, input_map)
        output_nodes.append(new_node)
        print(output_nodes[-1])
    return output_nodes


def find_path(input_map, no_collision_map=None, interest_points_to_look=None, add_new_seeds=False, max_iter=10000, patience=300):
    """
        input_map: 2d array, 0: void, 1: ground
        no_collision_map: 2d array, 0: void, 1: accessible ground
        interest_points_to_look: 2d array, shape (n, 2), the points to look
        add_new_seeds: bool, whether to add new seeds during the search
        max_iter: int, the maximum number of iterations
        patience: int, the maximum number of iterations without improvement
        print_interval: int, the interval to print the current best coverage
        return: a list of nodes, the path
    """
    if no_collision_map is None:
        no_collision_map = utils.compute_no_collision_map(
            input_map, HERO_RADIUS)
    start_nodes = get_start_nodes(input_map, no_collision_map)
    path = astar_search(start_nodes, input_map, no_collision_map,
                        interest_points_to_look, add_new_seeds, max_iter, patience)
    return path


if __name__ == "__main__":
    import open3d as o3d
    PCD_FILE = "/datax/3rscan/0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca/labels.instances.annotated.v2.ply"
    PCD = o3d.io.read_point_cloud(PCD_FILE)
    count_map, _, _ = utils.make_2d_count_map(PCD, 0.1)
    _, _, MAP = utils.get_2d_dense_edge_touchable_map(count_map, 0.75)
    path = find_path(MAP)
    utils.visualize_path(path, MAP)
