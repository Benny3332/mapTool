import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import cv2


def plan_to_pos_v3(start, goal, obstacles, G=None, vis=False):
    if G is None:
        G = nx.Graph()

    # 获取障碍物地图的形状
    rows, cols = obstacles.shape

    # 创建节点索引映射
    node_index_map = {}
    index_node_map = {}
    node_index = 0

    # 第一步：添加所有非障碍物节点
    for col in range(cols):
        for row in range(rows):
            if not obstacles[row, col]:  # 只考虑非障碍物节点
                node_index_map[(row, col)] = node_index
                index_node_map[node_index] = (row, col)
                G.add_node(node_index)
                node_index += 1

    # 检查起点和终点是否为非障碍物节点并使用断言
    start_tuple = (int(start[1]), int(start[0]))  # 转换为 (y, x) 格式
    goal_tuple = (int(goal[1]), int(goal[0]))  # 转换为 (y, x) 格式

    assert start_tuple in node_index_map, f"Start node {start} is an obstacle."
    assert goal_tuple in node_index_map, f"Goal node {goal} is an obstacle."

    # 第二步：建立节点之间的边关系
    for col in range(cols):
        for row in range(rows):
            if not obstacles[row, col]:  # 只考虑非障碍物节点
                current_index = node_index_map[(row, col)]

                # 检查四个方向的邻居
                neighbors = []
                if row > 0 and not obstacles[row - 1, col]:  # 上方
                    neighbors.append((row - 1, col))
                if row < rows - 1 and not obstacles[row + 1, col]:  # 下方
                    neighbors.append((row + 1, col))
                if col > 0 and not obstacles[row, col - 1]:  # 左侧
                    neighbors.append((row, col - 1))
                if col < cols - 1 and not obstacles[row, col + 1]:  # 右侧
                    neighbors.append((row, col + 1))

                for neighbor in neighbors:
                    if not obstacles[neighbor]:
                        neighbor_index = node_index_map[neighbor]
                        G.add_edge(current_index, neighbor_index)

    # 使用Dijkstra算法寻找最短路径
    try:
        path_indices = nx.dijkstra_path(G, source=node_index_map[start_tuple], target=node_index_map[goal_tuple])
    except nx.NetworkXNoPath:
        print("No path found")
        return None

    # 将路径索引转换回坐标
    path = [index_node_map[index] for index in path_indices]

    if vis:
        # 创建可视化的障碍物地图
        obs_map_vis = (obstacles[:, :, None] * 255).astype(np.uint8)
        obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])

        # 绘制起点和终点
        obs_map_vis = cv2.circle(obs_map_vis, (int(start[0]), int(start[1])), 3, (255, 0, 0), -1)  # 起点为蓝色
        obs_map_vis = cv2.circle(obs_map_vis, (int(goal[0]), int(goal[1])), 3, (0, 0, 255), -1)  # 终点为红色

        # 绘制路径
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            obs_map_vis = cv2.line(obs_map_vis, (p1[1], p1[0]), (p2[1], p2[0]), (0, 255, 0), 2)  # 路径为绿色

        # 调整窗口大小
        window_name = "planned path"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1000, 500)  # 设置窗口大小为 1000x500 像素
        cv2.imshow(window_name, obs_map_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return path


# 示例用法
if __name__ == "__main__":
    # 创建一个 50x100 的随机障碍物地图
    rows, cols = 50, 100
    obstacles = np.random.choice([True, False], size=(rows, cols), p=[0.3, 0.7])  # 30% 障碍物

    # 确保起点和终点是非障碍物节点
    start = [0, 0]  # (x, y) 格式
    goal = [99, 49]  # (x, y) 格式

    # 如果起点或终点是障碍物，重新选择
    while obstacles[int(start[1]), int(start[0])] or obstacles[int(goal[1]), int(goal[0])]:
        start = [np.random.randint(0, cols), np.random.randint(0, rows)]
        goal = [np.random.randint(0, cols), np.random.randint(0, rows)]

    path = plan_to_pos_v3(start, goal, obstacles, vis=True)
    print("Path:", path)



