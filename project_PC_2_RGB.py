import numpy as np
import open3d as o3d
import cv2
from numba import jit
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from gmlDogRecordFilePath import file_path,file_pre_path

# new application function
tr_matrix_lidar_2_depth = np.array([
    [-1.00799236e-02,  3.41871575e-01,  9.39692621e-01, 1.545999999999999978e-02],
    [-9.99949194e-01, -3.37640348e-03, -9.49790910e-03, 5.539000000000000173e-02],
    [-7.42837030e-05, -9.39740617e-01,  3.41888239e-01, 1.501099999999999934e-01],
    [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])

tr_matrix_lidar_2_dog = np.array([
    [0.93912, -0.00685, 0.34352, 0.00000],
    [0.00000, 0.99980, 0.01994, 0.00000],
    [-0.34359, -0.01873, 0.93893, 0.00000],
    [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])

# cjs revise
# tr_matrix_lidar_2_depth = np.array([
#     [0.0236432,  0.2727882,  0.9617836, 0.0297088],
#     [-0.9994875, -0.0143159,  0.0286304, -0.01],
#     [0.0215788, -0.9619676,  0.2723099, 0.2],
#     [0, 0, 0, 1]])

#ld revise
# tr_matrix_lidar_2_depth = np.array([
#     [0.0270124,  0.2756647,  0.9608743, 0.0297088],
#     [-0.9990314, -0.0259554,  0.0355314, -0.01],
#     [0.0347347, -0.9609034,  0.2746966, -0.12],
#     [0, 0, 0, 1]])


#ld
# tr_matrix_lidar_2_depth = np.array([
#     [0.0232266, -0.999284, 0.0298772, 0.0297088],
#     [0.280359, -0.0221751, -0.969636, -0.176122],
#     [0.959611, 0.0306658, 0.279653, -0.226056],
#     [0, 0, 0, 1]
# ])

# cjs
# tr_matrix_lidar_2_depth = np.array([
#     [0.0236437, -0.999487, 0.0215808, 0.0297088],
#     [0.272788, -0.0143172, -0.961966, -0.105938],
#     [0.961785, 0.0286315, 0.27231, -0.226056],
#     [0, 0, 0, 1]
# ])

# 可视化点云
def visualize_point_cloud(points, points_in_fov, T_world_to_camera, fov_horizontal, fov_vertical):
    # 创建点云并设置点
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # 计算Z轴的最大最小值以便归一化
    z_values = np.array(points)[:, 2]  # 提取所有点的Z坐标
    # 应用自定义的热图颜色映射
    colors = colormap_jet(z_values)
    # 将颜色分配给点云
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # 设置视野内的点云
    pcd_in_fov = o3d.geometry.PointCloud()
    pcd_in_fov.points = o3d.utility.Vector3dVector(points_in_fov)
    # 计算Z轴的最大最小值以便归一化
    fov_z_values = np.array(points)[:, 2]  # 提取所有点的Z坐标
    # 应用自定义的热图颜色映射
    fov_colors = colormap_jet(fov_z_values)
    # 将颜色分配给点云
    # pcd_in_fov.colors = o3d.utility.Vector3dVector(fov_colors)
    pcd_in_fov.paint_uniform_color([1, 0, 0])  # 将视野内的点云设为红色
    # # 获取相机位置
    # camera_position = T_world_to_camera[:3, 3]
    #
    # # 创建相机坐标系
    # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    # coord_frame.transform(T_world_to_camera)
    # 计算FOV边界
    half_fov_horizontal_rad = np.deg2rad(fov_horizontal / 2)
    half_fov_vertical_rad = np.deg2rad(fov_vertical / 2)
    # FOV边界点计算（假设距离为1）
    points_fov = [
        [np.tan(half_fov_horizontal_rad),
         np.tan(half_fov_vertical_rad),
         1],
        [np.tan(half_fov_horizontal_rad),
         -np.tan(half_fov_vertical_rad),
         1],
        [-np.tan(half_fov_horizontal_rad),
         np.tan(half_fov_vertical_rad),
         1],
        [-np.tan(half_fov_horizontal_rad),
         -np.tan(half_fov_vertical_rad),
         1]
    ]
    # 转换到世界坐标系
    # points_fov = np.dot(T_world_to_camera[:3, :3], np.array(points_fov).T).T + camera_position
    # 创建FOV边框线
    lines = [[0, 1], [1, 3], [3, 2], [2, 0]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_fov),
        lines=o3d.utility.Vector2iVector(lines)
    )
    # 设置颜色
    line_set.paint_uniform_color([1, 0, 0])
    # 创建全局坐标系
    global_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    # 调整点的大小
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(pcd_in_fov)
    # vis.add_geometry(coord_frame)
    vis.add_geometry(line_set)
    vis.add_geometry(global_coord_frame)
    render_option = vis.get_render_option()
    render_option.point_size = 0.05  # 设置点的大小
    vis.run()

def visualize_points_on_image(points_in_fov, u, v, img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image from path: {img_path}")
        return
    # 计算Z轴的最大最小值以便归一化
    fov_z_values = np.array(points_in_fov)[:, 2]  # 提取所有点的Z坐标
    # 应用自定义的热图颜色映射
    colors = colormap_jet(fov_z_values)
    # 由于colors是NumPy数组，我们可以直接使用它而无需再次转换
    colors_bgr = (colors * 255).astype(np.uint8)[:, ::-1]  # 一次性转换所有颜色到BGR并取整
    for i in range(len(points_in_fov)):
        # 直接使用转换后的BGR颜色
        cv2.rectangle(img, (int(u[i]), int(v[i])), (int(u[i]) + 1, int(v[i]) + 1), colors_bgr[i].tolist(), -1)
    cv2.imshow('Points on Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_global_point_cloud(global_pcd):
    if len(global_pcd.points) == 0:
        print("No points to visualize.")
        return
    print(f"Number of points: {len(global_pcd.points)}")
    z_values = np.asarray(global_pcd.points)[:, 2]
    colors = colormap_jet(z_values)
    global_pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(global_pcd)
    render_option = vis.get_render_option()
    render_option.point_size = 0.05
    vis.run()
    vis.destroy_window()

def visual_voxel(pcd):
    pc = o3d.geometry.PointCloud()
    # 将NumPy数组设置为新的点云对象的点
    pc.points = o3d.utility.Vector3dVector(pcd)
    # 计算Z轴的最大最小值以便归一化
    z_values = np.array(pc.points)[:, 2]  # 提取所有点的Z坐标
    # 应用自定义的热图颜色映射
    colors = colormap_jet(z_values)
    # 将颜色分配给点云
    pc.colors = o3d.utility.Vector3dVector(colors)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=0.05)
    origin = np.zeros(3)  # 坐标轴的原点
    axis_length = 1.0  # 坐标轴的长度
    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(axis_length, origin)
    o3d.visualization.draw_geometries([voxel_grid, axis_frame])

def read_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    # points = np.asarray(pcd.points)
    return pcd

def read_camera_pose(tum_format_str):
    data = tum_format_str.split()
    translation = np.array([float(data[0]), float(data[1]), float(data[2])])
    quaternion = np.array([float(data[3]), float(data[4]), float(data[5]), float(data[6])])
    rotation = R.from_quat(quaternion).as_matrix()
    T_world_to_camera = np.eye(4)
    T_world_to_camera[:3, :3] = rotation
    T_world_to_camera[:3, 3] = translation
    return T_world_to_camera

def read_and_tran_camera_pose(tum_format_str, tr_matrix):
    data = tum_format_str.split()
    translation = np.array([float(data[0]), float(data[1]), float(data[2])])
    quaternion = np.array([float(data[3]), float(data[4]), float(data[5]), float(data[6])])
    ori_rotation_matrix = R.from_quat(quaternion).as_matrix()
    ori_tr = np.eye(4)
    ori_tr[:3, :3] = ori_rotation_matrix
    ori_tr[:3, 3] = translation
    ori_tr_inv = np.linalg.inv(ori_tr)
    # tr_matrix = tr_matrix_lidar_2_depth
    tr_matrix_inv = np.linalg.inv(tr_matrix)
    T_world_to_camera = ori_tr @ tr_matrix

    tr_rotation = R.from_matrix(T_world_to_camera[:3, :3])
    tr_quaternion = tr_rotation.as_quat()
    # tr_quaternion = rot2quaternion(tr_rotation)
    tr_translate = T_world_to_camera[:3, 3]
    new_pose = [tr_translate, tr_quaternion]
    print(new_pose)
    return T_world_to_camera, ori_tr_inv

def read_and_tran_lidar_pose_2(tum_format_str, tr_matrix_lidar_2_dog, tr_matrix_lidar_2_camera):
    data = tum_format_str.split()
    translation = np.array([float(data[0]), float(data[1]), float(data[2])])
    quaternion = np.array([float(data[3]), float(data[4]), float(data[5]), float(data[6])])
    ori_rotation_matrix = R.from_quat(quaternion).as_matrix()
    ori_tr = np.eye(4)
    ori_tr[:3, :3] = ori_rotation_matrix
    ori_tr[:3, 3] = translation
    # ori_tr_inv = np.linalg.inv(ori_tr)
    # tr_matrix_inv = np.linalg.inv(tr_matrix)
    # T_world_to_camera = ori_tr @ tr_matrix
    # tr_matrix_lidar_2_camera_inv = np.linalg.inv(tr_matrix_lidar_2_camera)
    tr_matrix_lidar_2_camera_inv = np.linalg.inv(tr_matrix_lidar_2_camera)
    T_world_to_camera = ori_tr @ tr_matrix_lidar_2_dog @ tr_matrix_lidar_2_camera

    tr_rotation = R.from_matrix(T_world_to_camera[:3, :3])
    tr_quaternion = tr_rotation.as_quat()
    # tr_quaternion = rot2quaternion(tr_rotation)
    tr_translate = T_world_to_camera[:3, 3]
    new_pose = [tr_translate, tr_quaternion]
    print(new_pose)
    return T_world_to_camera

def transform_points(points, T_world_to_camera):
    points_homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    T_world_to_camera_inv = np.linalg.inv(T_world_to_camera)
    # points_camera = tr_matrix_lidar_2_depth @ T_radar_to_world @ points_homogeneous.T
    points_camera = T_world_to_camera_inv @ points_homogeneous.T
    points_camera = points_camera[:3, :].T
    mask = points_camera[:, 2] >= 0
    filtered_points_camera = points_camera[mask]
    original_indices = np.arange(points.shape[0])[mask]
    mask_far = filtered_points_camera[:, 2] < 9.5
    filtered_points_camera_near = filtered_points_camera[mask_far]
    original_indices_near = original_indices[mask_far]
    return filtered_points_camera_near, original_indices_near


# def transform_points_gpu(points, T_world_to_camera):
#     # 将数据移动到 GPU
#     points_gpu = cp.asarray(points)
#     T_radar_to_world = cp.linalg.inv(cp.asarray(T_world_to_camera))
#
#     # 转换为齐次坐标并进行变换
#     points_homogeneous = cp.column_stack((points_gpu, cp.ones(points_gpu.shape[0])))
#     points_camera = (T_radar_to_world @ points_homogeneous.T).T
#
#     # 提取三维坐标（忽略齐次坐标的第四维）
#     points_camera = points_camera[:, :3]
#
#     # 过滤 z >= 0 的点
#     mask = points_camera[:, 2] >= 0
#     filtered_points_camera = points_camera[mask]
#
#     # 将结果移回 CPU（如果需要）
#     return cp.asnumpy(filtered_points_camera)

def project_points(points_camera, K):
    # 将相机坐标系下的点投影到图像平面上
    # K 是相机的内参矩阵，points_camera.T 是点的转置，因为通常期望点的坐标是 (N, 3) 形状，而内参矩阵与 (3, N) 形状的数组相乘
    points_projected = K @ points_camera.T
    # 将投影后的点从齐次坐标转换为非齐次坐标
    # 通过除以第三维（深度信息）来实现
    points_projected = points_projected / points_projected[2, :]
    # 提取投影点的 x 坐标（图像上的 u 坐标）
    u = points_projected[0, :]
    # 提取投影点的 y 坐标（图像上的 v 坐标）
    v = points_projected[1, :]
    # 提取原始相机坐标系下点的深度信息
    # 注意：这里的深度信息并未经过投影变换，仍然是相机坐标系下的深度
    depth = points_camera[:, 2]
    # 返回投影点的 u, v 坐标和原始深度信息
    return u, v, depth



def filter_points_within_fov(u, v, depth, img_shape, fov_horizontal, fov_vertical, points_camera):
    height, width = img_shape
    half_fov_horizontal = np.deg2rad(fov_horizontal / 2)
    half_fov_vertical = np.deg2rad(fov_vertical / 2)
    # 提取相机坐标系下的 x, y 坐标
    x_camera = points_camera[:, 0]
    y_camera = points_camera[:, 1]
    max_x = np.tan(half_fov_horizontal) * depth
    min_x = -max_x
    max_y = np.tan(half_fov_vertical) * depth
    min_y = -max_y
    mask = (u >= 0) & (u < width) & (v >= 0) & (v < height) & \
           (x_camera >= min_x) & (x_camera <= max_x) & (y_camera >= min_y) & (y_camera <= max_y)
    return mask

# 应用热力图颜色映射
def colormap_jet(z_values):
    # 使用matplotlib的jet colormap
    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=np.min(z_values), vmax=np.max(z_values))
    colors = cmap(norm(z_values))[:, :3]  # 只取RGB值，不取alpha通道
    return colors

def create_depth_map(u, v, depth, img_shape, K, voxel_size):
    depth_map = np.full(img_shape, np.inf, dtype=np.float32)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u_valid = np.round(u).astype(int)
    v_valid = np.round(v).astype(int)

    # 遍历每个有效的 u 和 v 坐标
    for i in range(len(u_valid)):
        z = depth[i]  # 获取当前点的深度值

        # 计算当前点对应的体素在图像平面上的投影宽度和高度的一半
        dx = (voxel_size * fx / z) / 2
        dy = (voxel_size * fy / z) / 2

        # 计算膨胀区域的边界
        left = max(0, int(round(u_valid[i] - dx)))
        right = min(img_shape[1] - 1, int(round(u_valid[i] + dx)))
        top = max(0, int(round(v_valid[i] - dy)))
        bottom = min(img_shape[0] - 1, int(round(v_valid[i] + dy)))

        # 使用 NumPy 向量化操作生成膨胀区域的行和列索引
        # np.meshgrid 用于生成两个二维数组，分别表示膨胀区域的行和列索引
        rows, cols = np.meshgrid(np.arange(top, bottom + 1), np.arange(left, right + 1))

        # 更新膨胀区域内的深度值为当前深度值与原有深度值中的较小值
        # 使用 NumPy 的广播机制同时更新多个像素的深度值
        depth_map[rows, cols] = np.minimum(depth_map[rows, cols], z)

    # 将深度图中仍为无穷大的像素值设为 0，表示这些像素没有有效的深度值
    depth_map[depth_map == np.inf] = 0

    return depth_map


@jit(fastmath=True, parallel=True, nopython=True)
def create_depth_map_gpu(u, v, depth, img_shape, K, voxel_size, point_indices):
    depth_map = np.full(img_shape, np.inf, dtype=np.float32)
    index_map = np.full(img_shape, -1, dtype=np.int32)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u_valid = np.round(u).astype(np.int32)
    v_valid = np.round(v).astype(np.int32)

    # 遍历每个有效的 u 和 v 坐标
    for i in range(len(u_valid)):
        z = depth[i]  # 获取当前点的深度值

        # 计算当前点对应的体素在图像平面上的投影宽度和高度的一半
        dx = (voxel_size * fx / z) / 2
        dy = (voxel_size * fy / z) / 2

        # 计算膨胀区域的边界
        left = max(0, int(round(u_valid[i] - dx)))
        right = min(img_shape[1] - 1, int(round(u_valid[i] + dx)))
        top = max(0, int(round(v_valid[i] - dy)))
        bottom = min(img_shape[0] - 1, int(round(v_valid[i] + dy)))

        # 更新膨胀区域内的深度值为当前深度值与原有深度值中的较小值
        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                if 0 <= row < img_shape[0] and 0 <= col < img_shape[1]:
                    if z < depth_map[row, col]:  # 只有当 z 更小时才更新
                        depth_map[row, col] = z
                        index_map[row, col] = point_indices[i]

    # 将深度图中仍为无穷大的像素值设为 0，表示这些像素没有有效的深度值
    for row in range(img_shape[0]):
        for col in range(img_shape[1]):
            if depth_map[row, col] == np.inf:
                depth_map[row, col] = 0
    for row in range(img_shape[0]):
        for col in range(img_shape[1]):
            if depth_map[row, col] < 1.5:
                depth_map[row, col] = 0
    return depth_map, index_map


def display_depth_map(depth_map):
    depth_map_normalized = (depth_map / np.max(depth_map) * 255).astype(np.uint8)
    depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    cv2.imshow('Depth Map', depth_map_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 从文件中读取雷达位姿
def read_poses(file_path):
    poses = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 7:
                translation = np.array(parts[:3], dtype=float)
                quaternion = np.array(parts[3:], dtype=float)
                pose = [translation, quaternion]
                poses.append(pose)
    return poses

def get_voxels_center(voxel_grid):
    # 提取体素中心坐标
    voxels = voxel_grid.get_voxels()
    voxel_indices = np.array([voxel.grid_index for voxel in voxels])
    grid_max = np.max(voxel_indices, axis=0)
    grid_min = np.min(voxel_indices, axis=0)
    grid_size = np.ceil((grid_max - grid_min) + 1).astype(int)
    voxel_centers = voxel_indices * voxel_grid.voxel_size + voxel_grid.origin
    return np.array(voxel_centers), voxel_indices


def unproject_depth_map(depth_map, K, T_world_to_camera):
    h, w = depth_map.shape

    # 初始化相机内参矩阵
    cam_mat = K
    cam_mat_inv = np.linalg.inv(cam_mat)
    # 生成网格的x和y坐标
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    # 将x坐标调整为表示像素中心的位置
    x = x.reshape((1, -1))[:, :]
    # 将y坐标调整为表示像素中心的位置
    y = y.reshape((1, -1))[:, :]
    # 将深度值重塑为一维数组
    z = depth_map.reshape((1, -1))[:, :]

    # 将x, y坐标和1（表示齐次坐标）堆叠成二维点坐标矩阵p_2d
    p_2d = np.vstack([x, y, np.ones_like(x)])
    # 使用相机内参矩阵的逆矩阵将二维点坐标转换为三维点坐标
    pc = cam_mat_inv @ p_2d
    # 将三维点坐标的z分量与对应的深度值相乘，得到最终的三维点云坐标
    pc = pc * z
    pc_homo = np.vstack([pc, np.ones((1, pc.shape[1]))])

    # T_camera_to_world = np.linalg.inv(T_world_to_camera)

    pc_global_homo = T_world_to_camera @ pc_homo

    return pc_global_homo[:3, :].T, pc.T

def get_depth_point(T_world_to_camera, index_map, points, K):
    # 提取所有非负的有效索引
    valid_indices = np.unique(index_map[index_map >= 0])
    # 使用有效索引来过滤点云
    filtered_pc = points[valid_indices]
    points_homogeneous = np.column_stack((filtered_pc, np.ones(filtered_pc.shape[0])))
    T_world_to_camera_inv = np.linalg.inv(T_world_to_camera)
    filtered_pc_camera = T_world_to_camera_inv @ points_homogeneous.T
    filtered_pc_camera = filtered_pc_camera[:3, :].T

    points_projected = K @ filtered_pc_camera.T
    # 通过除以第三维（深度信息）来实现
    points_projected = points_projected / points_projected[2, :]
    # 提取投影点的 x 坐标（图像上的 u 坐标）
    u = points_projected[0, :]
    # 提取投影点的 y 坐标（图像上的 v 坐标）
    v = points_projected[1, :]
    # 提取原始相机坐标系下点的深度信息
    # 注意：这里的深度信息并未经过投影变换，仍然是相机坐标系下的深度
    depth = filtered_pc_camera[:, 2]

    return filtered_pc_camera, u, v, depth

def main():
    # 相机内参
    K = np.array([
    [604.5459594726562, 0, 432.69287109375],
    [0, 604.0941772460938, 254.2894287109375],
    [0, 0, 1]
    ])
    # 筛选出在相机FOV内的点云
    img_shape = (480, 848)  # 图像尺寸
    fov_horizontal = 70.08  # 水平FOV角度
    fov_vertical = 43.31  # 垂直FOV角度
    voxel_size = 0.05
    # 文件路径
    pcd_path = file_pre_path + file_path + 'scans.pcd'
    # pcd_path = file_pre_path + file_path + '_livox_lidar/1737357392_600268602.pcd'
    # tum_path = file_pre_path + file_path + 'poses.txt'
    img_path = file_pre_path + file_path + '_camera_color_image_raw/1743498555_062030792.png'
    # pcd_path = '/media/benny/GML_FLOOR7/2025-1-21/pcd/1.pcd'
    # img_path = '/media/benny/GML_FLOOR7/2025-1-21/IMG/1.jpg'
    # store_path = file_pre_path + file_path + 'depth/'
    tum_format_str = "5.965400696 -1.297876477 2.146819592 0.248787075 0.022817513 -0.721599042 0.645661831"
    # poses = read_poses(tum_path)
    # 读取点云数据
    points = read_pcd(pcd_path)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(points, voxel_size=voxel_size)
    points, voxel_indices = get_voxels_center(voxel_grid)

    # no tran
    # T_world_to_camera = read_camera_pose(tum_format_str)


    # with tran lidar to camera
    T_world_to_camera, ori_tr_inv = read_and_tran_camera_pose(tum_format_str, tr_matrix_lidar_2_depth)
    # T_world_to_camera = read_and_tran_lidar_pose_2(tum_format_str, tr_matrix_lidar_2_dog, tr_matrix_lidar_2_depth)


    points_camera, original_indices = transform_points(points, T_world_to_camera)
    visual_voxel(points_camera)

    u, v, depth = project_points(points_camera, K)
    mask = filter_points_within_fov(u, v, depth, img_shape, fov_horizontal, fov_vertical, points_camera)
    points_in_fov = points_camera[mask]
    original_indices_in_fov = original_indices[mask]
    # visual_voxel(points_in_fov)

    u_f = u[mask]
    v_f = v[mask]
    depth_map, index_map = create_depth_map_gpu(u_f, v_f, points_in_fov[:, 2], img_shape, K, voxel_size, original_indices_in_fov)
    # display_depth_map(depth_map)
    print(f"points_in_fov num：{len(points_in_fov)} ")

    filtered_pc_camera, u_ft, v_ft, depth = get_depth_point(T_world_to_camera, index_map, points, K)
    # visual_voxel(points_in_fov)
    visual_voxel(filtered_pc_camera)
    # 可视化结果
    visualize_points_on_image(filtered_pc_camera, u_ft, v_ft, img_path)





# def base_index_map_create_depth_map(T_world_to_camera, depth_map, img_shape, index_map, points):
#     # 根据index_map从原始点云中提取点，并保留对应的索引
#     extracted_points = []
#     extracted_indices = []
#     for row in range(img_shape[0]):
#         for col in range(img_shape[1]):
#             idx = index_map[row, col]
#             if idx != -1:
#                 extracted_points.append(points[idx])
#                 extracted_indices.append(idx)
#     extracted_points = np.array(extracted_points)
#     extracted_indices = np.array(extracted_indices)
#     # 将提取的点转换回相机坐标系
#     points_camera_from_index_map, _ = transform_points(extracted_points, T_world_to_camera)
#     # 提取转换后的点的深度信息
#     depths_from_index_map = points_camera_from_index_map[:, 2]
#     # 创建一个与img_shape相同的零数组来存储depths_from_index_map
#     depths_from_index_map_full = np.zeros(img_shape, dtype=np.float32)
#     # 使用extracted_indices来填充depths_from_index_map_full
#     for i, idx in enumerate(extracted_indices):
#         row, col = np.where(index_map == idx)
#         if len(row) > 0 and len(col) > 0:
#             depths_from_index_map_full[row[0], col[0]] = depths_from_index_map[i]
#
#     def normalize_depth_map(depth_map):
#         depth_map_normalized = (depth_map / np.max(depth_map) * 255).astype(np.uint8)
#         return cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
#
#     depth_map_colored = normalize_depth_map(depth_map)
#     depths_from_index_map_colored = normalize_depth_map(depths_from_index_map_full)
#     # 水平拼接两张图片
#     combined_image = np.hstack((depth_map_colored, depths_from_index_map_colored))
#     cv2.imshow('Depth Maps', combined_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


if __name__ == '__main__':
    main()