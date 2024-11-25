import numpy as np

def transform_points(points, T):
    """
    将点云从世界坐标系转换到相机坐标系。
    :param points: N x 3 的点云数据
    :param T: 4 x 4 的外参矩阵
    :return: N x 3 的转换后的点云数据
    """
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_camera = T @ points_homogeneous.T
    points_camera = points_camera[:3, :].T
    return points_camera

def project_points(points_camera, K):
    """
    将点云投影到图像平面上。
    :param points_camera: N x 3 的相机坐标系下的点云数据
    :param K: 3 x 3 的相机内参矩阵
    :return: N x 2 的图像坐标和 N x 1 的深度值
    """
    points_projected = K @ points_camera.T
    points_projected = points_projected / points_projected[2, :]
    u = points_projected[0, :]
    v = points_projected[1, :]
    depth = points_camera[:, 2]
    return u, v, depth

def filter_points_within_fov(u, v, depth, img_shape, fov_horizontal, fov_vertical):
    """
    筛选出在相机FOV内的点云。
    :param u: N x 1 的图像x坐标
    :param v: N x 1 的图像y坐标
    :param depth: N x 1 的深度值
    :param img_shape: (height, width) 图像尺寸
    :param fov_horizontal: 水平FOV角度（度）
    :param fov_vertical: 垂直FOV角度（度）
    :return: 在FOV内的点云索引
    """
    height, width = img_shape
    half_fov_horizontal = np.deg2rad(fov_horizontal / 2)
    half_fov_vertical = np.deg2rad(fov_vertical / 2)
    
    # 计算FOV边界
    max_x = np.tan(half_fov_horizontal) * depth
    min_x = -max_x
    max_y = np.tan(half_fov_vertical) * depth
    min_y = -max_y
    
    # 筛选条件
    mask = (u >= 0) & (u < width) & (v >= 0) & (v < height) & \
           (u >= min_x) & (u <= max_x) & (v >= min_y) & (v <= max_y)
    
    return mask

# 示例参数
points = np.random.rand(1000, 3)  # 生成随机点云数据
K = np.array([
    [606.160278320312, 0, 433.248992919922],
    [0, 606.088684082031, 254.570159912109],
    [0, 0, 1]
])  # 相机内参矩阵
T = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])  # 外参矩阵
img_shape = (480, 848)  # 图像尺寸
fov_horizontal = 69.94  # 水平FOV角度
fov_vertical = 43.18  # 垂直FOV角度

# 转换点云到相机坐标系
points_camera = transform_points(points, T)

# 投影点云到图像平面
u, v, depth = project_points(points_camera, K)

# 筛选出在FOV内的点云
mask = filter_points_within_fov(u, v, depth, img_shape, fov_horizontal, fov_vertical)

# 获取在FOV内的点云
points_in_fov = points[mask]

print(points_in_fov)
