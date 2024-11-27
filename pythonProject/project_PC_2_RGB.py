import numpy as np
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation as R


def read_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points


def read_camera_pose(tum_format_str):
    data = tum_format_str.split()
    translation = np.array([float(data[0]), float(data[1]), float(data[2])])
    quaternion = np.array([float(data[3]), float(data[4]), float(data[5]), float(data[6])])
    rotation = R.from_quat(quaternion).as_matrix()
    T_world_to_camera = np.eye(4)
    T_world_to_camera[:3, :3] = rotation
    T_world_to_camera[:3, 3] = translation
    return T_world_to_camera


def transform_points(points, T_radar_to_camera, T_world_to_camera):
    T_radar_to_world = np.linalg.inv(T_world_to_camera)
    T_radar_to_final = T_world_to_camera @ T_radar_to_world @ T_radar_to_camera
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_camera = T_radar_to_final @ points_homogeneous.T
    points_camera = points_camera[:3, :].T
    return points_camera


def project_points(points_camera, K):
    points_projected = K @ points_camera.T
    points_projected = points_projected / points_projected[2, :]
    u = points_projected[0, :]
    v = points_projected[1, :]
    depth = points_camera[:, 2]
    return u, v, depth


def filter_points_within_fov(u, v, depth, img_shape, fov_horizontal, fov_vertical):
    height, width = img_shape
    half_fov_horizontal = np.deg2rad(fov_horizontal / 2)
    half_fov_vertical = np.deg2rad(fov_vertical / 2)

    max_x = np.tan(half_fov_horizontal) * depth
    min_x = -max_x
    max_y = np.tan(half_fov_vertical) * depth
    min_y = -max_y

    mask = (u >= 0) & (u < width) & (v >= 0) & (v < height) & \
           (u >= min_x) & (u <= max_x) & (v >= min_y) & (v <= max_y)

    return mask

# 可视化点云
def visualize_point_cloud(points, points_in_fov):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    pcd_in_fov = o3d.geometry.PointCloud()
    pcd_in_fov.points = o3d.utility.Vector3dVector(points_in_fov)
    pcd_in_fov.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([pcd, pcd_in_fov])

# 在图像上绘制点和FOV范围框
def draw_points_and_fov_on_image(u, v, mask, img_path, img_shape, fov_horizontal, fov_vertical):
    img = cv2.imread(img_path)
    for i in range(len(u)):
        if mask[i]:
            cv2.circle(img, (int(u[i]), int(v[i])), 2, (0, 0, 255), -1)

    # 计算FOV边界
    half_fov_horizontal = np.deg2rad(fov_horizontal / 2)
    half_fov_vertical = np.deg2rad(fov_vertical / 2)
    tan_half_fov_horizontal = np.tan(half_fov_horizontal)
    tan_half_fov_vertical = np.tan(half_fov_vertical)

    # 假设深度为1（单位距离）
    depth = 1.0
    max_x = int(tan_half_fov_horizontal * depth * img_shape[1] / 2)
    min_x = img_shape[1] // 2 - max_x
    max_y = int(tan_half_fov_vertical * depth * img_shape[0] / 2)
    min_y = img_shape[0] // 2 - max_y

    # 绘制FOV范围框
    cv2.rectangle(img, (min_x, min_y), (img_shape[1] - min_x, img_shape[0] - min_y), (0, 255, 0), 2)

    # 显示图像
    cv2.imshow('Points and FOV on Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_points_on_image(points_in_fov, u, v, img_path, mask):
    img = cv2.imread(img_path)
    for i in range(len(points_in_fov)):
        if mask[i]:
            cv2.circle(img, (int(u[i]), int(v[i])), 2, (0, 0, 255), -1)
    cv2.imshow('Points on Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # 文件路径
    pcd_path = '/home/benny/docker/noetic_container_data/bag/extra/gml_2024-11-12-17-25-51/scans.pcd'
    tum_format_str = "0.0364 0.068658 -0.169297 -0.15050933064004318 -0.6836503969983218 0.6943026003327405 0.16707176675165072"
    img_path = '/home/benny/docker/noetic_container_data/bag/extra/gml_2024-11-12-17-25-51/_camera_color_image_raw/1731403708_046922684.png'

    # 读取点云数据
    points = read_pcd(pcd_path)

    # 读取相机位姿
    T_world_to_camera = read_camera_pose(tum_format_str)

    # 雷达到相机的外参
    T_radar_to_camera = np.array([
    [-0.0252995, -0.999668, 0.00479095, 0.025508],
    [-0.0283215, -0.00407382, -0.999591, -0.106692],
    [0.999279, -0.0254248, -0.028209, -0.224522],
    [0, 0, 0, 1]
    ])

    # 将点云从雷达坐标系转换到相机坐标系
    points_camera = transform_points(points, T_radar_to_camera, T_world_to_camera)

    # 相机内参
    K = np.array([
    [606.160278320312, 0, 433.248992919922],
    [0, 606.088684082031, 254.570159912109],
    [0, 0, 1]
    ])

    # 投影点云到图像平面上
    u, v, depth = project_points(points, K)

    # 筛选出在相机FOV内的点云
    img_shape = (480, 848)  # 图像尺寸
    fov_horizontal = 69.94  # 水平FOV角度
    fov_vertical = 43.18  # 垂直FOV角度

    mask = filter_points_within_fov(u, v, depth, img_shape, fov_horizontal, fov_vertical)
    points_in_fov = points[mask]

    # 可视化点云
    visualize_point_cloud(points, points_in_fov)
    # 可视化结果
    # visualize_points_on_image(points_in_fov, u, v, img_path, mask)

if __name__ == '__main__':
    main()