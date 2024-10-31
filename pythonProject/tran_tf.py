# 从poses_path中读取数据，
# txt文件中每行包含一个Tum格式的雷达世界位姿，如： 0.0564132 -0.1065 -0.0905786 0.0102955 0.0238291 -0.231191 0.972562
# 用tr_matrix转换成摄像头的世界位姿，
# 然后将这些位姿用Tum格式写入store_path中。

import numpy as np
from scipy.spatial.transform import Rotation as R

# 转换矩阵，确保是numpy数组
tr_matrix_lidar_2_depth = np.array([
    [-0.0347702, -0.999329, -0.0115237, 0.113479],
    [0.0359214, 0.0102735, -0.999302, -0.216314],
    [0.99875, -0.0351599, 0.0355401, -0.00212184],
    [0, 0, 0, 1]
])
tr_imu_2_lidar =np.array([11,23.29,-44.12])
# 四元数转旋转矩阵
def quaternion2rot(quaternion):
    r = R.from_quat(quaternion) # 顺序为 (x, y, z, w)
    rot = r.as_matrix()
    return rot

# 旋转矩阵转换为四元数
def rot2quaternion(rotation_matrix):
    r3 = R.from_matrix(rotation_matrix)
    qua = r3.as_quat()
    return qua.tolist()

def read_poses(file_path):
    """从文件中读取雷达位姿"""
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


def transform_poses(poses, rotation_matrix, translation_matrix):
    """使用转换矩阵转换位姿"""
    transformed_poses = []
    for pose in poses:
        translation, quaternion = pose
        # 将四元数转换为旋转矩阵
        ori_rotation_matrix = quaternion2rot(quaternion)
        tr_rotation = np.round(np.dot(ori_rotation_matrix, rotation_matrix ), 8)
        tr_quaternion = rot2quaternion(tr_rotation)
        tr_translate =np.round([translation[0] + translation_matrix[0], translation[1] + translation_matrix[1], translation[2] + translation_matrix[2]],6)
        new_pose = [tr_translate, tr_quaternion]
        transformed_poses.append(new_pose)
    return transformed_poses


def write_poses(poses, store_path):
    """将位姿写入文件"""
    with open(store_path, 'w') as file:
        for pose in poses:
            translation, quaternion = pose
            combined = list(translation.flatten()) + quaternion
            # 将旋转矩阵（现在是平展的）和平移向量写回一行
            line = ' '.join(map(str, combined))
            file.write(line + '\n')


if __name__ == '__main__':
    folder_name = 'gml_2024-10-15-10-45-17/'
    file_name = 'pose_mid_360.txt'
    base_path = '/media/benny/bennyMove/data/dog_origin/'
    poses_path = base_path + folder_name + file_name
    store_path = base_path + folder_name + 'poses.txt'

    # 读取雷达位姿
    poses = read_poses(poses_path)

    # 转换位姿 mid_360_2_depth
    transformed_poses = transform_poses(poses, tr_matrix_lidar_2_depth[:3, :3], tr_matrix_lidar_2_depth[:3, 3])

    # depth_2_correct_dir
    rot_change = R.from_euler('XYZ', [-90, 90, 0], degrees=True).as_matrix()
    transformed_poses = transform_poses(transformed_poses, rot_change, np.zeros((3, 1)))
    # rot_change = R.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()
    # transformed_poses = transform_poses(transformed_poses, rot_change, np.zeros((3, 1)))

    # rot_change = R.from_euler('xyz', [-90, 0, -90], degrees=True).as_matrix()
    # transformed_poses = transform_poses(transformed_poses, rot_change, np.zeros((3, 1)))

    # 写入文件
    write_poses(transformed_poses, store_path)
    print("位姿转换完成并已写入文件。")
