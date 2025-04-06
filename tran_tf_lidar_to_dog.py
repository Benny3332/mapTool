# 从poses_path中读取数据，
# txt文件中每行包含一个Tum格式的雷达世界位姿，如： 0.0564132 -0.1065 -0.0905786 0.0102955 0.0238291 -0.231191 0.972562
# 用tr_matrix转换成摄像头的世界位姿，
# 然后将这些位姿用Tum格式写入store_path中。

import numpy as np
from scipy.spatial.transform import Rotation as R
from gmlDogRecordFilePath import file_path
from gmlDogRecordFilePath import file_pre_path
from add_timestamp_to_poses import prepend_timestamps
# 转换矩阵
# new application function
tr_matrix_lidar_2_dog = np.array([
    [0.93912, -0.00685, 0.34352, 0.00000],
    [0.00000, 0.99980, 0.01994, 0.00000],
    [-0.34359, -0.01873, 0.93893, 0.00000],
    [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])

tr_matrix_lidar_2_camera = np.array([
    [-1.00799236e-02,  3.41871575e-01,  9.39692621e-01, 1.545999999999999978e-02],
    [-9.99949194e-01, -3.37640348e-03, -9.49790910e-03, 5.539000000000000173e-02],
    [-7.42837030e-05, -9.39740617e-01,  3.41888239e-01, 1.501099999999999934e-01],
    [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])

# cjs revise depth_floor5_small
# tr_matrix_lidar_2_depth = np.array([
#     [0.0236432,  0.2727882,  0.9617836, 0.0297088],
#     [-0.9994875, -0.0143159,  0.0286304, -0.105938],
#     [0.0215788, -0.9619676,  0.2723099, -0.326056],
#     [0, 0, 0, 1]])

# cjs revise rgb_floor5_small
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

# mm
tr_imu_2_lidar = np.array([11,23.29,-44.12])
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


def transform_poses(poses, tr_matrix_lidar_2_dog, tr_matrix_lidar_2_camera):
    """使用转换矩阵转换位姿"""
    transformed_poses = []
    for pose in poses:
        translation, quaternion = pose
        # 将四元数转换为旋转矩阵
        ori_rotation_matrix = R.from_quat(quaternion).as_matrix()
        ori_tr = np.eye(4)
        ori_tr[:3, :3] = ori_rotation_matrix
        ori_tr[:3, 3] = translation
        # ori_tr_inv = np.linalg.inv(ori_tr)
        # tr_matrix_inv = np.linalg.inv(tr_matrix)

        # tr_camera = tr_matrix @ ori_tr @ tr_matrix_inv
        tr_camera = tr_matrix_lidar_2_dog @ ori_tr @ tr_matrix_lidar_2_camera
        tr_rotation = R.from_matrix(tr_camera[:3, :3])
        tr_quaternion = tr_rotation.as_quat()
        tr_translate = tr_camera[:3, 3]
        new_pose = [tr_translate, tr_quaternion]
        transformed_poses.append(new_pose)
    return transformed_poses


def write_poses(poses, store_path):
    """将位姿写入文件"""
    with open(store_path, 'w') as file:
        for pose in poses:
            translation, quaternion = pose
            combined = list(translation.flatten()) + list(quaternion.flatten())
            # 将旋转矩阵（现在是平展的）和平移向量写回一行
            line = ' '.join(map(str, combined))
            file.write(line + '\n')


if __name__ == '__main__':
    folder_name = file_path
    file_name = 'pose_mid_360.txt'
    base_path = file_pre_path
    poses_path = base_path + folder_name + file_name
    store_path = base_path + folder_name + 'poses_go2.txt'
    stamp_path = base_path + folder_name + 'pose_mid_360_tamp.txt'
    print("file full path：" + base_path + folder_name)
    # 读取雷达位姿
    poses = read_poses(poses_path)

    transformed_poses = transform_poses(poses, tr_matrix_lidar_2_dog, tr_matrix_lidar_2_camera)
    # 写入文件
    write_poses(transformed_poses, store_path)

    # 添加时间戳
    tamp_store_path  = base_path + folder_name + 'poses_go2_tamp.txt'
    prepend_timestamps(store_path, stamp_path, tamp_store_path)

    print("位姿转换完成并已写入文件。")
