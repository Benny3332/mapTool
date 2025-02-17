# 从poses_path中读取数据，
# txt文件中每行包含一个Tum格式的雷达世界位姿，如： 0.0564132 -0.1065 -0.0905786 0.0102955 0.0238291 -0.231191 0.972562
# 用tr_matrix转换成摄像头的世界位姿，
# 然后将这些位姿用Tum格式写入store_path中。

import numpy as np
from scipy.spatial.transform import Rotation as R
from gmlDogRecordFilePath import file_path
from gmlDogRecordFilePath import file_pre_path

# 转换矩阵
#ld revise
tr_matrix_lidar_2_depth = np.array([
    [0.0270124,  0.2756647,  0.9608743, 0.0297088],
    [-0.9990314, -0.0259554,  0.0355314, -0.105938],
    [0.0347347, -0.9609034,  0.2746966, -0.12],
    [0, 0, 0, 1]])

# mm
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


def transform_poses(poses, tr_matrix):
    """使用转换矩阵转换位姿"""
    transformed_poses = []
    for pose in poses:
        translation, quaternion = pose
        # 将四元数转换为旋转矩阵
        ori_rotation_matrix = R.from_quat(quaternion).as_matrix()
        ori_tr = np.eye(4)
        ori_tr[:3, :3] = ori_rotation_matrix
        ori_tr[:3, 3] = translation
        ori_tr_inv = np.linalg.inv(ori_tr)
        tr_matrix_inv = np.linalg.inv(tr_matrix)

        tr_camera =  ori_tr @ tr_matrix
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
    store_path = base_path + folder_name + 'poses_camera.txt'
    print("file full path：" + base_path + folder_name)
    # 读取雷达位姿
    poses = read_poses(poses_path)

    transformed_poses = transform_poses(poses, tr_matrix_lidar_2_depth)


    # 写入文件
    write_poses(transformed_poses, store_path)
    print("位姿转换完成并已写入文件。")
