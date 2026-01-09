import numpy as np
from scipy.spatial.transform import Rotation as R
from gmlDogRecordFilePath import file_path
from gmlDogRecordFilePath import file_pre_path
from add_timestamp_to_poses import prepend_timestamps

# 转换矩阵
# new application function
tr_matrix_lidar_2_depth = np.array(
    [
        [-0.0260711, -0.999474, -0.0193065, 0.105058],
        [0.0253212, 0.0186466, -0.999505, 0.0714462],
        [0.999339, -0.026547, 0.0248217, -0.0824605],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# --- 修正后的相机内部变换 (OpenCV -> Your Habitat Definition) ---
# OpenCV: (x_r, y_d, z_f) -> Habitat: (x_r, y_d, z_b)
# 只需要反转 Z 轴
T_C_correction = np.array([
    [1,  0,  0, 0],
    [0,  1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
])

# --- 世界系变换 (Lidar World -> Habitat World) ---
# 根据你之前的逻辑：Habitat_X = -LiDAR_Y, Habitat_Y = LiDAR_Z, Habitat_Z = -LiDAR_X
# 请再次确认 Habitat_Y 是否依然是 LiDAR_Z (向上)？ 
# 如果 Habitat 相机是 Y 朝下，而世界系 Y 是朝上的，这通常符合惯例。
T_Wl_Wh = np.array([
    [ 0, -1,  0, 0], 
    [ 0,  0,  1, 0], 
    [-1,  0,  0, 0], 
    [ 0,  0,  0, 1]
])

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


def transform_poses(poses, T_lc_opencv):
    transformed_poses = []
    
    # --- 关键：定义从 OpenCV 相机系到你要求的 Habitat 相机系 (X右, Y下, Z后) ---
    # OpenCV 是 (X右, Y下, Z前)。要变 Z后，只需绕 X 旋转 180 度或反转 Z。
    T_cv_2_hcam = np.array([
        [1,  0,  0, 0],
        [0,  1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])

    # --- 关键：定义世界系转换 T_Wl_Wh ---
    # 我们希望当 LiDAR 在世界系是 Identity 时，
    # 转换后的相机位姿在 Habitat 世界系也是 Identity。
    # 数学推导：T_habitat = T_Wl_Wh @ T_wl @ inv(T_lc_opencv) @ T_cv_2_hcam
    # 若令 T_wl = I, T_habitat = I，则 T_Wl_Wh = inv(inv(T_lc_opencv) @ T_cv_2_hcam)
    # 也就是 T_Wl_Wh = inv(T_cv_2_hcam) @ T_lc_opencv
    
    T_Wl_Wh = np.linalg.inv(T_cv_2_hcam) @ T_lc_opencv

    for pose in poses:
        trans, quat = pose
        T_wl = np.eye(4)
        T_wl[:3, :3] = R.from_quat(quat).as_matrix()
        T_wl[:3, 3] = trans
        
        # 1. 计算相机在原世界系下的位姿 (OpenCV 定义)
        T_wc_cv = T_wl @ np.linalg.inv(T_lc_opencv)
        
        # 2. 将整个世界系搬移到 Habitat 空间，并修正相机局部轴向
        T_habitat = T_Wl_Wh @ T_wc_cv @ T_cv_2_hcam
        
        # 提取结果
        res_R = T_habitat[:3, :3]
        # 针对 scipy 的 from_matrix，如果是左手系（det=-1）需要处理
        # 但通过上面的推导，T_habitat 理论上保持了右手系性质
        rot = R.from_matrix(res_R)
        new_quat = rot.as_quat() # (x, y, z, w)
        new_trans = T_habitat[:3, 3]
        
        transformed_poses.append([new_trans, new_quat])
        
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
    file_name = 'poses_lidar.txt'
    base_path = file_pre_path
    poses_path = base_path + folder_name + file_name
    store_path = base_path + folder_name + 'poses_camera_h3.txt'
    # stamp_path = base_path + folder_name + 'pose_mid_360_tamp.txt'
    print("file full path：" + base_path + folder_name)
    
    # 读取雷达位姿
    poses = read_poses(poses_path)

    # 传入 tr_correction 进行坐标系转换
    transformed_poses = transform_poses(poses, tr_matrix_lidar_2_depth)
    
    # 写入文件
    write_poses(transformed_poses, store_path)

    # 添加时间戳
    # tamp_store_path  = base_path + folder_name + 'poses_camera_tamp.txt'
    # prepend_timestamps(store_path, stamp_path, tamp_store_path)

    print("位姿转换完成并已写入文件。")