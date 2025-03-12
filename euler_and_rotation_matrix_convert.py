import numpy as np

rotation_matrix = np.array([
    [-0.00394076, -0.999482, 0.0319351],
    [0.146906, -0.0321675, -0.988627],
    [0.989143, 0.000795496, 0.146956]
])

def euler_to_rotation_matrix(yaw, pitch, roll):
    """
    将欧拉角（yaw, pitch, roll）转换为旋转矩阵。
    欧拉角顺序：Z-Y-X (Tait-Bryan angles)
    :param yaw: 绕z轴旋转的角度（弧度）
    :param pitch: 绕y轴旋转的角度（弧度）
    :param roll: 绕x轴旋转的角度（弧度）
    :return: 3x3旋转矩阵
    """
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(roll),       -np.sin(roll)     ],
                    [0,         np.sin(roll),        np.cos(roll)      ]])

    R_y = np.array([[np.cos(pitch),   0,      np.sin(pitch)   ],
                    [0,               1,      0               ],
                    [-np.sin(pitch),  0,      np.cos(pitch)   ]])

    R_z = np.array([[np.cos(yaw),    -np.sin(yaw),    0],
                    [np.sin(yaw),     np.cos(yaw),    0],
                    [0,               0,              1]])

    # ZYX顺序
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def rotation_matrix_to_euler(R):
    """
    将旋转矩阵转换为欧拉角（yaw, pitch, roll）。
    欧拉角顺序：Z-Y-X (Tait-Bryan angles)
    :param R: 3x3旋转矩阵
    :return: 欧拉角（yaw, pitch, roll），单位为弧度
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([z, y, x])

# 示例用法
if __name__ == "__main__":
    # 转换为欧拉角
    euler_angles = rotation_matrix_to_euler(rotation_matrix)
    print("Euler Angles (radians):")
    print(euler_angles)

    # 如果需要角度制的结果
    euler_angles_degrees = np.degrees(euler_angles)
    print("\nEuler Angles (degrees):")
    print(euler_angles_degrees)

    # 增加俯仰角（pitch）10度（转换为弧度）
    pitch_increase_radians_y = np.radians(-30)
    pitch_increase_radians_x = np.radians(10)

    new_pitch_1 = euler_angles[1] + pitch_increase_radians_y
    new_pitch_2 = euler_angles[2] + pitch_increase_radians_x
    new_euler_angles = np.array([euler_angles[0], new_pitch_1, new_pitch_2])
    print("\nNew Euler Angles (radians) after increasing pitch by 10 degrees:")
    print(new_euler_angles)

    # 再次将新的欧拉角转换为旋转矩阵以验证
    converted_rotation_matrix = euler_to_rotation_matrix(*new_euler_angles)
    print("\nConverted Rotation Matrix with increased pitch:")
    print(converted_rotation_matrix)

    # 转换为欧拉角
    euler_angles = rotation_matrix_to_euler(converted_rotation_matrix)
    print("Euler Angles (radians):")
    print(euler_angles)

    # 如果需要角度制的结果
    euler_angles_degrees = np.degrees(euler_angles)
    print("\nEuler Angles (degrees):")
    print(euler_angles_degrees)


