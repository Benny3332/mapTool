import os
from PIL import Image
import numpy as np
import cv2
from gmlDogRecordFilePath import file_path, file_pre_path

# 从file_path 中读取全部.png 深度图
# 并去除所有大于3000的深度值，然后把单位从毫米转换为米
# 转化成格式为float32 的numpy 二维数组
# 并保存到store_path 中

def calculate_perspective_transform_matrix(depth_fov_x, depth_fov_y, rgb_fov_x, rgb_fov_y, image_shape):
    height, width = image_shape

    # Calculate focal lengths for both cameras
    fx_depth = width / (2 * np.tan(np.deg2rad(depth_fov_x) / 2))
    fy_depth = height / (2 * np.tan(np.deg2rad(depth_fov_y) / 2))

    fx_rgb = width / (2 * np.tan(np.deg2rad(rgb_fov_x) / 2))
    fy_rgb = height / (2 * np.tan(np.deg2rad(rgb_fov_y) / 2))

    # Define intrinsic matrices for both cameras
    K_depth = np.array([[fx_depth, 0, width / 2],
                        [0, fy_depth, height / 2],
                        [0, 0, 1]])

    K_rgb = np.array([[fx_rgb, 0, width / 2],
                      [0, fy_rgb, height / 2],
                      [0, 0, 1]])

    # Assume no rotation between the two cameras, only a change in focal length
    R = np.eye(3)

    # Compute the perspective transform matrix
    P_depth = K_depth @ R
    P_rgb = K_rgb @ R

    H = P_rgb @ np.linalg.inv(P_depth)

    return H

def process_depth_images(file_path, store_path, depth_fov_x, depth_fov_y, rgb_fov_x, rgb_fov_y, image_shape):
    # 确保输出目录存在
    os.makedirs(store_path, exist_ok=True)

    # 计算透视变换矩阵
    H = calculate_perspective_transform_matrix(depth_fov_x, depth_fov_y, rgb_fov_x, rgb_fov_y, image_shape)

    # 遍历file_path中的所有.png文件
    for filename in os.listdir(file_path):
        if filename.endswith(".png"):
            # 完整路径
            full_path = os.path.join(file_path, filename)

            # 使用Pillow读取图像
            img = Image.open(full_path)

            # 将图像数据转换为numpy数组
            depth_array = np.array(img, dtype=np.float32)

            # 去除所有大于3000的深度值
            depth_array[depth_array > 3000] = 0  # 设置为0或其他处理方式

            # 将单位从毫米转换为米
            depth_array /= 1000.0

            # 应用透视变换
            transformed_depth_array = cv2.warpPerspective(depth_array, H, (image_shape[1], image_shape[0]), flags=cv2.INTER_NEAREST)

            # 构造输出文件名（这里简单使用原文件名，但你可以根据需要修改）
            output_filename = os.path.join(store_path, filename.replace('.png', '.npy'))

            # 使用numpy的save函数保存数组
            np.save(output_filename, transformed_depth_array)


if __name__ == "__main__":
    folder_name = file_path
    base_path = file_pre_path
    print("file full path：" + base_path + folder_name)
    file_path = base_path + folder_name + 'depth_png/'
    store_path = base_path + folder_name + 'depth_aligned/'

    # FOV values and image shape
    depth_fov_x = 89.79
    depth_fov_y = 58.84
    rgb_fov_x = 70.08
    rgb_fov_y = 43.31
    image_shape = (848, 480)

    process_depth_images(file_path, store_path, depth_fov_x, depth_fov_y, rgb_fov_x, rgb_fov_y, image_shape)



