import os
from PIL import Image
import numpy as np
import cv2
from gmlDogRecordFilePath import file_path, file_pre_path

# 从file_path 中读取全部.png 深度图
# 并去除所有大于3000的深度值，然后把单位从毫米转换为米
# 转化成格式为float32 的numpy 二维数组
# 并保存到store_path 中

def calculate_perspective_transform_matrix(K_depth, K_rgb):
    # Compute the perspective transform matrix
    H = K_rgb @ np.linalg.inv(K_depth)
    return H

def process_depth_images(file_path, store_path, K_depth, K_rgb, image_shape):
    # 确保输出目录存在
    os.makedirs(store_path, exist_ok=True)

    # 计算透视变换矩阵
    H = calculate_perspective_transform_matrix(K_depth, K_rgb)

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
            depth_array[depth_array > 8000] = 0  # 设置为0或其他处理方式

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

    # 内参矩阵
    K_depth = np.array([
        [425.571807861328, 0, 425.977661132812],
        [0, 425.571807861328, 241.062408447266],
        [0, 0, 1]
    ])

    K_rgb = np.array([
        [604.545959472656, 0, 432.69287109375],
        [0, 604.094177246094, 254.289428710938],
        [0, 0, 1]
    ])

    # 图像形状
    image_shape = (484, 848)

    process_depth_images(file_path, store_path, K_depth, K_rgb, image_shape)



