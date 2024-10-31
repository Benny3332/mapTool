import os
from PIL import Image
import numpy as np


# 从file_path 中读取全部.png 深度图
# 并去除所有大于3000的深度值，然后把单位从毫米转换为米
# 转化成格式为float32 的numpy 二维数组
# 并保存到store_path 中

def process_depth_images(file_path, store_path):
    # 确保输出目录存在
    os.makedirs(store_path, exist_ok=True)

    # 遍历file_path中的所有.png文件
    for filename in os.listdir(file_path):
        if filename.endswith(".png"):
            # 完整路径
            full_path = os.path.join(file_path, filename)

            # 使用Pillow读取图像
            img = Image.open(full_path)

            # 将图像转换为灰度图（如果已经是灰度图，这一步可以省略）
            # img = img.convert('L')

            # 将图像数据转换为numpy数组
            depth_array = np.array(img, dtype=np.float32)

            # 去除所有大于3000的深度值
            depth_array[depth_array > 3000] = 0  # 或者可以选择其他处理方式，如设置为NaN

            # 将单位从毫米转换为米
            depth_array /= 1000.0

            # 构造输出文件名（这里简单使用原文件名，但你可以根据需要修改）
            output_filename = os.path.join(store_path, filename)

            # 注意：这里我们直接保存为numpy数组，但numpy数组不直接支持保存为图像文件
            # 如果你需要保存为图像文件，你需要将numpy数组转换回图像格式
            # 但由于numpy数组已经是float32的二维数组，我们可以直接保存numpy数组到文件
            # 使用numpy的save函数保存数组
            np.save(output_filename.replace('.png', '.npy'), depth_array)


if __name__ == "__main__":
    folder_name = 'gml_2024-10-15-19-38-50/'
    base_path = '/media/benny/bennyMove/data/dog_origin/'

    file_path = base_path + folder_name + 'depth_png/'
    store_path = base_path + folder_name + 'depth/'
    process_depth_images(file_path, store_path)
