import os
from PIL import Image
import numpy as np
from gmlDogRecordFilePath import file_path, file_pre_path
def convert_npy_to_png(npy_file_path, png_save_path):
    # 确保输出目录存在
    os.makedirs(os.path.dirname(png_save_path), exist_ok=True)

    # 加载numpy数组
    depth_array = np.load(npy_file_path)

    # 将单位从米转换为毫米
    depth_array *= 1000.0

    # 设置所有小于等于0的值为1（避免在转换为uint16时出现问题）
    depth_array[depth_array <= 0] = 1

    # 将深度值限制在合理的范围内（例如0到6000毫米）
    depth_array = np.clip(depth_array, 0, 8000)

    # 将浮点数数组转换为无符号16位整数数组
    depth_array_uint16 = depth_array.astype(np.uint16)

    # 创建PIL图像对象
    img = Image.fromarray(depth_array_uint16)

    # 保存图像
    img.save(png_save_path)

def process_npys_to_pngs(npy_folder_path, png_save_folder_path):
    # 遍历npy_folder_path中的所有.npy文件
    for filename in os.listdir(npy_folder_path):
        if filename.endswith(".npy"):
            # 完整路径
            full_path = os.path.join(npy_folder_path, filename)

            # 构造输出文件名（这里简单使用原文件名，但你可以根据需要修改）
            output_filename = os.path.join(png_save_folder_path, filename.replace('.npy', '.png'))

            # 调用convert_npy_to_png函数进行转换
            convert_npy_to_png(full_path, output_filename)

if __name__ == "__main__":
    folder_name = file_path
    base_path = file_pre_path
    npy_folder_path = base_path + folder_name + 'depth_aligned'  # 替换为你的.npy文件夹路径
    png_save_folder_path = base_path + folder_name + 'depth_aligned_png'  # 替换为你想要保存.png文件的路径

    print("file full path：" + base_path + folder_name)

    process_npys_to_pngs(npy_folder_path, png_save_folder_path)



