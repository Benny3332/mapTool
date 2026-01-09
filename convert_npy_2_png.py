import os
import numpy as np
import matplotlib.pyplot as plt
from gmlDogRecordFilePath import file_path, file_pre_path

def convert_npy_to_heatmap_png(npy_file_path, png_save_path, vmin=0, vmax=8000):
    os.makedirs(os.path.dirname(png_save_path), exist_ok=True)

    depth_array = np.load(npy_file_path) * 1000.0  # 转为毫米
    depth_array[depth_array <= 0] = np.nan  # 可选：将无效值设为 NaN（在 colormap 中显示为透明或默认色）

    # 创建热力图（使用 jet、viridis、plasma 等 colormap）
    plt.figure(figsize=(depth_array.shape[1]/100, depth_array.shape[0]/100), dpi=100)
    plt.imshow(depth_array, cmap='jet', vmin=vmin, vmax=vmax)
    plt.axis('off')  # 去掉坐标轴
    plt.tight_layout(pad=0)
    
    # 保存为 PNG
    plt.savefig(png_save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_npys_to_pngs(npy_folder_path, png_save_folder_path):
    for filename in os.listdir(npy_folder_path):
        if filename.endswith(".npy"):
            full_path = os.path.join(npy_folder_path, filename)
            output_filename = os.path.join(png_save_folder_path, filename.replace('.npy', '.png'))
            convert_npy_to_heatmap_png(full_path, output_filename)

if __name__ == "__main__":
    folder_name = file_path
    base_path = file_pre_path
    npy_folder_path = os.path.join(base_path, folder_name, 'depth_1')
    png_save_folder_path = os.path.join(base_path, folder_name, 'depth_heatmap_png')

    print("file full path：" + os.path.join(base_path, folder_name))

    process_npys_to_pngs(npy_folder_path, png_save_folder_path)