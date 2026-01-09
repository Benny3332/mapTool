import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def batch_convert_npy_to_png(npy_folder, output_folder=None, colormap='plasma', invalid_value=0, normalize=True):
    """
    批量将 .npy 深度图转为 PNG 热力图
    
    参数:
        npy_folder (str): .npy 深度图所在文件夹（必须存在）
        output_folder (str): 输出 PNG 文件夹（默认为 npy_folder 下的 depth_png）
        colormap (str): 热力图颜色方案，如 'plasma', 'inferno', 'viridis'
        invalid_value: 视为无效的深度值（如 0），将被设为白色
        normalize: 是否对每张图单独归一化（强烈建议 True）
    """
    npy_folder = Path(npy_folder).resolve()
    if not npy_folder.is_dir():
        raise FileNotFoundError(f"❌ 深度图文件夹不存在，请检查路径: {npy_folder}")

    if output_folder is None:
        output_folder = npy_folder / "depth_png"
    else:
        output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(npy_folder.glob("*.npy"))
    if not npy_files:
        print("⚠️  警告：在该文件夹中未找到任何 .npy 文件！")
        return

    print(f"✅ 找到 {len(npy_files)} 个 .npy 文件，开始转换...")

    for npy_path in npy_files:
        try:
            depth = np.load(npy_path)

            # 处理无效值（如 0 表示无深度）
            if invalid_value is not None:
                depth = depth.astype(np.float32)
                depth[depth == invalid_value] = np.nan

            # 归一化到 [0, 1]（仅基于有效像素）
            if normalize and np.any(~np.isnan(depth)):
                vmin, vmax = np.nanmin(depth), np.nanmax(depth)
                if vmin != vmax:
                    depth = (depth - vmin) / (vmax - vmin)
                else:
                    depth = np.nan_to_num(depth, nan=0.0)  # 全相同值设为 0
            else:
                depth = np.nan_to_num(depth, nan=0.0)

            # 渲染为 colormap 图像
            cmap = plt.get_cmap(colormap)
            colored = cmap(depth)  # (H, W, 4)

            # 保存为 RGB PNG（NaN 区域已转为 0，colormap 中是最低色，通常是黑/紫）
            # 若希望无效区域为白色，可后处理：
            mask_invalid = np.isnan(np.load(npy_path).astype(np.float32))
            if invalid_value is not None and np.any(mask_invalid):
                colored[mask_invalid] = [1.0, 1.0, 1.0, 1.0]  # 白色

            rgb_img = (colored[:, :, :3] * 255).astype(np.uint8)

            # 保存
            png_path = output_folder / (npy_path.stem + ".png")
            plt.imsave(png_path, rgb_img, format='png')

        except Exception as e:
            print(f"❌ 处理 {npy_path.name} 时出错: {e}")
            continue

    print(f"🎉 转换完成！PNG 已保存至:\n{output_folder}")


if __name__ == "__main__":
    npy_file_folder = "/home/ws/dataset/HM3D_enviroment/vlmaps_dataset/JmbYfDe2QKZ_2/depth/"

    # 可选：指定输出文件夹，若为 None 则自动在原目录下建 depth_png/
    output_folder = "/home/ws/Pictures/fig3/depth"

    # 执行转换
    batch_convert_npy_to_png(
        npy_folder=npy_file_folder,
        output_folder=output_folder,
        colormap='plasma',
        invalid_value=0,
        normalize=True
    )