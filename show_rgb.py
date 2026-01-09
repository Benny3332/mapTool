import cv2
import os
from gmlDogRecordFilePath import file_path, file_pre_path


file_name = "000301"

# 构建完整路径（使用os.path.join确保跨平台兼容性）
my_base_path = file_pre_path
folder_name = file_path

# 构建RGB图像路径
rgb_dir = os.path.join(my_base_path, folder_name, "rgb")
rgb_file_path = os.path.join(rgb_dir, f"{file_name}.png")

# 可选：构建深度图路径（根据需求使用）
depth_dir = os.path.join(my_base_path, folder_name, "depth_1")
depth_file_path = os.path.join(depth_dir, f"{file_name}.npy")
lidar_depth_file_path = depth_file_path  # 与深度图相同（根据原始描述）

# 检查RGB文件是否存在
if not os.path.exists(rgb_file_path):
    print(f"错误: RGB文件不存在! 路径: {rgb_file_path}")
    print("请检查以下配置:")
    print(f"- 基础路径: {my_base_path}")
    print(f"- 场景文件夹: {folder_name}")
    print(f"- 图像目录: {rgb_dir}")
    print(f"- 文件名: {file_name}.png")
    exit(1)

# 读取并显示RGB图像
rgb_image = cv2.imread(rgb_file_path)

if rgb_image is None:
    print(f"错误: 无法加载图像! 路径: {rgb_file_path}")
    print("可能原因: 文件损坏、不支持的格式或OpenCV编译问题")
    exit(1)

print(f"成功加载图像! 尺寸: {rgb_image.shape[1]}x{rgb_image.shape[0]}")
print(f"数据类型: {rgb_image.dtype}")

# 显示图像
window_name = f"RGB Image: {file_name}"
cv2.imshow(window_name, rgb_image)
cv2.waitKey()
cv2.destroyAllWindows()