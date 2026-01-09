import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from gmlDogRecordFilePath import file_path, file_pre_path

# 配置文件路径
file_name = "000131"
file_name_2 = "000131"
folder_name = file_path
my_base_path = file_pre_path

depth_file_path = my_base_path + folder_name + f"depth/{file_name}.npy"
rgb_file_path = my_base_path + folder_name + f"rgb/{file_name_2}.png"

def overlay_depth_heatmap_on_rgb(alpha=0.5, downsample_factor=1):
    """
    将深度图热力版本降采样后叠加到RGB图像上
    
    参数:
    alpha -- 热力图透明度 (0-1)
    downsample_factor -- 降采样因子 (1=原始尺寸, 2=半尺寸等)
    """
    # 读取数据
    depth_data = np.load(depth_file_path)
    rgb_image = np.array(Image.open(rgb_file_path))
    
    # 深度图归一化 (处理可能的极值)
    valid_depth = depth_data[np.isfinite(depth_data) & (depth_data > 0)]
    if len(valid_depth) == 0:
        raise ValueError("深度图中没有有效数据")
    
    # 使用95%分位数避免异常值影响
    max_depth = np.percentile(valid_depth, 95)
    min_depth = np.min(valid_depth)
    
    # 归一化到0-1范围
    normalized_depth = np.clip((depth_data - min_depth) / (max_depth - min_depth), 0, 1)
    
    # 转换为热力图 (BGR格式)
    heatmap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # 降采样热力图
    target_size = (rgb_image.shape[1] // downsample_factor, rgb_image.shape[0] // downsample_factor)
    heatmap_resized = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_AREA)
    
    # 调整热力图到原始RGB尺寸
    heatmap_full = cv2.resize(heatmap_resized, (rgb_image.shape[1], rgb_image.shape[0]), 
                             interpolation=cv2.INTER_LINEAR)
    
    # 转换颜色空间 (BGR -> RGB 用于matplotlib)
    heatmap_rgb = cv2.cvtColor(heatmap_full, cv2.COLOR_BGR2RGB)
    
    # 叠加热力图和RGB图像
    overlay = cv2.addWeighted(rgb_image, 1 - alpha, heatmap_rgb, alpha, 0)
    
    # 创建显示结果
    plt.figure(figsize=(15, 10))
    
    # 原始RGB
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_image)
    plt.title('原始RGB图像')
    plt.axis('off')
    
    # 热力图
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_rgb)
    plt.title('深度热力图')
    plt.axis('off')
    
    # 叠加结果
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f'叠加结果 (alpha={alpha}, 降采样={downsample_factor}x)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'overlay_{file_name}.png', bbox_inches='tight')
    plt.show()
    
    return overlay

if __name__ == '__main__':
    # 调用函数: alpha=0.4 (40%透明度), 降采样因子=2
    result = overlay_depth_heatmap_on_rgb(alpha=0.4, downsample_factor=2)
    
    # 保存结果
    cv2.imwrite(f'overlay_result_{file_name}.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"结果已保存为 overlay_result_{file_name}.jpg")