import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.cm as cm
from PIL import Image
from gmlDogRecordFilePath import file_path, file_pre_path
# file_path = "/home/benny/conda/MattHabitat/vlmaps_dataset/5LpN3gDmAk7_1/depth/000333.npy"
file_name = "000345"
folder_name = file_path
my_base_path = file_pre_path

depth_file_path = my_base_path + folder_name + f"depth_aligned/{file_name}.npy"
lidar_depth_file_path = my_base_path + folder_name + f"depth_1/{file_name}.npy"
rgb_file_path = my_base_path + folder_name + f"rgb/{file_name}.png"
# file_full_path = '/home/benny/docker/data/gml_2025-01-20-15-16-32/'
# depth_file_path =file_full_path + f"depth_aligned/{file_name}.npy"
# rgb_file_path = file_full_path + f"rgb/{file_name}.png"
data = np.load(depth_file_path)
lidar_data = np.load(lidar_depth_file_path)
rgb_image = np.array(Image.open(rgb_file_path))

click_count = 0

# print(data)

# print(data.shape)

# print(data.dtype)

# np.set_printoptions(threshold=1000, precision=3)  # 显示所有元素，小数点后3位

# print(data[0][0])
def show_depth_npy():
    global click_count
    # 假设深度图的最大值远大于255（PNG的单通道最大值），你可能需要归一化或截断数据
    # 这里我们简单地将数据归一化到0-1范围（对于显示目的），但注意这样会丢失原始深度值的比例
    # 如果你知道具体的最大深度值，可以用那个值来归一化
    print(f"data shape is {data.shape}, dtype is {data.dtype}")
    print(f"max depth is {data.max()}, min depth is {data.min()}")

    max_depth = np.max(data)  # 获取最大深度值
    # 归一化深度图并应用颜色映射
    depth_map_normalized = (data / max_depth * 255).astype(np.uint8)
    depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

    # 转换彩色深度图为RGB格式以便在matplotlib中显示
    depth_map_colored_rgb = cv2.cvtColor(depth_map_colored, cv2.COLOR_BGR2RGB)

    max_depth_lidar = np.max(lidar_data)
    depth_map_normalized_lidar = (lidar_data / max_depth_lidar * 255).astype(np.uint8)
    depth_map_colored_lidar = cv2.applyColorMap(depth_map_normalized_lidar, cv2.COLORMAP_JET)
    depth_map_colored_rgb_lidar = cv2.cvtColor(depth_map_colored_lidar, cv2.COLOR_BGR2RGB)

    # if max_depth > 0:
    #     normalized_data = data / max_depth  # 归一化
    # else:
    #     normalized_data = data  # 如果最大深度为0，则直接使用原始数据（但这样通常没有意义）

    # 创建一个图形和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 创建一个图形和轴
    # fig, ax = plt.subplots()
    im1 = ax1.imshow(depth_map_colored_rgb)
    ax1.axis('off')
    ax1.set_title('Depth camera')

    # 显示RGB图像
    im2 = ax2.imshow(depth_map_colored_rgb_lidar)
    ax2.set_title('Depth LiDAR')
    ax2.axis('off')

    # 定义一个回调函数来处理鼠标点击事件
    def on_click(event):
        global click_count
        if event.inaxes is None or event.inaxes != ax1:  # 确保只在深度图上进行点击
            return
            # 将鼠标位置（以像素为单位）转换为数据坐标
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            # 四舍五入到最近的整数索引（假设数据是整数索引的）
            ix, iy = int(np.round(x)), int(np.round(y))
            # 确保索引在数据范围内
            if 0 <= ix < data.shape[1] and 0 <= iy < data.shape[0]:
                # 在深度图上添加红点
                ax1.plot(x, y, 'ro')  # 'ro' 表示红色圆圈
                ax1.text(x, y, f'{click_count + 1}', ha='center', va='bottom', color='red')
                plt.draw()  # 重新绘制图形

                # 显示原始深度值和点击次数
                print(f"Click {click_count + 1}: Depth at ({ix}, {iy}): {data[iy, ix]}, lidar at ({ix}, {iy}): {lidar_data[iy, ix]}")
                click_count += 1

    # 如果你想保留更多的深度信息，可以考虑使用对数尺度或其他方法来映射深度值到灰度级
    # 这里我们使用简单的线性映射

    # 连接到鼠标点击事件
    fig.canvas.mpl_connect('button_press_event', on_click)

    # 显示图像
    plt.show()

    # # 使用matplotlib的imshow来显示图像，并设置cmap为灰度图
    # plt.imshow(data, cmap='gray')
    # plt.axis('off')  # 关闭坐标轴
    # plt.title('Depth Map')
    #
    # # 保存为PNG文件
    # plt.savefig('depth_map.png', bbox_inches='tight', pad_inches=0)
    #
    # # 显示图像（如果你在一个支持图形界面的环境中）
    # plt.show()


if __name__ == '__main__':
    show_depth_npy()