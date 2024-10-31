import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_depth_map_with_click(file_path):
    # 使用OpenCV加载PNG图像，以保留原始深度信息（如16位）
    depth_map = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    # OpenCV默认以BGR格式加载彩色图像，但深度图通常是灰度图
    # 如果加载的是彩色图像，需要转换为灰度图
    if len(depth_map.shape) > 2:
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
    #打印最大值的x,y和深度值
    x,y = np.unravel_index(np.argmax(depth_map), depth_map.shape)
    print(f"Maximum value of the depth map:{depth_map[x, y]} at coordinates: {x} , {y}")
    # 创建一个图形和轴
    fig, ax = plt.subplots()
    # 显示深度图，使用灰度颜色映射
    im = ax.imshow(depth_map, cmap='gray')
    ax.axis('off')  # 关闭坐标轴
    ax.set_title('Depth Map')

    # 定义一个回调函数来处理鼠标点击事件
    def on_click(event):
        if event.inaxes is None:
            return
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            ix, iy = int(np.round(x)), int(np.round(y))
            if 0 <= ix < depth_map.shape[1] and 0 <= iy < depth_map.shape[0]:
                # OpenCV的坐标是(y, x)，而matplotlib的坐标是(x, y)，这里已经进行了转换
                # 显示原始深度值（注意：这里深度图已经是灰度图，所以直接访问像素值即可）
                print(f"Depth at ({ix}, {iy}): {depth_map[iy, ix]}")

    # 连接到鼠标点击事件
    fig.canvas.mpl_connect('button_press_event', on_click)

    # 显示图形
    plt.show()


if __name__ == "__main__":
    # 示例用法
    file_path = '/home/benny/data/dog_origin/gml_2024-10-15-10-45-17/depth_png/' + '000333.png'
    display_depth_map_with_click(file_path)