import cv2

# 读取本地图片
image_path = '/media/benny/bennyMove/temp2/123.png'  # 替换为你的图片路径
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not read the image from {image_path}")
else:
    # 将图片转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 保存灰度图
    output_path = '/media/benny/bennyMove/temp2/1234.png'  # 输出文件名
    cv2.imwrite(output_path, gray_image)
    print(f"Gray image saved as {output_path}")