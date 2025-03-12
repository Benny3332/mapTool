import cv2
import os
from pathlib import Path
from gmlDogRecordFilePath import file_path
from gmlDogRecordFilePath import file_pre_path

def create_video_from_images(image_dir, output_video_path, fps=15):
    # 获取所有 PNG 文件
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    if not image_files:
        print("No PNG files found in the directory.")
        return

    # 读取第一张图片以确定视频的宽度和高度
    first_image_path = os.path.join(image_dir, image_files[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # 定义视频编码器和输出文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 逐帧读取图片并写入视频
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # 释放视频写入器
    video_writer.release()
    print(f"Video saved to {output_video_path}")

def main():
    folder_path = file_path
    base_path = file_pre_path
    # 图片文件夹路径
    image_dir = base_path + folder_path + "rgb/"
    # 输出视频文件路径
    output_video_path = base_path + folder_path + "video.mp4"
    print("file full path：" + base_path + folder_path)

    # 创建视频
    create_video_from_images(image_dir, output_video_path)

if __name__ == "__main__":
    main()