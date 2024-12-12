import os
from pathlib import Path
import shutil
from gmlDogRecordFilePath import file_path,file_pre_path

def get_png_filenames(dir_path):
    file_names = []
    for entry in os.scandir(dir_path):
        if entry.is_file() and entry.name.endswith('.png'):
            name = entry.name
            if name != '.' and name != '..':
                underscore_pos = name.find('_')
                time_str = name[:underscore_pos] + '.' + name[underscore_pos + 1:-4]
                file_names.append(time_str)
    return file_names


def main():
    print("This is Change RGB file function")
    # 图片文件夹路径
    # file_name = "gml_2024-11-05-14-49-29"
    file_name = file_path
    is_rgb = True

    base_dir = file_pre_path

    dir_path, tran_path = ('', '')

    if is_rgb:
        dir_path = os.path.join(base_dir, file_name, "_camera_color_image_raw")
        tran_path = os.path.join(base_dir, file_name, "rgb")
    else:
        dir_path = os.path.join(base_dir, file_name, "_camera_depth_image_rect_raw")
        tran_path = os.path.join(base_dir, file_name, "depth_png")

    if not os.path.exists(tran_path):
        os.makedirs(tran_path)
        print(f"Created directory: {tran_path}")

    print(f"dirPath: {dir_path}")
    print(f"tranPath: {tran_path}")

    png_files = get_png_filenames(dir_path)
    png_files.sort(key=lambda x: float(x))

    for i, png_file in enumerate(png_files):
        stamp = png_file.replace('.', '_')
        file_name = stamp + ".png"
        # 构建原始文件路径和新文件路径
        original_file_path = os.path.join(dir_path, file_name)
        new_file_name = f"{i:06}.png"
        new_file_path = os.path.join(tran_path, new_file_name)
        # 复制文件
        shutil.copy2(original_file_path, new_file_path)
        # print(new_file_path)


if __name__ == "__main__":
    main()