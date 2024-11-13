import os
import re
from pathlib import Path
import bisect
from gmlDogRecordFilePath import file_path


class PoseData:
    def __init__(self, timestamp, position_quaternion):
        self.timestamp = timestamp
        self.position_quaternion = position_quaternion


def read_pose_data(file_path):
    poses = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = list(map(float, line.split()))
            timestamp = parts[0]
            position_quaternion = parts[1:]
            poses.append(PoseData(timestamp, position_quaternion))
    return sorted(poses, key=lambda p: p.timestamp)


def get_png_filenames(dir_path):
    file_names = []
    for entry in os.scandir(dir_path):
        if entry.is_file() and entry.name.endswith('.png'):
            name = entry.name
            match = re.match(r'(\d+)_(\d+)\.png', name)
            if match:
                timestamp = f'{match.group(1)}.{match.group(2)}'
                file_names.append(timestamp)
    return sorted(file_names, key=float)


def find_closest_pose(poses, timestamp):
    timestamps = [p.timestamp for p in poses]
    index = bisect.bisect_left(timestamps, timestamp)
    if index == 0:
        return poses[0]
    if index == len(poses):
        return poses[-1]
    before = poses[index - 1]
    after = poses[index]
    if (after.timestamp - timestamp) < (timestamp - before.timestamp):
        return after
    else:
        return before


def main():
    print("This is extra RGB TUM function")
    # file_name = "gml_2024-11-05-14-49-29"
    file_name = file_path
    base_dir = "/media/benny/bennyMove/data/dog_origin/"

    dir_path = os.path.join(base_dir, file_name, "_camera_color_image_raw")
    pose_file_path = os.path.join(base_dir, file_name, "pose_200hz.txt")
    output_file_path = os.path.join(base_dir, file_name, "pose_mid_360.txt")
    output_tamp_file_path = os.path.join(base_dir, file_name, "pose_mid_360_tamp.txt")

    poses = read_pose_data(pose_file_path)
    png_files = get_png_filenames(dir_path)

    with open(output_file_path, 'w') as output_file, \
            open(output_tamp_file_path, 'w') as output_tamp_file:
        for file_name in png_files:
            timestamp = float(file_name)
            closest_pose = find_closest_pose(poses, timestamp)

            output_line = ' '.join(map(str, closest_pose.position_quaternion))
            tamp_line = f'{file_name} {output_line}'

            output_file.write(output_line + '\n')
            output_tamp_file.write(tamp_line + '\n')


if __name__ == "__main__":
    main()