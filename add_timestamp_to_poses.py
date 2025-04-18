from gmlDogRecordFilePath import file_path
from gmlDogRecordFilePath import file_pre_path

def prepend_timestamps(poses_path, stamp_path, store_path):
    # 读取时间戳文件
    with open(stamp_path, 'r') as stamp_file:
        timestamps = [line.split()[0] for line in stamp_file]  # 假设时间戳是每行的第一个词

    # 读取poses文件
    poses = []
    with open(poses_path, 'r') as poses_file:
        poses = poses_file.readlines()

    # 检查两个文件行数是否一致
    # if len(timestamps) != len(poses):
    #     raise ValueError("时间戳文件和poses文件的行数不一致")

    # 将时间戳添加到poses文件的每一行前
    stamped_poses = [f"{timestamp} {pose}" for timestamp, pose in zip(timestamps, poses)]

    # 写入到新的文件
    with open(store_path, 'w') as store_file:
        store_file.writelines(stamped_poses)





if __name__ == '__main__':
    folder_name = file_path
    # file_name = 'poses.txt'

    file_name = 'poses_camera.txt'

    base_path = file_pre_path
    print("file full path：" + base_path + folder_name)
    poses_path = base_path + folder_name + file_name
    store_path = base_path + folder_name + 'poses_stamped.txt'
    stamp_path = base_path + folder_name + 'pose_mid_360_tamp.txt'

    prepend_timestamps(poses_path, stamp_path, store_path)