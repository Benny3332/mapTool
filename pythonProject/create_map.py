import open3d as o3d
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Set
from scipy.spatial.transform import Rotation as R

from gmlDogRecordFilePath import file_path, file_pre_path

def load_depth_npy(depth_filepath: Union[Path, str]):
    with open(depth_filepath, "rb") as f:
        depth = np.load(f)
    return depth

def backproject_depth(
    depth: np.ndarray,
    calib_mat: np.ndarray,
    depth_sample_rate: int,
    min_depth: float = 0.1,
    max_depth: float = 10,
) -> np.ndarray:
    pc, mask = depth2pc(depth, intr_mat=calib_mat, min_depth=min_depth, max_depth=max_depth)  # (3, N)
    shuffle_mask = np.arange(pc.shape[1])
    np.random.shuffle(shuffle_mask)
    shuffle_mask = shuffle_mask[::depth_sample_rate]
    mask = mask[shuffle_mask]
    pc = pc[:, shuffle_mask]
    pc = pc[:, mask]
    return pc

def depth2pc(depth, fov=90, intr_mat=None, min_depth=0.1, max_depth=10):
    # 获取深度图的高度和宽度
    h, w = depth.shape
    # 初始化相机内参矩阵
    cam_mat = intr_mat
    cam_mat_inv = np.linalg.inv(cam_mat)
    # 生成网格的x和y坐标
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    # 将x坐标调整为表示像素中心的位置
    x = x.reshape((1, -1))[:, :] + 0.5
    # 将y坐标调整为表示像素中心的位置
    y = y.reshape((1, -1))[:, :] + 0.5
    # 将深度值重塑为一维数组
    z = depth.reshape((1, -1))[:, :]
    # 将x, y坐标和1（表示齐次坐标）堆叠成二维点坐标矩阵p_2d
    p_2d = np.vstack([x, y, np.ones_like(x)])
    # 使用相机内参矩阵的逆矩阵将二维点坐标转换为三维点坐标
    pc = cam_mat_inv @ p_2d
    # 将三维点坐标的z分量与对应的深度值相乘，得到最终的三维点云坐标
    pc = pc * z
    # 生成一个布尔数组，表示哪些点的z分量大于最小深度值
    mask = pc[2, :] > min_depth
    # 使用逻辑与操作，进一步筛选出在最小和最大深度值之间的点
    mask = np.logical_and(mask, pc[2, :] < max_depth)
    # pc = pc[:, mask]  # 这行代码被注释掉了，可能是不需要过滤点云或者为了调试
    # 返回三维点云坐标和掩码
    return pc, mask

def transform_pc(pc, pose):
    """
    pose: the pose of the camera coordinate where the pc is in
    """
    # pose_inv = np.linalg.inv(pose)

    pc_homo = np.vstack([pc, np.ones((1, pc.shape[1]))])

    pc_global_homo = pose @ pc_homo

    return pc_global_homo[:3, :]

def cvt_pose_vec2tf(pos_quat_vec: np.ndarray) -> np.ndarray:
    """
    pos_quat_vec: (px, py, pz, qx, qy, qz, qw)
    """
    pose_tf = np.eye(4)
    pose_tf[:3, 3] = pos_quat_vec[:3].flatten()
    rot = R.from_quat(pos_quat_vec[3:].flatten())
    pose_tf[:3, :3] = rot.as_matrix()
    # rot_change = R.from_euler('xyz', [-90, 0, -90], degrees=True)
    # rot_change = R.from_euler('xyz', [0, 90, 90], degrees=True)
    # rot = R.from_quat(pos_quat_vec[3:].flatten())
    # rot = rot * rot_change
    # pose_tf[:3, :3] = rot.as_matrix()
    return pose_tf

def main():
    calib_mat = np.array([
    [604.5459594726562, 0, 432.69287109375],
    [0, 604.0941772460938, 254.2894287109375],
    [0, 0, 1]
    ])

    depth_sample_rate = int(100)

    file_full_path = '/home/benny/docker/noetic_container_data/bag/extra/gml_2025-01-20-15-16-32'
    data_dir = Path(file_full_path)
    depth_paths_dir = data_dir / "depth_1"
    depth_paths = sorted(depth_paths_dir.glob("*.npy"))
    pose_path = data_dir / "poses.txt"
    camera_pose_tfs = np.loadtxt(pose_path)
    camera_pose_tfs = [cvt_pose_vec2tf(x) for x in camera_pose_tfs]
    pbar = tqdm(
        zip(depth_paths, camera_pose_tfs),
        total=len(depth_paths),
        desc="Get Global Map",
    )
    global_pcd = o3d.geometry.PointCloud()
    for frame_i, (depth_path, camera_pose_tf) in enumerate(pbar):
        # 加载深度图数据
        depth = load_depth_npy(depth_path.as_posix())
        # 将深度图反投影到三维空间，生成点云
        pc = backproject_depth(depth, calib_mat, depth_sample_rate, min_depth=0.1, max_depth=10)
        # 获取相机姿态变换（此处假设transform_tf已经包含了所有必要的变换）
        transform_tf = camera_pose_tf  # @ self.habitat2cam_rot_tf（这一行被注释掉了，可能是为了展示原始的旋转变换）
        # 将点云从相机坐标系转换到全局坐标系
        pc_global = transform_pc(pc, transform_tf)  # (3, N)，其中N是点的数量
        # 创建一个新的全局点云对象
        pcd_global = o3d.geometry.PointCloud()
        # 设置全局点云的点数据（注意：这里假设global_pcd可以累加点云，实际可能需要使用o3d.geometry.PointCloud.concatenate_point_clouds）
        pcd_global.points = o3d.utility.Vector3dVector(pc_global.T)
        # 将新生成的全局点云累加到全局点云集合中（注意：这里假设global_pcd是一个可以累加点云的特殊对象或列表）
        global_pcd += pcd_global


    origin = np.zeros(3)  # 坐标轴的原点
    axis_length = 1.0  # 坐标轴的长度
    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(axis_length, origin)
    o3d.visualization.draw_geometries([global_pcd, axis_frame])

if __name__ == "__main__":
    main()