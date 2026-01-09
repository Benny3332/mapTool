from gmlDogRecordFilePath import file_path,file_pre_path
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import h5py
def visual_voxel(pcd):
    pc = o3d.geometry.PointCloud()
    # 将NumPy数组设置为新的点云对象的点
    pc.points = o3d.utility.Vector3dVector(pcd)
    # 计算Z轴的最大最小值以便归一化
    z_values = np.array(pc.points)[:, 2]  # 提取所有点的Z坐标
    # 应用自定义的热图颜色映射
    colors = colormap_jet(z_values)
    # 将颜色分配给点云
    pc.colors = o3d.utility.Vector3dVector(colors)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=0.05)
    o3d.visualization.draw_geometries([voxel_grid])

def colormap_jet(z_values):
    # 使用matplotlib的jet colormap
    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=np.min(z_values), vmax=np.max(z_values))
    colors = cmap(norm(z_values))[:, :3]  # 只取RGB值，不取alpha通道
    return colors


def show_single_pcd():
    single_frame_pcd_path = "/home/benny/docker/data/collect_tran/k1020250228floor5big/scans.pcd"
    # single_frame_pcd_path2 = file_pre_path + file_path + '_livox_lidar/1731403560_500056982.pcd'
    # single_frame_pcd_path3 = file_pre_path + file_path + '_livox_lidar/1731403560_600377083.pcd'
    pcd = o3d.io.read_point_cloud(single_frame_pcd_path)
    # pcd2 = o3d.io.read_point_cloud(single_frame_pcd_path2)
    # pcd3 = o3d.io.read_point_cloud(single_frame_pcd_path3)
    # pcd = pcd + pcd2 + pcd3
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
    o3d.visualization.draw_geometries([voxel_grid])



def show_vlmaps_create_result():
    global pcdRS
    # map_path = file_pre_path + file_path + "vlmaps_cam.h5df"
    map_path = "/home/ws/dataset/HM3D_enviroment/vlmaps_dataset/EU6Fwq7SyZv_1/vlmap/vlmaps.h5df"
    # map_path = "/home/benny/fsdownload/vlmaps_cam_lidarDepth.h5df"
    # map_path = file_pre_path + file_path + "vlmaps_cam_lidarDepth.h5df"
    with h5py.File(map_path, "r") as f:
        mapped_iter_list = f["mapped_iter_list"][:].tolist()
        grid_feat = f["grid_feat"][:]
        grid_pos = f["grid_pos"][:]
        weight = f["weight"][:]
        occupied_ids = f["occupied_ids"][:]
        grid_rgb = f["grid_rgb"][:]
        pcd_min = f["pcd_min"][:]
        pcd_max = f["pcd_max"][:]
        # cs = f["cs"][()]
    grid_height = grid_pos[:, 2] * 0.05
    grid_height_mask = np.logical_and(grid_height > -1.55, grid_height < 20.0)
    grid_pos = grid_pos[grid_height_mask, :]
    rgb = grid_rgb[grid_height_mask, :]
    rgb = rgb / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(grid_pos)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
        # 计算 XY 方向的最大、最小值
    xy_min = grid_pos[:, :2].min(axis=0)   # shape (2,)  [x_min, y_min]
    xy_max = grid_pos[:, :2].max(axis=0)   # shape (2,)  [x_max, y_max]

    print("XY min:", xy_min)
    print("XY max:", xy_max)
    o3d.visualization.draw_geometries([pcd])


def show_pcd():
    pc_path = file_pre_path + file_path + "scans.pcd"
    path = "/home/benny/docker/data/collect_tran/k1020250228floor5big/scans.pcd"
    global_pcd = o3d.io.read_point_cloud(path)
    pc = np.asarray(global_pcd.points)
    mask = pc[:, 2] <= 2.75
    pc_filter = pc[mask]
    visual_voxel(pc_filter)


if __name__ == '__main__':
    # show_single_pcd()
    show_vlmaps_create_result()
    # show_pcd()
    print("hello world")
