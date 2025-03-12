# 写一个从/home/benny/conda/vlmaps_dataset/5LpN3gDmAk7_1/pose 中提取出poses.txt的程序
# pose中有若干5LpN3gDmAk7_0.txt的文件
# 每个文件中只有一行四元数
# 提取出每个文件中的四元数
# 保存到/home/benny/conda/vlmaps_dataset/5LpN3gDmAk7_1/poses.txt中


import os

# 定义源目录和目标文件路径
source_dir = '/home/benny/conda/vlmaps_dataset/5LpN3gDmAk7_1/pose'
target_file = '/home/benny/conda/vlmaps_dataset/5LpN3gDmAk7_1/poses.txt'

# 检查源目录是否存在
if not os.path.exists(source_dir):
    print(f"源目录 {source_dir} 不存在，请检查路径。")
    exit(1)

# 创建一个列表来存储所有提取的四元数
quaternions = []

# 遍历源目录下的所有文件
for filename in sorted(os.listdir(source_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x and x.endswith('.txt') else float('inf')):
    if filename.startswith('5LpN3gDmAk7_') and filename.endswith('.txt'):
        # 构造完整的文件路径
        file_path = os.path.join(source_dir, filename)

        # 尝试打开并读取文件
        try:
            with open(file_path, 'r') as file:
                # 读取文件中的一行，即四元数
                quaternion = file.readline().strip()
                # 将四元数添加到列表中
                quaternions.append(quaternion)
        except Exception as e:
            print(f"无法读取文件 {file_path}，错误：{e}")

# 将所有四元数写入目标文件
try:
    with open(target_file, 'w') as file:
        for quaternion in quaternions:
            file.write(quaternion + '\n')
except Exception as e:
    print(f"无法写入文件 {target_file}，错误：{e}")

print("四元数提取完成并已保存到目标文件。")