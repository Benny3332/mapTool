# from go2_robot_fast_lio_wzh import Go2_Robot, node_init
import math


def send_position():
    # robot = Go2_Robot()
    # [[0.0, 0.0], [5.438, 0.841]]
    # [1.864, 7.385], [1.214, 7.385], [0.064, 8.035], [-0.236, 8.585], [-0.286, 8.635], [-1.786, 8.685]

    aim_list = [[2.764, -0.015], [2.764, -0.015]]
    start_pos = aim_list[0]
    next_pos = [0, 0]
    angle_radians = 0.0
    print(f"start send position")
    for i in range(len(aim_list) - 1):
        print("\r\n----------------------------")
        next_pos = aim_list[i + 1]
        print(f"next pos(aim_list[{i}]): {next_pos}")
        if not (next_pos[1] == start_pos[1] and next_pos[0] == start_pos[0]):
            angle_radians = math.atan2(next_pos[1] - start_pos[1], next_pos[0] - start_pos[0])
        YAM = math.degrees(angle_radians)
        print(f"next Yam : {YAM}")
        # result = robot.set_base_position(start_pos[0], start_pos[1], angle_radians)
        start_pos = next_pos
        print("----------------------------\r\n")
    print(f"finish send position")


if __name__ == '__main__':
    # node_init()
    send_position()