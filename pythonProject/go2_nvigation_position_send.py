# from go2_robot_fast_lio import Go2_Robot, node_init
import math
def send_position():
    # robot = Go2_Robot()
    # [[0.0, 0.0], [5.438, 0.841]]

    aim_list =  [[2.764, -0.015], [212.3826834323651, 165.07612046748872],
                 [198.0, 165.0], [175.0, 178.0], [169.0, 189.0], [168.0, 190.0], [141.0, 191.0], [114.0, 191.0]]
    start_pos = aim_list[0]
    next_pos = [0, 0]
    print(f"start send position")
    print("----------------------------")
    for i in range(len(aim_list) - 1):
        next_pos = aim_list[i + 1]
        print(f"next pos: {start_pos}")
        angle_radians = math.atan2(next_pos[1] - start_pos[1], next_pos[0] - start_pos[0])
        YAM = math.degrees(angle_radians)
        print(f"next Yam : {YAM}")
        print("----------------------------")
        # robot.set_base_position(0, 0, angle_radians)
    # robot.set_base_position(5.438, 0.841, angle_radians)
    print(f"finish send position")
if __name__ == '__main__':
    # node_init()
    send_position()