# from go2_robot_fast_lio import Go2_Robot, node_init
import math
def send_position():
    # robot = Go2_Robot()
    # [[0.0, 0.0], [5.438, 0.841]]
    # [[5.438, 0.891], [5.438, 2.141], [5.588, 6.541], [5.588, 6.991], [5.538, 7.141], [4.288, 10.191]]
    angle_radians = math.atan2(0.841 - 0, 5.438 - 0)
    YAM = math.degrees(angle_radians)
    print(YAM)
    # robot.set_base_position(0, 0, angle_radians)
    # robot.set_base_position(5.438, 0.841, angle_radians)

if __name__ == '__main__':
    # node_init()
    send_position()