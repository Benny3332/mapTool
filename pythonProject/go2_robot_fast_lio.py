# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Dict, List, Tuple

import numpy as np

from reality.robots.frame_ids import SpotFrameIds
from reality.robots.camera_ids import SpotCamIds
from reality.robots.base_robot import BaseRobot
import time
import rospy
from nav_msgs.msg import Odometry
from go2_pkg.msg import go2_fb
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs import point_cloud2
from rospy.numpy_msg import numpy_msg
import yaml
import tf
import tf.transformations
import ros_numpy
import math 
from livox_ros_driver2.msg import CustomMsg
import math

MAX_CMD_DURATION = 5

recv_first_fb = False
x, y, z, pitch_fb, roll_fb, yaw_fb = None, None, None, None, None, None
vx, vy, vz, vyaw=None, None, None, None
d435_depth = None
d435_color = None
mid_360_stack_num=10  #采集10次，增加点云稠密度
mid_360=[]
mid_360_vlmaps_stack_num=10  #采集10次，增加点云稠密度
mid_360_vlmaps=[]
recv_depth=False
recv_color=False
recv_mid_360=False
recv_mid_360_vlmaps=False
go2_vel_pub_seq=0

yaw_pid_set=0
go2_yaw_set=0
arm_mode=False

class Go2_Robot(BaseRobot):
    def __init__(self):
        pass
    
    @property
    def xy_yaw(self) -> Tuple[np.ndarray, float]:
        """Returns [x, y], yaw"""
        global x, y, yaw_fb
        return np.array([x, y]), yaw_fb
    
    def _get_camera_image(self, camera_source: str):
        global arm_mode, yaw_fb, go2_yaw_set, d435_depth, d435_color, yaw_pid_set, recv_depth, recv_color, recv_mid_360, mid_360, recv_mid_360_vlmaps, mid_360_vlmaps
        time.sleep(2)
        if 'front' in camera_source:
            if arm_mode == True:
                yaw_pid_set = go2_yaw_set
                arm_mode=False
                self._set_yaw()
            if camera_source == SpotCamIds.FRONT_DEPTH1 or camera_source == SpotCamIds.FRONT_DEPTH2:
                recv_depth=False
                while not recv_depth:
                    time.sleep(0.1)
                return d435_depth
            if camera_source == SpotCamIds.FRONT_RGB1 or camera_source == SpotCamIds.FRONT_RGB2:
                recv_color=False
                while not recv_color:
                    time.sleep(0.1)
                return d435_color
            
        if 'hand' in camera_source:
            if arm_mode == False:
                go2_yaw_set = yaw_fb
                arm_mode=True
                self._set_yaw()
            print(abs(yaw_fb-go2_yaw_set))
            if camera_source == SpotCamIds.HAND_DEPTH:
                recv_depth=False
                while not recv_depth:
                    time.sleep(0.1)
                return d435_depth
            if camera_source == SpotCamIds.HAND_RGB:
                recv_color=False
                while not recv_color:
                    time.sleep(0.1)
                return d435_color

        if camera_source == SpotCamIds.VLMAPS_RGB:
            recv_color=False
            while not recv_color:
                time.sleep(0.1)
            return d435_color
            
        if 'lidar' in camera_source:
            if camera_source == SpotCamIds.MID_360:
                mid_360=[]
                recv_mid_360=False
                while not recv_mid_360:
                    time.sleep(0.1)
                return mid_360
            elif camera_source == SpotCamIds.VLMAPS_PCD:
                mid_360_vlmaps=[]
                recv_mid_360_vlmaps=False
                while not recv_mid_360_vlmaps:
                    time.sleep(0.1)
                return mid_360_vlmaps


        raise NotImplementedError(f'{camera_source} not implemented')

    def _get_camera_info(self, camera_source: str):
        global d435i_depth_info, d435i_color_info
        if 'front' in camera_source:
            if camera_source == SpotCamIds.FRONT_DEPTH1 or camera_source == SpotCamIds.FRONT_DEPTH2:
                return d435i_depth_info
            if camera_source == SpotCamIds.FRONT_RGB1 or camera_source == SpotCamIds.FRONT_RGB2:
                return d435i_color_info
            
        if 'hand' in camera_source:                
            if camera_source == SpotCamIds.HAND_DEPTH:
                return d435i_depth_info
            if camera_source == SpotCamIds.HAND_RGB:
                return d435i_color_info
        
        if camera_source == SpotCamIds.VLMAPS_RGB:
            return d435i_color_info
        raise NotImplementedError(f'{camera_source} not implemented')


    def get_camera_images(self, camera_source: List[str]) -> Dict[str, np.ndarray]:
        """Returns a dict of images mapping camera ids to images

        Args:
            camera_source (List[str]): List of camera ids to get images from

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping camera ids to images
        """
        return {
            source: np.rot90(self._get_camera_image(source)[0], k=3) if 'lidar' not in source else self._get_camera_image(source)
            for source in camera_source
        }

    def command_base_velocity(self, ang_vel: float, lin_vel: float) -> None:
        """Commands the base to execute given angular/linear velocities, non-blocking

        Args:
            ang_vel (float): Angular velocity in radians per second
            lin_vel (float): Linear velocity in meters per second
        """
        # Just make the robot stop moving if both velocities are very low
        global arm_mode
        if arm_mode == True:
            global go2_yaw_set, yaw_pid_set
            yaw_pid_set = go2_yaw_set
            arm_mode=False
            self._set_yaw()
        self._command_base_velocity(ang_vel, lin_vel)


    def _command_base_velocity(self, ang_vel: float, lin_vel: float, sides_way: float=0) -> None:
        """Commands the base to execute given angular/linear velocities, non-blocking

        Args:
            ang_vel (float): Angular velocity in radians per second
            lin_vel (float): Linear velocity in meters per second
        """
        # if ang_vel != 0:
            # raise ValueError
        # if lin_vel != 0:
        #     raise ValueError
        # Just make the robot stop moving if both velocities are very low
        global go2_vel_pub_seq, go2_vel_pub
        msg2pub = Vector3Stamped()
        msg2pub.header.seq=go2_vel_pub_seq
        msg2pub.header.stamp = rospy.Time.now()
        msg2pub.header.frame_id = 'go2_vel_ctrl'
        msg2pub.vector.x=0 if np.abs(lin_vel) < 0.005 else lin_vel
        msg2pub.vector.z=0 if np.abs(ang_vel) < 0.005 else ang_vel
        msg2pub.vector.y=MAX_CMD_DURATION
        go2_vel_pub.publish(msg2pub)
        go2_vel_pub_seq += 1

    def get_transform(self, frame: str = SpotFrameIds.BODY) -> np.ndarray:
        """Returns the transformation matrix of the robot's base (body) or a link

        Args:
            frame (str, optional): Frame to get the transform of. Defaults to
                SpotFrameIds.BODY.

        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        global roll_fb, pitch_fb, yaw_fb, x, y, z, arm_mode, go2_yaw_set
        calc_yaw=yaw_fb
        if arm_mode: 
            calc_yaw=go2_yaw_set
        return self._get_transform(x=x, y=y, z=z, roll=roll_fb, pitch=pitch_fb, yaw=calc_yaw)
    
    def _get_transform(self, x, y, z, roll, pitch, yaw):
        return tf.transformations.translation_matrix([x,y,z])@tf.transformations.euler_matrix(roll,pitch,yaw)
    
    def set_yaw(self, angle):
        global yaw_pid_set, arm_mode
        yaw_pid_set=angle
        if arm_mode:
            self._set_yaw()
            
    def _set_yaw(self):
        global yaw_fb, yaw_pid_set
        angle=yaw_pid_set
        last_err=angle-yaw_fb
        out=0
        while(abs(yaw_fb-angle) > 0.02 or abs(vyaw) > 0.05):# 10度
            print(f'\r{angle-yaw_fb=:8.4f}\t{(out, 0)=}', end='')
            err = angle-yaw_fb
            p_out=err*yaw_pid_kp
            d_out=yaw_pid_kd*(err-last_err)
            last_err=err
            out=p_out+d_out
            if out < -yaw_pid_out_limit:
                out = -yaw_pid_out_limit
            elif out > yaw_pid_out_limit:
                out = yaw_pid_out_limit
            self._command_base_velocity(out, 0)
            time.sleep(0.02)
        print(f'\r{angle-yaw_fb=}')
        
        self._command_base_velocity(0,0)# stop move
        # breakpoint()

    def get_camera_data(self, srcs: List[str]) -> Dict[str, Dict[str, Any]]:
        """Returns a dict that maps each camera id to its image, focal lengths, and
        transform matrix (from camera to global frame).

        Args:
            srcs (List[str]): List of camera ids to get images from

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping camera ids to images
        """
        imgs = {
            src: self._camera_response_to_data(src) for src in srcs
        }
        return imgs

    def _camera_response_to_data(self, src: str) -> Dict[str, Any]:
        image_tf = self._get_camera_image(src)
        if 'image' in src or 'depth' in src:
            image = image_tf[0]
            RT_i = self._get_transform(*image_tf[1:])

            tf_camera_to_global = RT_i@np.linalg.inv(mid360_to_d435i_tf)
            cam_info:CameraInfo = self._get_camera_info(src)
            fx: float = float(cam_info['Fx'])
            fy: float = float(cam_info['Fy'])
            # breakpoint()
            return {
                "image": image,
                "fx": fx,
                "fy": fy,
                "tf_camera_to_global": tf_camera_to_global,
            }
        elif src == SpotCamIds.VLMAPS_RGB:
            cam_info:CameraInfo = self._get_camera_info(src)
            fx: float = float(cam_info['Fx'])
            fy: float = float(cam_info['Fy'])
            return {
                "image": image_tf[0],
                "fx": fx,
                "fy": fy,
                "tf_camera_to_global": self._get_transform(*image_tf[1:]),
            }
        else:
            return {
                "image": np.concatenate([image[0] for image in image_tf]),
                "fx": 0,
                "fy": 0,
                "tf_camera_to_global": self._get_transform(*image_tf[0][1:]),
            }

    def get_cmd_feedback(self, cmd_id):
        return 1
    
    def _set_base_position(
        self,
        x_pos,
        y_pos,
        yaw,
        end_time,
        relative=False,
        max_fwd_vel=1.2,
        max_hor_vel=0.5,
        max_ang_vel=np.pi / 2,
        disable_obstacle_avoidance=False,
        blocking=False,
    ):
        pass

    def set_base_position(
        self,
        x_pos,
        y_pos,
        yaw,
        end_time,
        relative=False,
        max_fwd_vel=1.2,
        max_hor_vel=0.5,
        max_ang_vel=np.pi / 2,
        disable_obstacle_avoidance=False,
        blocking=False,
    ):
        global x, y, yaw_fb, vx, vy, vyaw, yaw_pid_set
        yaw_pid_set = math.atan2(y_pos-y, x_pos-x)
                
        distance = (x-x_pos)**2 + (y-y_pos)**2
        last_err=distance
        skip=False

        yaw_angle=yaw_pid_set
        yaw_last_err=yaw_angle-yaw_fb
        # t0 = time.time()
        yaw_out, out=0, 0

        while (abs(x-x_pos) > 0.2 or abs(y-y_pos) > 0.2 or abs(vx) > 0.1 or abs(vy) > 0.1) and not skip:
            print(f'\r{x-x_pos=:2.4f},{y-y_pos=:2.4f},{yaw_angle-yaw_fb=:2.4f},{yaw_out=:2.4f}, {out=:2.4f}', end='')
            # if time.time() - t0 > 1:
            #     self._command_base_velocity(0, 0)
            #     breakpoint()
            yaw_angle = math.atan2(y_pos-y, x_pos-x)
            yaw_err = yaw_angle-yaw_fb
            while yaw_err > math.pi:
                yaw_err -= 2*math.pi
            while yaw_err < -math.pi:
                yaw_err += 2*math.pi
            p_out=yaw_err*yaw_pid_kp
            d_out=yaw_pid_kd*(yaw_err-yaw_last_err)
            yaw_last_err=yaw_err
            yaw_out=p_out+d_out
            if yaw_out < -yaw_pid_out_limit:
                yaw_out = -yaw_pid_out_limit
            elif yaw_out > yaw_pid_out_limit:
                yaw_out = yaw_pid_out_limit

            
            distance = math.sqrt((x-x_pos)**2 + (y-y_pos)**2)
            # print(f'\rmove {distance}', end='')
            err = distance
            p_out=err*yaw_pid_kp
            d_out=yaw_pid_kd*(err-last_err)
            last_err=err
            out=p_out+d_out
            if out < -yaw_pid_out_limit/5:
                out = -yaw_pid_out_limit/5
            elif out > yaw_pid_out_limit/5:
                out = yaw_pid_out_limit/5
            if yaw_err > 0.1 and abs(vyaw) > 0.05:
                out=0
            self._command_base_velocity(yaw_out, out)
            time.sleep(0.02)
        print(f'\r{x-x_pos=:2.4f} {y-y_pos=:2.4f} {yaw_angle-yaw_fb=:2.4f}')
        
        while(abs(yaw_fb-yaw_angle) > 0.1 or abs(vyaw) > 0.05):# 6度
            print(f'\r{yaw_angle-yaw_fb=:8.4f}\t {(yaw_out, 0)=}', end='')
            yaw_angle = yaw
            yaw_err = yaw_angle-yaw_fb
            p_out=yaw_err*yaw_pid_kp
            d_out=yaw_pid_kd*(yaw_err-yaw_last_err)
            yaw_last_err=yaw_err
            yaw_out=p_out+d_out
            if yaw_out < -yaw_pid_out_limit:
                yaw_out = -yaw_pid_out_limit
            elif yaw_out > yaw_pid_out_limit:
                yaw_out = yaw_pid_out_limit
            self._command_base_velocity(yaw_out, 0)
            time.sleep(0.02)
        self._command_base_velocity(0, 0)
        print(f'\r{yaw_angle-yaw_fb=}')
        
            # print(f'please move to {x_pos=}, \n{y_pos=}, \n{yaw=}')
            # print(f'current_pos{x=}, \n{y=}, \n{yaw_fb=}')
            # time.sleep(0.1)
        # vel_limit = SE2VelocityLimit(
        #     max_vel=SE2Velocity(
        #         linear=Vec2(x=max_fwd_vel, y=max_hor_vel), angular=max_ang_vel
        #     ),
        #     min_vel=SE2Velocity(
        #         linear=Vec2(x=-max_fwd_vel, y=-max_hor_vel), angular=-max_ang_vel
        #     ),
        # )
        # params = spot_command_pb2.MobilityParams(
        #     vel_limit=vel_limit,
        #     obstacle_params=spot_command_pb2.ObstacleParams(
        #         disable_vision_body_obstacle_avoidance=disable_obstacle_avoidance,
        #         disable_vision_foot_obstacle_avoidance=False,
        #         disable_vision_foot_constraint_avoidance=False,
        #         obstacle_avoidance_padding=0.05,  # in meters
        #     ),
        # )
        # curr_x, curr_y, curr_yaw = self.get_xy_yaw(use_boot_origin=True)
        # coors = np.array([x_pos, y_pos, 1.0])
        # if relative:
        #     local_T_global = self._get_local_T_global(curr_x, curr_y, curr_yaw)
        #     x, y, w = local_T_global.dot(coors)
        #     global_x_pos, global_y_pos = x / w, y / w
        #     global_yaw = wrap_heading(curr_yaw + yaw)
        # else:
        #     global_x_pos, global_y_pos, global_yaw = self.xy_yaw_home_to_global(
        #         x_pos, y_pos, yaw
        #     )
        # robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        #     goal_x=global_x_pos,
        #     goal_y=global_y_pos,
        #     goal_heading=global_yaw,
        #     frame_name=VISION_FRAME_NAME,
        #     params=params,
        # )
        # cmd_id = self.command_client.robot_command(
        #     robot_cmd, end_time_secs=time.time() + end_time
        # )

        # if blocking:
        #     cmd_status = None
        #     while cmd_status != 1:
        #         time.sleep(0.1)
        #         feedback_resp = self.get_cmd_feedback(cmd_id)
        #         cmd_status = (
        #             feedback_resp.feedback.synchronized_feedback
        #         ).mobility_command_feedback.se2_trajectory_feedback.status
        #     return None

        return 1
    
    def set_origin(self):
        global x_zeros, y_zeros, z_zeros, pitch_fb_zeros, roll_fb_zeros, yaw_fb_zeros
        x_zeros, y_zeros, z_zeros, pitch_fb_zeros, roll_fb_zeros, yaw_fb_zeros = x, y, z, pitch_fb, roll_fb, yaw_fb
        
    @property
    def lidar2cam_tf(self):
        global mid360_to_d435i_tf
        return mid360_to_d435i_tf

x_zeros, y_zeros, z_zeros, pitch_fb_zeros, roll_fb_zeros, yaw_fb_zeros=0, 0, 0, 0, 0, 0
last_x, last_y, last_yaw_fb=None, None, None
yaw_cnt=0
last_time = None
def fb_callback(msg: Odometry):
    global x, y, z, pitch_fb, roll_fb, yaw_fb, recv_first_fb, vyaw, vx, vy, last_x, last_y, last_yaw_fb, last_time, yaw_cnt
    
    recv_first_fb=True
    x, y, z = msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z
    (roll_fb, pitch_fb, yaw_fb) = tf.transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
    if last_x is not None:
        vx, vy, vyaw = (x-last_x)/(last_time - msg.header.stamp).to_sec(), (y-last_y)/(last_time - msg.header.stamp).to_sec(), (yaw_fb-last_yaw_fb)/(last_time - msg.header.stamp).to_sec()
    last_x=x
    last_y=y
    # yaw_fb += 2*math.pi*yaw_cnt
    # if yaw_fb - last_yaw_fb > math.pi:
    #     yaw_cnt -= 1
    # if yaw_fb - last_yaw_fb < -math.pi:
    #     yaw_cnt += 1

    last_yaw_fb= yaw_fb
    last_time = msg.header.stamp

    # recv_first_fb=True
    # vec3=msg.rpy
    # roll_fb, pitch_fb, yaw_fb = vec3.x - pitch_fb_zeros, vec3.y - roll_fb_zeros, vec3.z - yaw_fb_zeros
    # vec3=msg.xyz
    # x, y, z = (vec3.x - x_zeros)*math.cos(yaw_fb) - (vec3.y - y_zeros)*math.sin(yaw_fb), (vec3.x - x_zeros)*math.sin(yaw_fb) + (vec3.y - y_zeros)*math.cos(yaw_fb), vec3.z - z_zeros
    # vec3=msg.vxyz
    # vx, vy, vz = vec3.x, vec3.y, vec3.z
    # vec3=msg.gyro
    # _, _, vyaw = vec3.x, vec3.y, vec3.z

def d435_depth_callback(msg: Image):
    global d435_depth, x, y, z, pitch_fb, roll_fb, yaw_fb, recv_depth
    d435_depth = (ros_numpy.numpify(msg), float(x), float(y), float(z), float(pitch_fb), float(roll_fb), float(yaw_fb))
    recv_depth=True

def d435_color_callback(msg: Image):
    global d435_color, x, y, z, pitch_fb, roll_fb, yaw_fb, recv_color
    d435_color = (ros_numpy.numpify(msg)[:, :, ::-1], float(x), float(y), float(z), float(pitch_fb), float(roll_fb), float(yaw_fb))
    recv_color=True

def mid360_callback(msg: CustomMsg):
    global mid_360_vlmaps, x, y, z, pitch_fb, roll_fb, yaw_fb, recv_mid_360_vlmaps, mid_360_vlmaps_stack_num
    if len(mid_360_vlmaps) < mid_360_vlmaps_stack_num+1:
        mid_360_vlmaps.append(([[point.x, point.y, point.z] for point in msg.points], float(x), float(y), float(z), float(pitch_fb), float(roll_fb), float(yaw_fb)))
    else:
        recv_mid_360_vlmaps=True
    
def pcl2_callback(msg: PointCloud2):
    global mid_360, x, y, z, pitch_fb, roll_fb, yaw_fb, recv_mid_360, mid_360_stack_num
    if len(mid_360) < mid_360_stack_num+1:
        tmp = np.array([l for l in point_cloud2.read_points(msg, field_names=("x", "y", "z"))])
        tmp[:, 0], tmp[:, 1], tmp[:, 2] = tmp[:, 2], -tmp[:, 0], -tmp[:, 1]
        mid_360.append((tmp, float(x), float(y), float(z), float(pitch_fb), float(roll_fb), float(yaw_fb)))
    else:
        recv_mid_360=True

def node_init():
    global go2_vel_pub, go2_to_xt16_tf, mid360_to_d435i_tf, mid360_to_xt16_tf, xt16_to_mid360_tf, d435i_depth_info, d435i_color_info, yaw_pid_kp, yaw_pid_kd, yaw_pid_out_limit
    
    rospy.init_node('go2_robot', anonymous=True)
    rospy.Subscriber('/Odometry', Odometry, fb_callback, queue_size=1)
    go2_vel_pub = rospy.Publisher('/go2_ctrl/move', Vector3Stamped, queue_size=10)

    with open('vlfm/reality/robots/go2_rm_pkg/config/go2_to_xt16.yaml') as f:
        go2_to_xt16_tf = np.array(yaml.load(f, Loader=yaml.FullLoader)['transform'])
    with open('vlfm/reality/robots/go2_rm_pkg/config/mid360_to_d435i.yaml') as f:
        mid360_to_d435i_tf = np.array(yaml.load(f, Loader=yaml.FullLoader)['transform'])
    with open('vlfm/reality/robots/go2_rm_pkg/config/mid360_to_xt16.yaml') as f:
        mid360_to_xt16_tf = np.array(yaml.load(f, Loader=yaml.FullLoader)['transform'])
    with open('vlfm/reality/robots/go2_rm_pkg/config/xt16_to_mid360.yaml') as f:
        xt16_to_mid360_tf = np.array(yaml.load(f, Loader=yaml.FullLoader)['transform'])
    with open('vlfm/reality/robots/go2_rm_pkg/config/d435i_param_depth.yaml') as f:
        d435i_depth_info = yaml.load(f, Loader=yaml.FullLoader)
    with open('vlfm/reality/robots/go2_rm_pkg/config/d435i_param_color.yaml') as f:
        d435i_color_info = yaml.load(f, Loader=yaml.FullLoader)
    with open('vlfm/reality/robots/go2_rm_pkg/config/yaw_pid.yaml') as f:
        data=yaml.load(f, Loader=yaml.FullLoader)
        yaw_pid_kp = data['kp']
        yaw_pid_kd = data['kd']
        yaw_pid_out_limit = data['out_limit']


    while not recv_first_fb and not rospy.is_shutdown():
        print("not recv fb_data, please start fast lio node first!")
        time.sleep(1)#wait to recv msg first
        
    if rospy.is_shutdown():
        exit()

    rospy.Subscriber('/camera/depth/image_rect_raw', numpy_msg(Image), d435_depth_callback, queue_size=1)
    rospy.Subscriber('/camera/color/image_raw', numpy_msg(Image), d435_color_callback, queue_size=1)
    rospy.Subscriber('/livox/lidar', CustomMsg, mid360_callback, queue_size=1)
    rospy.Subscriber('/cloud_registered', PointCloud2, pcl2_callback, queue_size=1)

    while d435_depth is None and not rospy.is_shutdown():
        print("not recv d435_depth, please start test_sensor node first!")
        time.sleep(1)#wait to recv msg first
        
    while d435_color is None and not rospy.is_shutdown():
        print("not recv d435_color, please restart test_sensor node first!")
        time.sleep(1)#wait to recv msg first

    if rospy.is_shutdown():
        exit()


if __name__ == '__main__':
    node_init()
    while not rospy.is_shutdown():
        if recv_first_fb:
            print('recvived msg')
        time.sleep(1)