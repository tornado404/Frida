#! /usr/bin/env python
import os

import mujoco
import numpy as np
import time
import cv2

from loguru import logger

from brush_stroke import euler_from_quaternion

##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

class Robot:
    '''
        Low-level action functionality of the robot.
        This is an abstract class, see its children for usage.
    '''
    def __init__(self, debug, node_name="painting"):
        self.debug_bool = debug

    def debug(self, msg):
        if self.debug_bool:
            print(msg)

    def good_morning_robot(self):
        raise Exception("This method must be implemented")

    def good_night_robot(self):
        raise Exception("This method must be implemented")

    def go_to_cartesian_pose(self, positions, orientations, precise=False, move_by_joint=False, speed=80):
        raise Exception("This method must be implemented")

class XArm(Robot, object):
    '''
        Low-level action functionality of the robot.
        This is an abstract class, see its children for usage.
    '''
    def __init__(self, ip, debug):
        from xarm.wrapper import XArmAPI
        self.debug_bool = debug

        self.arm = XArmAPI(ip)

    def debug(self, msg):
        if self.debug_bool:
            print(msg)

    def good_morning_robot(self):
        self.arm.motion_enable(enable=True)
        self.arm.reset(wait=True)
        self.arm.set_mode(0)
        self.arm.reset(wait=True)
        self.arm.set_state(state=0)

        self.arm.reset(wait=True)

    def good_night_robot(self):
        self.arm.disconnect()

    def go_to_cartesian_pose(self, positions, orientations,
            speed=250):
        # positions in meters
        positions, orientations = np.array(positions), np.array(orientations)
        if len(positions.shape) == 1:
            positions = positions[None,:]
            orientations = orientations[None,:]
        for i in range(len(positions)):
            x,y,z = positions[i][1], positions[i][0], positions[i][2]
            x,y,z = x*1000, y*-1000, z*1000 #m to mm
            q = orientations[i]
            
            euler= euler_from_quaternion(q[0], q[1], q[2], q[3])#quaternion.as_quat_array(orientations[i])
            roll, pitch, yaw = 180, 0, 0#euler[0], euler[1], euler[2]
            # https://github.com/xArm-Developer/xArm-Python-SDK/blob/0fd107977ee9e66b6841ea9108583398a01f227b/xarm/x3/xarm.py#L214
            
            wait = True 
            failure, state = self.arm.get_position()
            if not failure:
                curr_x, curr_y, curr_z = state[0], state[1], state[2]
                # print('curr', curr_y, y)
                dist = ((x-curr_x)**2 + (y-curr_y)**2 + (z-curr_z)**2)**0.5
                # print('dist', dist)
                # Dist in mm
                if dist < 5:
                    wait=False
                    speed=600
                    # print('less')

            try:
                r = self.arm.set_position(
                        x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                        speed=speed, wait=wait
                )
                # print(r)
                if r:
                    print("failed to go to pose, resetting.")
                    self.arm.clean_error()
                    self.good_morning_robot()
                    self.arm.set_position(
                            x=x, y=y, z=z+5, roll=roll, pitch=pitch, yaw=yaw,
                            speed=speed, wait=True
                    )
                    self.arm.set_position(
                            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                            speed=speed, wait=True
                    )
            except Exception as e:
                self.good_morning_robot()
                print('Cannot go to position', e)


class Franka(Robot, object):
    '''
        Low-level action functionality of the Franka robot.
    '''
    def __init__(self, debug, node_name="painting"):
        import sys
        sys.path.append('~/Documents/frankapy/frankapy/')
        from frankapy import FrankaArm

        self.debug_bool = debug
        self.fa = FrankaArm()

        # reset franka to its home joints
        self.fa.reset_joints()

    def debug(self, msg):
        if self.debug_bool:
            print(msg)

    def good_morning_robot(self):
        # reset franka to its home joints
        self.fa.reset_joints()

    def good_night_robot(self):
        # reset franka back to home
        self.fa.reset_joints()

    def create_rotation_transform(pos, quat):
        from autolab_core import RigidTransform
        rot = RigidTransform.rotation_from_quaternion(quat)
        rt = RigidTransform(rotation=rot, translation=pos,
                from_frame='franka_tool', to_frame='world')
        return rt

    def sawyer_to_franka_position(pos):
        # Convert from sawyer code representation of X,Y to Franka
        pos[0] *= -1 # The x is oposite sign from the sawyer code
        pos[:2] = pos[:2][::-1] # The x and y are switched compared to sawyer for which code was written
        return pos
    
    def go_to_cartesian_pose(self, positions, orientations):
        """
            Move to a list of points in space
            args:
                positions (np.array(n,3)) : x,y,z coordinates in meters from robot origin
                orientations (np.array(n,4)) : x,y,z,w quaternion orientation
                precise (bool) : use precise for slow short movements. else use False, which is fast but unstable
        """
        positions, orientations = np.array(positions), np.array(orientations)
        if len(positions.shape) == 1:
            positions = positions[None,:]
            orientations = orientations[None,:]

        # if precise:
        #     self.go_to_cartesian_pose_precise(positions, orientations, stiffness_factor=stiffness_factor)
        # else:
        #     self.go_to_cartesian_pose_stable(positions, orientations)
        self.go_to_cartesian_pose_stable(positions, orientations)


    def go_to_cartesian_pose_stable(self, positions, orientations):
        abs_diffs = []
        # start = time.time()
        for i in range(len(positions)):
            pos = Franka.sawyer_to_franka_position(positions[i])
            rt = Franka.create_rotation_transform(pos, orientations[i])

            # Determine speed/duration
            curr_pos = self.fa.get_pose().translation
            # print(curr_pos, pos)
            dist = ((curr_pos - pos)**2).sum()**.5 
            # print('distance', dist*100)
            duration = dist * 5 # 1cm=.1s 1m=10s
            duration = max(0.6, duration) # Don't go toooo fast
            if dist*100 < 0.8: # less than 0.8cm
                # print('very short')
                duration = 0.3
            # print('duration', duration, type(duration))
            duration = float(duration)
            if pos[2] < 0.05:
                print('below threshold!!', pos[2])
                continue
            try:
                self.fa.goto_pose(rt,
                        duration=duration, 
                        force_thresholds=[10,10,10,10,10,10],
                        ignore_virtual_walls=True,
                        buffer_time=0.0
                )    
            except Exception as e:
                print('Could not goto_pose', e)
            abs_diff = sum((self.fa.get_pose().translation-rt.translation)**2)**0.5 * 100
            # print(abs_diff, 'cm stable')
            abs_diffs.append(abs_diff)
        # print(max(abs_diffs), sum(abs_diffs)/len(abs_diffs), '\t', time.time() - start)
        if abs_diffs[-1] > 1:
            # Didn't get to the last position. Try again.
            print('Off from final position by', abs_diffs[-1], 'cm')
            self.fa.goto_pose(rt,
                        duration=duration+3, 
                        force_thresholds=[10,10,10,10,10,10],
                        ignore_virtual_walls=True,
                        buffer_time=0.0
                )
            abs_diff = sum((self.fa.get_pose().translation-rt.translation)**2)**0.5 * 100
            if abs_diff > 1:
                print('Failed to get to end of trajectory again. Resetting Joints')
                self.fa.reset_joints()

    def go_to_cartesian_pose_precise(self, positions, orientations, hertz=300, stiffness_factor=3.0):
        """
            This is a smooth version of this function. It can very smoothly go betwen the positions.
            However, it is unstable, and will result in oscilations sometimes.
            Recommended to be used only for fine, slow motions like the actual brush strokes.
        """
        start = time.time()
        from frankapy import SensorDataMessageType
        from frankapy import FrankaConstants as FC
        from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
        from frankapy.proto import PosePositionSensorMessage, CartesianImpedanceSensorMessage
        from franka_interface_msgs.msg import SensorDataGroup
        from frankapy.utils import min_jerk, min_jerk_weight
        import rospy
        
        def get_duration(here, there):
            dist = ((here.translation - there.translation)**2).sum()**.5
            duration = dist *  10#5 # 1cm=.1s 1m=10s
            duration = max(0.4, duration) # Don't go toooo fast
            duration = float(duration)
            return duration, dist

        def smooth_trajectory(poses, window_width=50):
            # x = np.cumsum(delta_ts)
            from scipy.interpolate import interp1d
            from scipy.ndimage import gaussian_filter1d

            for c in range(3):
                coords = np.array([p.translation[c] for p in poses])

                coords_smooth = gaussian_filter1d(coords, 31)
                print(len(poses), len(coords_smooth))
                for i in range(len(poses)-1):
                    coords_smooth[i]
                    poses[i].translation[c] = coords_smooth[i]
            return poses

        pose_trajs = []
        delta_ts = []
        
        # Loop through each position/orientation and create interpolations between the points
        p0 = self.fa.get_pose()
        for i in range(len(positions)):
            p1 = Franka.create_rotation_transform(\
                Franka.sawyer_to_franka_position(positions[i]), orientations[i])

            duration, distance = get_duration(p0, p1)

            # needs to be high to avoid torque discontinuity error controller_torque_discontinuity
            STEPS = max(10, int(duration*hertz))
            # print(STEPS, distance)

            # if distance*100 > 5:
            #     print("You're using the precise movement wrong", distance*100)

            ts = np.arange(0, duration, duration/STEPS)
            # ts = np.linspace(0, duration, STEPS)
            weights = [min_jerk_weight(t, duration) for t in ts]

            if i == 0 or i == len(positions)-1:
                # Smooth for the first and last way points
                pose_traj = [p0.interpolate_with(p1, w) for w in weights]
            else:
                # linear for middle points cuz it's fast and accurate
                pose_traj = p0.linear_trajectory_to(p1, STEPS)
            # pose_traj = [p0.interpolate_with(p1, w) for w in weights]
            # pose_traj = p0.linear_trajectory_to(p1, STEPS)

            pose_trajs += pose_traj
            
            delta_ts += [duration/len(pose_traj),]*len(pose_traj)
            
            p0 = p1
            
        T = float(np.array(delta_ts).sum())

        # pose_trajs = smooth_trajectory(pose_trajs)

        pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        
        # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
        self.fa.goto_pose(pose_trajs[1], duration=T, dynamic=True, 
            buffer_time=T+10,
            force_thresholds=[10,10,10,10,10,10],
            cartesian_impedances=(np.array(FC.DEFAULT_TRANSLATIONAL_STIFFNESSES)*stiffness_factor).tolist() + FC.DEFAULT_ROTATIONAL_STIFFNESSES,
            ignore_virtual_walls=True,
        )
        abs_diffs = []
        try:
            init_time = rospy.Time.now().to_time()
            for i in range(2, len(pose_trajs)):
                timestamp = rospy.Time.now().to_time() - init_time
                traj_gen_proto_msg = PosePositionSensorMessage(
                    id=i, timestamp=timestamp, 
                    position=pose_trajs[i].translation, quaternion=pose_trajs[i].quaternion
                )
                fb_ctrlr_proto = CartesianImpedanceSensorMessage(
                    id=i, timestamp=timestamp,
                    # translational_stiffnesses=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES[:2] + [z_stiffness_trajs[i]],
                    translational_stiffnesses=(np.array(FC.DEFAULT_TRANSLATIONAL_STIFFNESSES)*stiffness_factor).tolist(),
                    rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES
                )
                
                ros_msg = make_sensor_group_msg(
                    trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                        traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
                    feedback_controller_sensor_msg=sensor_proto2ros_msg(
                        fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
                )

                # rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
                pub.publish(ros_msg)
                # rate = rospy.Rate(1 / (delta_ts[i]))
                rate = rospy.Rate(hertz)
                rate.sleep()

                # if i%100==0:
                #     print(self.fa.get_pose().translation[-1] - pose_trajs[i].translation[-1], 
                #         '\t', self.fa.get_pose().translation[-1], '\t', pose_trajs[i].translation[-1])
                if i%10 == 0:
                    abs_diff = sum((self.fa.get_pose().translation-pose_trajs[i].translation)**2)**0.5 * 100
                    # print(abs_diff, 'cm')
                    abs_diffs.append(abs_diff)
                    # print(self.fa.get_pose().translation[-1] - pose_trajs[i].translation[-1], 
                    #     '\t', self.fa.get_pose().translation[-1], '\t', pose_trajs[i].translation[-1])
        except Exception as e:
            print('unable to execute skill', e)
        print(max(abs_diffs), sum(abs_diffs)/len(abs_diffs), '\t', time.time() - start)
        # Stop the skill
        self.fa.stop_skill()
        
class SimulatedRobot(Robot, object):
    def __init__(self, debug=True):
        pass

    def good_morning_robot(self):
        pass

    def good_night_robot(self):
        pass

    def go_to_cartesian_pose(self, position, orientation, move_by_joint=False):
        pass


class SimulatedTrajectoryRecordRobot(Robot, object):
    def __init__(self, debug=True):
        super(SimulatedTrajectoryRecordRobot, self).__init__(debug)
        self.current_position = None
        self.current_orientation = None
        self.video_dir = os.path.join(os.getcwd(), "trajectory")
        self.video_file = os.path.join(self.video_dir, "robot_trajectory.mp4")
        self.height, self.width = 1080, 1920
        self.touch_depth = 100  # 贴近纸面时的高度，代表基础高度
        self.press_depth = 2  # 按压时的最大高度，代表毛笔从贴近纸面到按压时的最大深度
        self.canvas = None
        self.out = None
        self.debug("SimulatedTrajectoryRecordRobot initialized")

    def setSize(self, opts):
        self.width = opts.CANVAS_WIDTH_M * opts.render_width
        self.height = opts.CANVAS_HEIGHT_M  * opts.render_height

    def good_morning_robot(self):
        """初始化机器人并开始记录轨迹"""
        pass

    def good_night_robot(self):
        """关闭机器人并完成轨迹记录"""
        # 如果没有轨迹点或视频写入器，直接返回
        self.debug(f"Trajectory recording completed, saved to {self.video_file}")
        return True

    def go_to_cartesian_pose(self, positions, orientations, move_by_joint=False, precise=False, speed=80):
        """模拟机器人移动到指定位置并记录轨迹
        每次调用此方法时，会实时记录轨迹点并绘制线条
        """
        positions, orientations = np.array(positions), np.array(orientations)
        # 确保positions是二维数组，即使只有一个位置点
        if len(positions.shape) == 1:
            positions = positions.reshape(1, -1)
            orientations = orientations.reshape(1, -1) if len(orientations.shape) == 1 else orientations
            
        logger.info("positions shape {}", positions.shape)

        # 处理新的轨迹点
        for i, item in enumerate(positions):
            # 将位置添加到轨迹中 (x, y, z)
            logger.info("item:{}", item)
        time.sleep(0.05)


        return True


class SimulatedMyCobot(Robot, object):
    '''
        使用MuJoCo实现的MyCobot 280机械臂仿真类
    '''
    def __init__(self, debug=True, node_name="painting"):
        super(SimulatedMyCobot, self).__init__(debug, node_name)
        # import rospy
        # logger.info("rospy.init_node start")
        # rospy.init_node(node_name)
        # logger.info("rospy.init_node success")

        import mujoco
        import mujoco.viewer

        self.model = mujoco.MjModel.from_xml_path(r'D:\code\frida\ROS\URDF\universal_robots_ur5e\scene.xml')
        # 加载URDF到MuJoCo模型
        self.data = mujoco.MjData(self.model)
        
        # 初始化关节名称映射
        self.joint_names = [
            "shoulder_pan_joint",    # 底座旋转关节
            "shoulder_lift_joint",   # 肩部关节
            "elbow_joint",          # 肘部关节
            "wrist_1_joint",        # 腕部第一关节
            "wrist_2_joint",        # 腕部第二关节
            "wrist_3_joint"         # 腕部第三关节
        ]
        
        # 获取关节ID
        self.joint_ids = [self.model.joint(name).id for name in self.joint_names]
        
        # 初始化查看器
        self.viewer = None
        
        # 初始化机械臂位置
        self.reset_position()
    
    def reset_position(self):
        """重置机械臂到初始位置"""
        # 设置初始关节角度为0
        for i, joint_id in enumerate(self.joint_ids):
            self.data.qpos[joint_id] = 0.0
        
        # 更新模型
        mujoco.mj_forward(self.model, self.data)

    def good_morning_robot(self):
        import mujoco.viewer
        """初始化机械臂和查看器"""
        self.debug("Initializing MuJoCo simulation for MyCobot 280")
        
        # 重置机械臂位置
        self.reset_position()
        
        # 创建查看器
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 1.0  # 设置相机距离
            self.viewer.cam.azimuth = 90    # 设置相机方位角
            self.viewer.cam.elevation = -20  # 设置相机仰角

    def good_night_robot(self):
        """关闭查看器和仿真"""
        self.debug("Shutting down MuJoCo simulation")
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def moveit_go_to_cartesian_pose(self, positions, orientations):
        moveit_commander.roscpp_initialize(sys.argv)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        group = moveit_commander.MoveGroupCommander("arm")

        for i in range(len(positions)):
            pose_goal = group.get_current_pose().pose
            pose_goal.position.x = positions[i][0]
            pose_goal.position.y = positions[i][1]
            pose_goal.position.z = positions[i][2]
            pose_goal.orientation.x = orientations[i][0]
            pose_goal.orientation.y = orientations[i][1]
            pose_goal.orientation.z = orientations[i][2]
            pose_goal.orientation.w = orientations[i][3]

            group.set_pose_target(pose_goal)
            plan = group.go(wait=True)
            group.stop()
            group.clear_pose_targets()

        moveit_commander.roscpp_shutdown()

    def inverse_kinematics(self, position, orientation):
        """简化版逆运动学求解，将笛卡尔坐标转换为关节角度
        
        Args:
            position: [x, y, z] 位置坐标 (米)
            orientation: [x, y, z, w] 四元数方向
            
        Returns:
            关节角度列表
        """
        # 这里使用简化的逆运动学算法
        # 实际应用中应该使用更复杂的IK求解器
        
        # 将位置从米转换为模型单位
        x, y, z = position[0], position[1], position[2]
        
        # 简化的IK计算 (这只是一个示例，实际应用需要更准确的IK)
        # 在实际应用中，应该使用MuJoCo的内置IK或其他IK库
        
        # 计算到目标的距离
        distance = np.sqrt(x**2 + y**2 + z**2)
        
        # 计算基座旋转角度 (绕Z轴)
        base_angle = np.arctan2(y, x)
        
        # 简化的手臂角度计算
        arm_angle = np.arcsin(z / distance) if distance > 0 else 0
        
        # 计算肘部角度
        elbow_angle = np.pi/4  # 45度，简化计算
        
        # 从四元数计算欧拉角
        roll, pitch, yaw = euler_from_quaternion(orientation[0], orientation[1], orientation[2], orientation[3])
        
        # 返回关节角度
        joint_angles = [
            base_angle,           # 底座旋转
            arm_angle,            # 肩部关节
            elbow_angle,          # 肘部关节
            0.0,                  # 前臂关节
            pitch,                # 腕部俯仰
            roll                  # 末端执行器旋转
        ]
        
        return joint_angles

    def go_to_cartesian_pose(self, positions, orientations, precise=False, move_by_joint=False, speed=80):
        """移动机械臂到指定的笛卡尔坐标
        
        Args:
            positions: 位置坐标列表 [x, y, z] (米)
            orientations: 方向四元数列表 [x, y, z, w]
            precise: 是否使用精确模式 (未实现)
            move_by_joint: 是否使用关节空间移动 (未实现)
            speed: 移动速度 (未实现)
        """
        positions, orientations = np.array(positions), np.array(orientations)
        if len(positions.shape) == 1:
            positions = positions[None,:]
            orientations = orientations[None,:]
        
        self.debug(f"Moving to {len(positions)} positions")
        
        # 确保查看器已初始化
        if self.viewer is None:
            self.good_morning_robot()
        
        # 对每个位置执行移动
        for i in range(len(positions)):
            # 计算关节角度
            joint_angles = self.inverse_kinematics(positions[i], orientations[i])
            
            # 设置关节角度
            for j, joint_id in enumerate(self.joint_ids):
                self.data.qpos[joint_id] = joint_angles[j]
            
            # 更新模型
            mujoco.mj_forward(self.model, self.data)
            
            # 更新查看器
            if self.viewer is not None:
                self.viewer.sync()
                time.sleep(0.05)  # 添加延迟以便观察运动
        
        self.debug("Movement completed")
        return True



class Sawyer(Robot, object):
    def __init__(self, debug=True):
        super(Sawyer, self).__init__(debug)
        import rospy


        from intera_core_msgs.srv import (
            SolvePositionIK,
            SolvePositionIKRequest,
            SolvePositionFK,
            SolvePositionFKRequest,
        )
        import intera_interface
        from intera_interface import CHECK_VERSION
        import PyKDL
        from tf_conversions import posemath

        self.limb = intera_interface.Limb(synchronous_pub=False)
        # print(self.limb)

        self.ns = "ExternalTools/right/PositionKinematicsNode/IKService"
        self.iksvc = rospy.ServiceProxy(self.ns, SolvePositionIK, persistent=True)
        rospy.wait_for_service(self.ns, 5.0)

    def good_morning_robot(self):
        import intera_interface
        import rospy
        self.debug("Getting robot state... ")
        rs = intera_interface.RobotEnable(False)
        init_state = rs.state().enabled
        self.debug("Enabling robot... ")
        rs.enable()

        def clean_shutdown():
            """
            Exits example cleanly by moving head to neutral position and
            maintaining start state
            """
            self.debug("\nExiting example...")
            limb = intera_interface.Limb(synchronous_pub=True)
            limb.move_to_neutral(speed=.2)
            # 1/0

        rospy.on_shutdown(clean_shutdown)
        self.debug("Excecuting... ")

        # neutral_pose = rospy.get_param("named_poses/{0}/poses/neutral".format(self.name))
        # angles = dict(list(zip(self.joint_names(), neutral_pose)))
        # self.set_joint_position_speed(0.1)
        # self.move_to_joint_positions(angles, timeout)
        intera_interface.Limb(synchronous_pub=True).move_to_neutral(speed=.2)
        
        return rs

    def good_night_robot(self):
        import rospy
        """ Tuck it in, read it a story """
        rospy.signal_shutdown("Example finished.")
        self.debug("Done")

    def go_to_cartesian_pose(self, position, orientation):
        #if len(position)
        position, orientation = np.array(position), np.array(orientation)
        if len(position.shape) == 1:
            position = position[None,:]
            orientation = orientation[None,:]

        # import rospy
        # import argparse
        from intera_motion_interface import (
            MotionTrajectory,
            MotionWaypoint,
            MotionWaypointOptions
        )
        from intera_motion_msgs.msg import TrajectoryOptions
        from geometry_msgs.msg import PoseStamped
        import PyKDL
        # from tf_conversions import posemath
        # from intera_interface import Limb

        limb = self.limb#Limb()

        traj_options = TrajectoryOptions()
        traj_options.interpolation_type = TrajectoryOptions.CARTESIAN
        traj = MotionTrajectory(trajectory_options = traj_options, limb = limb)

        wpt_opts = MotionWaypointOptions(max_linear_speed=0.8*1.5,
                                         max_linear_accel=0.8*1.5,
                                         # joint_tolerances=0.05,
                                         corner_distance=0.005,
                                         max_rotational_speed=1.57,
                                         max_rotational_accel=1.57,
                                         max_joint_speed_ratio=1.0)

        for i in range(len(position)):
            waypoint = MotionWaypoint(options = wpt_opts.to_msg(), limb = limb)

            joint_names = limb.joint_names()

            endpoint_state = limb.tip_state("right_hand")
            pose = endpoint_state.pose

            pose.position.x = position[i,0]
            pose.position.y = position[i,1]
            pose.position.z = position[i,2]

            pose.orientation.x = orientation[i,0]
            pose.orientation.y = orientation[i,1]
            pose.orientation.z = orientation[i,2]
            pose.orientation.w = orientation[i,3]

            poseStamped = PoseStamped()
            poseStamped.pose = pose

            joint_angles = limb.joint_ordered_angles()
            waypoint.set_cartesian_pose(poseStamped, "right_hand", joint_angles)

            traj.append_waypoint(waypoint.to_msg())

        result = traj.send_trajectory(timeout=None)
        # print(result.result)
        success = result.result
        # if success != True:
        #     print(success)
        if not success:
            import time
            # print('sleeping')
            time.sleep(2)
            # print('done sleeping. now to neutral')
            # Go to neutral and try again
            limb.move_to_neutral(speed=.3)
            # print('done to neutral')
            result = traj.send_trajectory(timeout=None)
            # print('just tried to resend trajectory')
            if result.result:
                print('second attempt successful')
            else:
                print('failed second attempt')
            success = result.result
        return success

    def move_to_joint_positions(self, position, timeout=3, speed=0.1):
        """
        args:
            dict{'right_j0',float} - dictionary of joint to joint angle
        """
        # rate = rospy.Rate(100)
        #try:
        # print('Positions:', position)
        self.limb.set_joint_position_speed(speed=speed)
        self.limb.move_to_joint_positions(position, timeout=timeout,
                                     threshold=0.008726646)
        self.limb.set_joint_position_speed(speed=.1)
        # rate.sleep()
        # except Exception as e:
        #     print('Exception while moving robot:\n', e)
        #     import traceback
        #     import sys
        #     print(traceback.format_exc())


    def display_image(self, file_path):
        import intera_interface
        head_display = intera_interface.HeadDisplay()
        # display_image params:
        # 1. file Path to image file to send. Multiple files are separated by a space, eg.: a.png b.png
        # 2. loop Display images in loop, add argument will display images in loop
        # 3. rate Image display frequency for multiple and looped images.
        head_display.display_image(file_path, False, 100)
    def display_frida(self):
        import rospkg
        rospack = rospkg.RosPack()
        # get the file path for rospy_tutorials
        ros_dir = rospack.get_path('paint')
        self.display_image(os.path.join(str(ros_dir), 'src', 'frida.jpg'))

    def take_picture(self):
        import cv2
        from cv_bridge import CvBridge, CvBridgeError
        import matplotlib.pyplot as plt
        def show_image_callback(img_data):
            """The callback function to show image by using CvBridge and cv
            """
            bridge = CvBridge()
            try:
                cv_image = bridge.imgmsg_to_cv2(img_data, "bgr8")
            except CvBridgeError as err:
                rospy.logerr(err)
                return

            # edge_str = ''
            # cv_win_name = ' '.join(['heyyyy', edge_str])
            # cv2.namedWindow(cv_win_name, 0)
            # refresh the image on the screen
            # cv2.imshow(cv_win_name, cv_image)
            # cv2.waitKey(3)
            plt.imshow(cv_image[:,:,::-1])
            plt.show()
        rp = intera_interface.RobotParams()
        valid_cameras = rp.get_camera_names()
        print('valid_cameras', valid_cameras)

        camera = 'head_camera'
        # camera = 'right_hand_camera'
        cameras = intera_interface.Cameras()
        if not cameras.verify_camera_exists(camera):
            rospy.logerr("Could not detect the specified camera, exiting the example.")
            return
        rospy.loginfo("Opening camera '{0}'...".format(camera))
        cameras.start_streaming(camera)
        cameras.set_callback(camera, show_image_callback,
            rectify_image=False)
        raw_input('Attach the paint brush now. Press enter to continue:')


class UltraArm340(Robot):
    '''
        UltraArm 机器人的低级动作功能实现。
        Low-level action functionality of the UltraArm robot.
    '''

    def __init__(self, debug=False):
        """
        初始化 UltraArm 机器人。
        该函数用于建立与机械臂的socket连接
        host: 机械臂的IP地址
        port: 机械臂的端口号
        """
        self.debug_bool = debug
        self.cobot = self.init_arm()

    def init_cobot(self, host, port):
        """
        初始化机械臂,
        过程中灯会闪烁, 提示机械臂初始化完成
        """
        from pymycobot import MyCobotSocket
        # 默认使用端口 9000
        return MyCobotSocket(host, port)

    def find_ch340_device(self):
        import serial
        import serial.tools.list_ports
        """Find CH340 device among connected COM ports."""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            # print(f"Checking port: {port.device}, description: {port.description}")
            if 'CH340' in port.description:
                # print(f"CH340 device found on {port.device}")
                return port.device
        print("No CH340 device found.")
        return None

    def create_uarm_object(self, ch340_port, arm_type="uArm"):
        if arm_type == "uArm":
            from pymycobot import ultraArm
            ua = ultraArm(ch340_port, 115200)
        else:
            from pymycobot import MyCobotSocket
            ua = MyCobotSocket('192.168.31.7', 9000)
        return ua

    def disable_enable_com_port(self, key_words: str):
        """使用 pnputil 禁用并重新启用指定的 COM 端口。"""
        try:
            # 获取设备实例 ID，使用pnputil列出所有COM端口
            list_devices_command = f'powershell "pnputil /enum-devices /connected | Select-String -Pattern \'{key_words}\' -Context 1,5"'
            logger.info(list_devices_command)
            list_result = subprocess.run(list_devices_command, shell=True, capture_output=True, text=True)
            if list_result.returncode != 0:
                logger.error(f"无法获取设备列表，错误信息：{list_result.stderr}")
                return

            # 解析设备列表，找到对应的设备实例 ID
            device_id = None
            lines = list_result.stdout.splitlines()
            for i, line in enumerate(lines):
                logger.info(line)
                if line.__contains__("Instance ID:") or line.__contains__("实例 ID"):
                    device_id = line.split(":", 1)[1].strip()
                    logger.info(f"找到与 {key_words} 对应的设备实例 ID: {device_id}")
                    break

            if not device_id:
                logger.error(f"未找到与 {key_words} 对应的设备实例 ID")
                return

            # 禁用设备
            disable_command = f"pnputil /disable-device \"{device_id}\""
            disable_result = subprocess.run(disable_command, shell=True, capture_output=True, text=True)
            if disable_result.returncode == 0:
                logger.info(f"{key_words} 已禁用。")
            else:
                logger.error(f"无法禁用 {key_words}，错误信息：{disable_result.stderr}")
                return
            time.sleep(2)
            # 启用设备
            enable_command = f"pnputil /enable-device \"{device_id}\""
            enable_result = subprocess.run(enable_command, shell=True, capture_output=True, text=True)
            if enable_result.returncode == 0:
                logger.info(f"{key_words} 已重新启用。")
            else:
                logger.error(f"无法重新启用 {key_words}，错误信息：{enable_result.stderr}")

        except Exception as e:
            logger.error(f"禁用/启用 {key_words} 时发生错误: {e}")

    def free_com_port(self, key_words="CH340"):
        try:
            self.disable_enable_com_port(key_words)
        except Exception as e:
            logger.error(f"释放端口 {key_words} 时发生错误: {e}")

    def init_arm(self, arm_type="uArm"):
        import serial.tools.list_ports
        ua = None
        ch340_port = self.find_ch340_device()
        try:
            if ch340_port:
                ua = self.create_uarm_object(ch340_port, arm_type)
                logger.info("成功初始化机械臂 {}", ch340_port)
            else:
                logger.error("CH340 device not found.")
        except serial.SerialException as e:
            if "PermissionError" in str(e) or "连到系统上的设备没有发挥作用" in str(e):
                self.free_com_port()
                try:
                    ua = self.create_uarm_object(ch340_port, arm_type)
                    logger.info("成功初始化机械臂 {}", ch340_port)
                except Exception as e:
                    logger.error("Failed to initialize ultraArm after freeing port: {}", e)
                    return None
            else:
                logger.error(f"Error accessing {ch340_port}: {e}")
                return None
        finally:
            pass

        return ua

    def debug(self, msg):
        if self.debug_bool:
            print(msg)

    def reset_joints(self):
        """
        重置 UltraArm 到其初始关节位置。
        """
        self.cobot.go_zero()
        # coords = ua.get_coords_info()
        # ua.sleep(2)
        # logger.info("初始点位 {}", coords)

        # 以下是280-Pi版本的代码
        # self.cobot.power_on()
        # self.cobot.set_fresh_mode(0)
        # self.cobot.set_color(0,0,255) #蓝灯亮
        # logger.info("上电所有舵机from 1 to 6 focus all servos")
        # for i in range(1, 7):
        #     self.cobot.focus_servo(i)
        # # 打开自由移动模式
        # self.cobot.set_free_mode(1)
        # time.sleep(1)    #等2秒
        # self.cobot.set_color(255,0,0) #红灯亮
        # # 打开夹爪
        # self.cobot.set_gripper_state(0, 70)
        # time.sleep(2)    #等2秒
        # # 关闭自由移动模式
        # self.cobot.set_free_mode(0)
        # self.cobot.sync_send_angles([0, 0, 0, 0, 0, -45], 20)
        # # 初始化完成, 绿灯亮
        # self.cobot.set_color(0,255,0)

    def grab_pen(self):
        self.cobot.send_angles([0, -90, 0, 0, 0, -45], 50)
        # grab pen
        # 打开夹爪
        self.cobot.set_gripper_state(0, 70)
        time.sleep(5)
        # 关闭夹爪
        self.cobot.set_gripper_state(1, 70)

    def good_morning_robot(self):
        """
        启动 UltraArm 机器人并准备好执行任务。
        """
        self.debug("启动 UltraArm 机器人...")
        self.reset_joints()

    def good_night_robot(self):
        """
        关闭 UltraArm 机器人并断开连接。
        """
        self.debug("关闭 UltraArm 机器人...")

    def go_to_cartesian_pose(self, positions, orientations, move_by_joint=False, speed=50):
        """
        将机器人移动到指定的笛卡尔坐标位置和方向。
        uarm的坐标系是yxz
        参数:
            positions (np.array(n,3)) : 目标��置的数组，包含n个点的y, x, z坐标（单位：米）。
            orientations (np.array(n,4)) : 目标方向的数组，包含n个点的四元数（x, y, z, w）。
            speed (int) : 移动速度，默认为50（单位：mm/s）。

        示例输入:
            positions = np.array([[0.5, 0.2, 0.1], [0.6, 0.3, 0.2]])  # 两个目标位置
            orientations = np.array([[0, 0, 0, 1], [0, 0, 0.7071, 0.7071]])  # 两个目标方向（四元数）

        示例返回值:
            None  # 此函数没有返回值，机器人将直接移动到指定位置和方向。
        """
        # logger.debug("positions: {}, orientations: {}".format(positions, orientations))
        positions, orientations = np.array(positions), np.array(orientations)
        if len(positions.shape) == 1:
            positions = positions[None, :]
            orientations = orientations[None, :]

        # 移动到指定位置
        # self.cobot.send_coords
        start = time.time()
        for i, item in enumerate(positions):
            # sync_send_coords(coords, speed, mode)
            # [x,y,z]表示的是机械臂头部在空间中的位置（该坐标系为直角坐标系），[rx,ry,rz]表示的是机械臂头部在该点的姿态（该坐标系为欧拉坐标）
            # [x,y,z,rx,ry,rz]��坐标值，长度为6, x,y,z的范围为-280mm ~ 280mm，rx,ry,yz的范围为-314  ~ 314

            formated_coord = [item[1] * 1000, item[0] * 1000, item[2] * 1000]
            # logger.debug("formated_item is {}", formated_coord)
            if formated_coord[0] < 150 or formated_coord[0] > 350 or formated_coord[1] < -200 or formated_coord[
                1] > 200:
                logger.error("机械臂超出画布范围，机械臂坐标为：{}", formated_coord)
                continue
            self.move_arm_step(formated_coord, speed)
            time.sleep(0.01)
        start = time.time() - start
        # logger.debug("移动机械臂, 总步数为：{}, 机械臂绘耗时为：{}", len(positions), start)

    def move_arm_step(self, coord, speed=50):
        try:
            self.cobot.set_coords(coord, speed)
        except Exception as e:
            logger.error(e)
        return True

    def move_to_joint_positions(self, joint_angles, speed=100):
        """
        移动 UltraArm 到指定的关节角度。

        :param joint_angles: 关节角度列表
        :param speed: 移动速度
        """
        self.cobot.send_angles(joint_angles, speed)

    def display_image(self, file_path):
        """
        显示指定路径的图像（如果有显示功能）。

        :param file_path: 图像文件路径
        """
        # 这里可以实现图像显示的功能
        pass

    def take_picture(self):
        """
        捕获图像（如果有相机功能）。
        """
        # 这里可以实现图像捕获的功能
        pass


class Cobot280(Robot):
    '''
        UltraArm 机器人的低级动作功能实现。
        Low-level action functionality of the UltraArm robot.
    '''

    def __init__(self, host="192.168.31.7", port=9000, debug=False):
        """
        初始化 UltraArm 机器人。
        该函数用于建立与机械臂的socket连接
        host: 机械臂的IP地址
        port: 机械臂的端口号
        """
        self.debug_bool = debug
        self.cobot = self.init_cobot(host, port)
        self.current_position = None

    def init_cobot(self, host, port):
        """
        初始化机械臂,
        过程中灯会闪烁, 提示机械臂初始化完成
        """
        from pymycobot import MyCobotSocket
        # 默认使用端口 9000
        return MyCobotSocket(host, port)

    def debug(self, msg):
        if self.debug_bool:
            print(msg)

    def reset_joints(self):
        """
        重置 UltraArm 到其初始关节位置。
        """
        self.cobot.set_color(0, 0, 255)  # 蓝灯亮
        from pymycobot.common import ProtocolCode
        self.cobot._mesg(ProtocolCode.SET_FRESH_MODE, 0)
        self.cobot._mesg(ProtocolCode.SET_FREE_MODE, 1)
        time.sleep(1)  # 等2秒
        self.cobot.set_color(255, 0, 0)  # 红灯亮
        # self.cobot.sync_send_angles([0, 0, 0, 0, 0, -45], 10)
        # self.cobot.sync_send_coords([0.0, 170.0, 125.0, 180.0, -0.0, -45.0], 10, 0)
        self.cobot.set_color(0, 255, 0)

    def move_to_position(self, positions, speed=50):
        self.cobot.send_coords(positions, speed, 0)

    def get_position(self):
        return self.cobot.get_coords()

    def grab_pen(self):
        # self.cobot.send_angles([0, -90, 0, 0, 0, -45], 50)
        # grab pen
        # 打开夹爪
        self.cobot.set_gripper_state(0, 70)
        time.sleep(5)
        # 关闭夹爪
        self.cobot.set_gripper_state(1, 70)

    def good_morning_robot(self):
        """
        启动 MyCobot280 机器人并准备好执行任务。
        """
        self.debug("启动 MyCobot280 机器人...")
        # 重置 UltraArm 到其初始关节位置
        self.reset_joints()

    def good_night_robot(self):
        """
        关闭 UltraArm 机器人并断开连接。
        """
        self.debug("关闭 MyCobot280 机器人...")
        self.cobot.stop()
        # self.cobot.power_off()

    def go_to_cartesian_pose(self, positions, orientations, move_by_joint=False, speed=80):
        """
        将机器人移动到指定的笛卡尔坐标位置和方向。

        参数:
            positions (np.array(n,3)) : 目标��置的数组，包含n个点的x, y, z坐标（单位：米）。
            orientations (np.array(n,4)) : 目标方向的数组，包含n个点的四元数（x, y, z, w）。
            speed (int) : 移动速度，默认为50（单位：mm/s）。

        示例输入:
            positions = np.array([[0.5, 0.2, 0.1], [0.6, 0.3, 0.2]])  # 两个目标位置
            orientations = np.array([[0, 0, 0, 1], [0, 0, 0.7071, 0.7071]])  # 两个目标方向（四元数）

        示例返回值:
            None  # 此函数没有返回值，机器人将直接移动到指定位置和方向。
        """
        positions, orientations = np.array(positions), np.array(orientations)
        if len(positions.shape) == 1:
            positions = positions[None, :]
            orientations = orientations[None, :]

        while self.current_position is None or self.current_position == -1:
            self.current_position = self.cobot.get_coords()
            logger.info(f"current_position is {self.current_position}")
            time.sleep(1)

        if len(positions) > 0:
            logger.info("current_position is {}", self.current_position)
            item = positions[0]
            converted_angles = quaternion_to_euler_degrees(orientations[0])
            # converted_angles[-1] = converted_angles[-1] + 60
            converted_angles[-1] = 145
            formated_item = [item[1] * 1000, item[0] * 1000, item[2] * 1000, *converted_angles]
            y0 = self.current_position[1]
            y1 = formated_item[1]
            logger.info("y0 - y1 = {}, move_by_joint: {}", y0 - y1, move_by_joint)
            if self.current_position is not None and (y0 * y1 < 0 or abs(y0 - y1) > 190):
                # move to safe position
                # self.cobot.sync_send_angles([0, -45, 0, 0, 0, -45], 20)
                middle_point = [233.8, -65.5, 200, -146.72, -29.68, -53.94]
                self.cobot.send_coords(middle_point, 50, 1)
                self.current_position = middle_point

        for i, item in enumerate(positions):
            # sync_send_coords(coords, speed, mode)
            # item的格式为 [x,y,z,rx,ry,rz]
            # [x,y,z]表示的是机械臂头部在空间中的位置（该坐标系为直角坐标系），
            # [rx,ry,rz]表示的是机械臂头部在该点的姿态（该坐标系为欧拉坐标）
            # x	0 ~ 281.45
            # y	-281.45 ~ 281.45
            # z	-70 ~ 412.67
            # 180, 0, -45 为 rx ry rz 的默认值  当 x>220 rz -45  当 x<220 rz 145
            converted_angles = quaternion_to_euler_degrees(orientations[i])
            # converted_angles[-1] = converted_angles[-1] + 60
            converted_angles[-1] = 145
            formated_item = [item[1] * 1000, item[0] * 1000, item[2] * 1000, *converted_angles]
            # if not self.safe_position_check(formated_item):
            #     logger.warning("机械臂超出安全范围，机械臂坐标为：{}", formated_item)
            logger.info(f"formated_item is {[int(x) for x in formated_item]}")
            # get current pos , if move across y from positive to negative or negative to positive, move to the safe position first
            self.current_position = formated_item
            if move_by_joint:
                # 0-非线性（默认），1-直线运动
                if not self.safe_position_check(formated_item):
                    logger.warning("机械臂超出安全范围，机械臂坐标为：{}", formated_item)
                self.cobot.send_coords(formated_item, speed, 0)
            else:

                self.cobot.send_coords(formated_item, speed, 1)
            self.current_position = formated_item

            # print(self.cobot.get_angles())

    def move_to_joint_positions(self, joint_angles, speed=100):
        """
        移动 UltraArm 到指定的关节角度。

        :param joint_angles: 关节角度列表
        :param speed: 移动速度
        """
        self.cobot.send_angles(joint_angles, speed)

    def safe_position_check(self, item):
        """
        item: [x, y, z, rx, ry, rz]
        检查item的xyz构成的臂展是否在280mm范围内
        z的范围不能低于10mm
        """
        if  item[1] < -180:
            # 绘画中不能触碰墨盒
            # 画布本身限制
            return False
        # 机械臂限制

        return True


def quaternion_to_euler_degrees(quaternion):
    """
    将四元数转换为角度制欧拉角，并保留小数点后一位。

    参数:
        quaternion (list or np.array): 四元数 [w, x, y, z]

    返回:
        list: 角度制欧拉角 [rx, ry, rz]，每个值保留小数点后1位
    """
    # 四元数转换为弧度制欧拉角
    w, x, y, z = quaternion
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    rx = np.arctan2(t0, t1)  # 绕 X 轴旋转

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)  # 防止超出 [-1, 1]
    ry = np.arcsin(t2)  # 绕 Y 轴旋转

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    rz = np.arctan2(t3, t4)  # 绕 Z 轴旋转

    # 转为角度制
    euler_angles_radians = [rx, ry, rz]
    euler_angles_degrees = np.degrees(euler_angles_radians)

    # 保留小数点后 1 位
    euler_angles_degrees = np.round(euler_angles_degrees, 1)

    return euler_angles_degrees


def convert_quaternion_to_euler(quaternion):
    """
    将四元数转换为欧拉角
    """
    w, x, y, z = quaternion
    # 计算欧拉角（roll, pitch, yaw）
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(2.0 * (w * y - z * x))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    return np.array([roll, pitch, yaw])


if __name__ == "__main__":
    mc = SimulatedMyCobot()
    mc.good_morning_robot()

    mc.good_night_robot()

