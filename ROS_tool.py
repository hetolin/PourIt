'''
# -*- coding: utf-8 -*-
# @Project   : robopose
# @File      : ROS_tool.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2021/7/7 12:37

# @Desciption: 
'''
import numpy as np
import cv2
from copy import deepcopy

## ROS ##
from cv_bridge import (
    CvBridgeError,
)

# ros
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo, JointState
import tf2_ros as tf2
import tf
import message_filters

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CAMERA_TYPE = 'realsense'
# CAMERA_TYPE = 'kinect_azure'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class rosImageReceiver():
    def __init__(self, robot_name='panda', cam_type='kinect_azure', sync=True):
        rospy.init_node("dream", anonymous=True)
        assert robot_name in ['panda', 'baxter'], '{} is not right, should be [baxter, panda]'
        assert cam_type in ['realsense', 'kinect_azure'], '{} is not right, should be [realsense, kinect_azure]'
        self.robot_name = robot_name
        self.cam_type = cam_type
        self.sync = sync

        self.config_topics()
        self.camera_initialization()
        self.publisher_initialization()
        self.robot_joint_names_initialization()

    def config_topics(self):
        self.joint_states_topic = '/joint_states'

        if self.cam_type == 'realsense':
            '''realsense topic'''
            self.image_topic = "/camera/color/image_raw"
            self.image_depth_topic =  "/camera/aligned_depth_to_color/image_raw"
            self.camera_info_topic = "/camera/color/camera_info" # ROS topic for listening to camera intrinsics
            self.cam_frame = 'camera_color_frame'
        elif self.cam_type == 'kinect_azure' :
            '''kinect azure topic'''
            self.image_topic = "/rgb/image_raw"
            self.image_depth_topic =  "/depth_to_rgb/image_raw"
            self.camera_info_topic = "/rgb/camera_info" # ROS topic for listening to camera intrinsics
            self.cam_frame = 'rgb_camera_link'

        self.results_copse_topic = '/results_copse'

    def camera_initialization(self):
        self.image_rgb = None
        self.image_depth = None
        self.joint_name_pos_dict = None
        self.joint_name_pos_list = None
        self.camera_K = None
        self.bridge = CvBridge()

        self.cam_tfbroadcaster = tf.TransformBroadcaster()
        # Create subscribers

        # Subscriber for camera intrinsics topic
        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.on_camera_info)

        if self.sync:
            # !!!cannot subscribe to the same topic twice from within the same node.!!!
            self.image_rgb_msgfilter = message_filters.Subscriber(self.image_topic, Image)
            self.image_depth_msgfilter = message_filters.Subscriber(self.image_depth_topic, Image)
            self.joint_states_msgfilter = message_filters.Subscriber(self.joint_states_topic, JointState)
            # self.timesync = message_filters.TimeSynchronizer([self.image_rgb_msgfilter, self.image_depth_msgfilter, self.joint_states_msgfilter], 10)
            self.timesync = message_filters.ApproximateTimeSynchronizer([self.image_rgb_msgfilter,
                                                                         self.image_depth_msgfilter,
                                                                         self.joint_states_msgfilter],
                                                                         queue_size=1,
                                                                         slop=0.33*3,
                                                                         allow_headerless=False)
            self.timesync.registerCallback(self.timesync_callback)
        else:
            self.image_rgb_sub = rospy.Subscriber(self.image_topic, Image, self.on_rgb_image)
            self.image_depth_sub = rospy.Subscriber(self.image_depth_topic, Image, self.on_depth_image)
            self.joint_states_sub = rospy.Subscriber(self.joint_states_topic, JointState, self.on_joint_states)

    def robot_joint_names_initialization(self):
        self.joint_name_pos_dict = {}
        self.joint_name_pos_list = []

        if self.robot_name == 'baxter':
            self.robopose_joint_names = [
                'torso_t0',
                'right_s0', 'left_s0',
                'right_s1', 'left_s1',
                'right_e0', 'left_e0',
                'right_e1', 'left_e1',
                'right_w0', 'left_w0',
                'right_w1', 'left_w1',
                'right_w2', 'left_w2',
            ]

            self.base_frame = '/base'

        elif self.robot_name == 'panda':
            self.robopose_joint_names = [
                'panda_joint1',
                'panda_joint2',
                'panda_joint3',
                'panda_joint4',
                'panda_joint5',
                'panda_joint6',
                'panda_joint7',
                'panda_finger_joint1'
            ]
            self.base_frame = '/panda_link0'

    def publisher_initialization(self):
        self.results_copse_pub = rospy.Publisher(self.results_copse_topic, Image, queue_size=1)

    def on_rgb_image(self, data):
        # (480, 640, 3)
        try:
            _image_rgb = CvBridge().imgmsg_to_cv2(data, desired_encoding="bgr8")
            if self.cam_type == 'kinect_azure':
                # self.image_rgb = cv2.resize(_image_rgb, (640, 480), interpolation=cv2.INTER_LINEAR)
                # self.image_rgb = _image_rgb[120:120+480, 320:320+640]
                self.image_rgb = _image_rgb[120:120+480, 400:400+480]
            elif self.cam_type == 'realsense':
                self.image_rgb = deepcopy(_image_rgb)

        except CvBridgeError as e:
            print(e)

    def on_depth_image(self, data):
        # (480, 640)
        try:
            _image_depth = CvBridge().imgmsg_to_cv2(data, desired_encoding='16UC1')
            # self.image_depth = CvBridge().imgmsg_to_cv2(data, desired_encoding='32FC1')
            if self.cam_type == 'kinect_azure':
                # self.image_depth = cv2.resize(_image_depth, (640, 480), interpolation=cv2.INTER_NEAREST)
                # self.image_depth = _image_depth[120:120+480, 320:320+640]
                self.image_depth = _image_depth[120:120+480, 400:400+480]
            elif self.cam_type == 'realsense':
                self.image_depth = deepcopy(_image_depth)

        except CvBridgeError as e:
            print(e)

    def on_camera_info(self, camera_info):
        if not isinstance(self.image_rgb, np.ndarray):
            return
        # Create camera intrinsics matrix
        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]
        _camera_K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        if self.cam_type == 'kinect_azure':
            self.camera_K = deepcopy(_camera_K)
            # self.camera_K[0,:] /= 2.
            # self.camera_K[1, :] /= 1.5
            # self.camera_K[0,2] -= 320
            # self.camera_K[1,2] -= 120
            self.camera_K[0,2] -= 400
            self.camera_K[1,2] -= 120
        elif self.cam_type == 'realsense':
            self.camera_K = deepcopy(_camera_K)

    def on_joint_states(self, joint_states):
        recv_joint_names = joint_states.name
        recv_joint_position = joint_states.position

        # if self.robot_name == 'baxter':
        #     self.joint_name_pos_dict.update({'torso_t0':0.0})
        # elif self.robot_name == 'panda':
        #     self.joint_name_pos_dict.update({'panda_joint1':0.0})

        for joint_name in self.robopose_joint_names:
            if joint_name in recv_joint_names:
                index = recv_joint_names.index(joint_name)
                self.joint_name_pos_dict.update({joint_name:recv_joint_position[index]})

        self.joint_name_pos_list = [self.joint_name_pos_dict[joint_name] for joint_name in self.robopose_joint_names]

    def timesync_callback(self, data_rgb, data_depth, data_joint_states):
        self.on_rgb_image(data_rgb)
        self.on_depth_image(data_depth)
        self.on_joint_states(data_joint_states)

        return self.image_rgb, self.image_depth, self.joint_name_pos_dict, self.joint_name_pos_list

    def on_camera_pose_pub(self, cam_pose, child_link_name, parent_link_name):
        # cam_pose[:3, :3] = np.matmul(cam_pose[:3, :3], graspnet_2_baxter_mapping)
        current_t = cam_pose[:3, 3]
        current_r = cam_pose

        # base in camera_frame (child parent)
        self.cam_tfbroadcaster.sendTransform((current_t[0], current_t[1], current_t[2]),
                                             tf.transformations.quaternion_from_matrix(current_r),
                                             rospy.Time.now(),
                                             child_link_name,
                                             parent_link_name)

    def on_results_pub(self, image):
        '''compressed image'''
        # using opencv for compressing
        compressed_params = [cv2.IMWRITE_JPEG_QUALITY, 30]
        msg = cv2.imencode(".jpg", image, params=compressed_params)[1]
        msg = (np.array(msg)).tobytes()
        msg_compressed = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)
        msg_compressed = CvBridge().cv2_to_imgmsg(image, encoding="bgr8")
        # using cv_bridge for compressing
        # msg = CvBridge().cv2_to_imgmsg(image, encoding="bgr8")
        # msg = CompressedImage()
        # msg.header.stamp = rospy.Time.now()
        # msg.format = "jpeg"
        # msg.data = np.array(cv2.imencode('.jpg', image)[1]).tobytes()
        # msg = CvBridge().cv2_to_compressed_imgmsg(image, dst_format="jpeg")
        self.results_copse_pub.publish(msg_compressed)

class rosTF(object):
    def __init__(self):
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()

    def publish_tf(self, RT, child_frame, parent_frame="/camera_color_frame"):
        send_RT = deepcopy(RT)
        # norm R
        send_RT[:3,:3] =  send_RT[:3,:3] / np.cbrt(np.linalg.det(send_RT[:3,:3]))
        curr_R = send_RT[:3, :3]
        curr_T = send_RT[:3, 3]

        current_quat = tf.transformations.quaternion_from_matrix(send_RT)
        current_quat = current_quat / np.linalg.norm(current_quat)
        self.tf_broadcaster.sendTransform(curr_T,
                                          current_quat,
                                          rospy.Time.now(),
                                          child_frame,
                                          parent_frame)

    def listen_tf(self, parent_frame, child_frame, latest=True):
        is_success = False
        try:
            if latest:
                # either, relax the requirement of time align
                latest = rospy.Time(0)  # latest time, the latest available transform in the buffer
                pose = self.tf_listener.lookupTransform(parent_frame, child_frame, latest)
            else:
                # or
                now = rospy.Time.now()  # current time
                self.tf_listener.waitForTransform(parent_frame, child_frame, now, rospy.Duration(1))
                pose = self.tf_listener.lookupTransform(parent_frame, child_frame, now)

            (trans_list, quat_list) = pose
            T_child_to_parent = np.eye(4)
            T_child_to_parent[:3,:3] = tf.transformations.quaternion_matrix(quat_list)[:3, :3]
            T_child_to_parent[:3, 3] = trans_list

            is_success = True
            return pose, T_child_to_parent, is_success
        except:
            # rospy.logerr('updating tf...')
            return None, None, is_success

if __name__ == '__main__':
    print('test ROS tool')
    ros_receiver = rosImageReceiver(sync=False)
    rate_controller = rospy.Rate(1)
    while not rospy.is_shutdown():
        if isinstance(ros_receiver.image_rgb, np.ndarray) and isinstance(ros_receiver.image_depth, np.ndarray):
            print(ros_receiver.image_rgb.shape)
            print(ros_receiver.image_depth.shape)
            print(ros_receiver.joint_name_pos_dict)
            print(ros_receiver.joint_name_pos_list)

        rate_controller.sleep()