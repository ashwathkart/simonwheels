import rospy
from constants import T_blender2opengl, UTM_ORIGIN_EAST, UTM_ORIGIN_NORTH, ORIGIN_HEIGHT
from sensor_msgs.msg import Image as ROSImage
from novatel_gps_msgs.msg import Inspva  # Adjusted to match the correct message type
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import numpy as np
import utm
from scipy.spatial.transform import Rotation as R
import torch
import cv2

class DataReader:

    def __init__(self, debug=False, window_size=None):
        self.debug = debug
        self.window_size = window_size
        self.bridge = CvBridge()
        self.start_timestamp = None
        self.i = 0
        self.scale = 0.5

        # Define topics and initialize subscribers
        self.rgb_topic = "/zed2/zed_node/right/image_rect_color"
        self.depth_topic = "/zed2/zed_node/depth/depth_registered"
        self.gps_topic = "/novatel/inspva"

        rospy.Subscriber(self.rgb_topic, ROSImage, self.rgb_callback)
        rospy.Subscriber(self.depth_topic, ROSImage, self.depth_callback)
        rospy.Subscriber(self.gps_topic, Inspva, self.gps_callback)
        
        # Initialize data lists
        self.rgbs = []
        self.depths = []
        self.positions = []
        self.poses = []
        self.timestamps = []

    def rgb_callback(self, msg):
        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        rgb = cv2.resize(rgb, self.window_size, interpolation=cv2.INTER_LINEAR)[..., :3].astype(np.float32) / 255
        self.rgbs.append(rgb)

    def depth_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        h, w = depth.shape
        h_new, w_new = int(h * self.scale), int(w * self.scale)
        depth = cv2.resize(depth, (w_new, h_new))
        self.depths.append(depth)

    def gps_callback(self, msg):
        lat = msg.latitude
        lon = msg.longitude
        height = msg.height
        roll = msg.roll
        pitch = msg.pitch
        azimuth = msg.azimuth
        
        utm_data = utm.from_latlon(lat, lon)
        east, north = utm_data[0], utm_data[1]

        east -= UTM_ORIGIN_EAST
        north -= UTM_ORIGIN_NORTH
        height -= ORIGIN_HEIGHT

        self.positions.append(np.array([east, north, height]))
        
        timestamp = rospy.get_time()
        if self.start_timestamp is None:
            self.start_timestamp = timestamp
        self.timestamps.append(timestamp - self.start_timestamp)
        
        if None not in (roll, pitch, azimuth):
            self.calculate_pose(east, north, height, roll, pitch, azimuth)

    def roll_callback(self, msg):
        self.roll = msg.data

    def pitch_callback(self, msg):
        self.pitch = msg.data

    def azimuth_callback(self, msg):
        self.azimuth = msg.data

    def calculate_pose(self, east, north, height, roll, pitch, azimuth):
        pos = np.array([east, north, height])
        euler = np.array([pitch, roll, -azimuth])
        r = R.from_euler('xyz', euler, degrees=True)
        rot = r.as_matrix()

        cam2world = np.eye(4)
        cam2world[:3, :3] = rot
        cam2world[:3, 3] = pos

        cv2gl = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])
        
        world2cam_cv = np.linalg.inv(cam2world)
        world2cam_gl = cv2gl @ world2cam_cv
        self.poses.append(world2cam_gl)

    def __len__(self):
        return len(self.rgbs)

    def __getitem__(self, item):
        return {
            "rgb": self.rgbs[item],
            "pose": self.poses[item] @ T_blender2opengl,
            "depth": self.depths[item]
        }

    def __next__(self):
        res = self[self.i]
        self.i = (self.i + 1) % len(self)
        return res