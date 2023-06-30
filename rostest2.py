
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class DepthSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.depth_scale = 0.001
        self.sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.callback)

    def callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        cv_image_meters = cv_image.astype(float) * self.depth_scale
        cv2.imshow('Depth image', cv_image_meters / cv_image_meters.max())
        cv2.waitKey(1)

if __name__ == "__main__":
    rospy.init_node('depth_subscriber', anonymous=True)
    ds = DepthSubscriber()
    rospy.spin()
