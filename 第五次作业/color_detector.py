import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ColorDetector(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # 方法1：识别蓝色色块（HSV阈值法）
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_result = cv2.bitwise_and(cv_img, cv_img, mask=blue_mask)
        
        # 方法2：识别红色色块（RGB阈值法）
        lower_red = np.array([0, 0, 100])
        upper_red = np.array([50, 50, 255])
        red_mask = cv2.inRange(cv_img, lower_red, upper_red)
        red_result = cv2.bitwise_and(cv_img, cv_img, mask=red_mask)
        
        # 显示结果
        cv2.imshow('Blue Detection (HSV Threshold)', blue_result)
        cv2.imshow('Red Detection (RGB Threshold)', red_result)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    color_detector = ColorDetector()
    rclpy.spin(color_detector)
    color_detector.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
