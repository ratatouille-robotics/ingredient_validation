#!/usr/bin/env python3

"""
Author: Sai Shruthi Balaji

This is an automated data collection script.

Usage: Run the script and select the object to capture.
It will be tracked in subsequent frames and images will be saved.
"""
import os
import cv2
import time
import rospy
import rospkg
import imutils
import argparse
import numpy as np

from imutils.video import VideoStream
from imutils.video import FPS
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


class DataCollection:
    """
    This class represents the Data Collection Node
    """

    def __init__(self):
        # Params
        self.br = CvBridge()
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(1)

        # Name of Ingredient being collected
        self.name = "ginger_garlic_paste"
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("ingredient_validation")
        self.data_path = os.path.join(package_path, "data/rgb_new/" + self.name)
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        # Counter
        self.count = 0

        # Publishers
        self.pub_img = rospy.Publisher("tracked_img", Image, queue_size=1)

        # Subscribers
        rospy.Subscriber(
            "/camera/color/image_raw",
            Image,
            self.callback,
            queue_size=1,
            buff_size=2 ** 24,
        )

        self.tracker = cv2.TrackerKCF_create()
        self.fps = None

    def callback(self, msg):
        """
        This callback is invoked whenever an image is obtained from the RGB Image subscriber
        """
        frame = self.br.imgmsg_to_cv2(msg)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        (H, W) = frame.shape[:2]

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # check to see if we are currently tracking an object
        if self.initBox is not None:
            print("Begin tracking")
            # grab the new bounding box coordinates of the object
            (success, box) = self.tracker.update(frame)
            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                # print(H,W,x,y,w,h)
                cropped = frame[y:y+h, x:x+w]
                filename = self.data_path + '/' + str(self.count).zfill(5) + ".jpg"
                cv2.imwrite(filename, cropped)
                self.count += 1

                cv2.rectangle(frame, (x, y), (x + w, y + h),
                    (0, 255, 0), 2)
                cv2.namedWindow("tracked", cv2.WINDOW_NORMAL)
                cv2.imshow('tracked', frame)

            else:
                print("Failed to track")
            # update the FPS counter
            self.fps.update()
            self.fps.stop()
            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(self.fps.fps())),
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if key == ord("s"):
            self.initBox = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)
            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            self.tracker.init(frame, self.initBox)
            print("Done selecting")
            self.fps = FPS().start()

        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            return

        # # Publish image
        self.pub_img.publish(self.br.cv2_to_imgmsg(frame))

    def start(self):
        while not rospy.is_shutdown():
            self.loop_rate.sleep()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Start the node
    rospy.init_node("data_collection_node", anonymous=True)
    node = DataCollection()
    node.initBox = None
    node.start()
