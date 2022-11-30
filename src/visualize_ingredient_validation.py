#!/usr/bin/env python3

"""
Author: Sai Shruthi Balaji

This is a simple test script that constantly runs, reads the current image, performs ingredient validation
and publishes an image with a bounding box and text indicating the identified ingredient.

Note: It is for visualization purposes only and not for use by the state machine.
"""

import rospy
import rospkg
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ingredient_validation.srv import ValidateIngredient
from ar_track_alvar_msgs.msg import AlvarMarkers



class IngredientValidation:
    """
    This class ties together the methods needed for Ingredient Validation.
    """

    def __init__(self):
        # Params
        self.br = CvBridge()
        self.response = ""
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(1)

        # Publishers
        self.pub = rospy.Publisher("ingredient", String, queue_size=1)
        self.pub_img = rospy.Publisher("ingredient_img", Image, queue_size=1)

        # Subscribers
        rospy.Subscriber(
            "/camera/color/image_raw",
            Image,
            self.img_callback,
            queue_size=1,
            buff_size=2 ** 24,
        )
        rospy.Subscriber(
            "/ar_pose_marker", AlvarMarkers, self.callback, queue_size=1
        )

        pasta = False
        if pasta:
            # Class names
            self.class_names = [
                "bell_pepper",
                "black_pepper",
                "cheese",
                "chilli_flakes",
                "garlic_powder",
                "marinara",
                "onion",
                "oregano",
                "pasta",
                "salt",
                "sugar",
                "sunflower_oil",
                "water"
            ]
        else:
            self.class_names = [
                "cumin_seeds",
                "ginger_garlic_paste",
                "kitchen_king",
                "onion",
                "paneer",
                "rice",
                "salt",
                "sunflower_oil",
                "turmeric",
                "water"
            ]

        rospack = rospkg.RosPack()
        weights_path = rospack.get_path("ingredient_validation")

        # Model
        self.model = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub",
            "nvidia_efficientnet_b4",
            verbose=False,
        )
        self.model.classifier.fc = nn.Linear(
            in_features=1792, out_features=len(self.class_names), bias=True
        )
        weights = torch.load(
            weights_path + "/model/efficientNet-b4-pulao-fvd-encore-epoch10.pth"
        )
        self.model.load_state_dict(weights)
        self.model.eval()

    def callback(self, msg):
        """
        This callback is invoked whenever an image is obtained from the RGB Image subscriber
        """
        id = None
        for marker in msg.markers:
            id = marker.id
            break
        if id is not None:
            rospy.wait_for_service("ingredient_validation")
            try:
                service_call = rospy.ServiceProxy(
                    "ingredient_validation", ValidateIngredient
                )
                response = service_call(mode="rgb", id=id)
                print(f"Service Response: {response.found_ingredient}")

            except rospy.ServiceException as e:
                print("Service error")

            self.response = response.found_ingredient
            self.pub.publish(self.response)
            print("Predicted ingredient: ", self.response)

    def img_callback(self, msg):

        # Image preprocessing
        image = self.br.imgmsg_to_cv2(msg)
        image_anno = image

        # Make annotated image
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 500)
        fontScale = 1
        fontColor = (255, 0, 0)
        thickness = 2
        lineType = 2

        # Adding rectangle and text
        height, width, _ = image_anno.shape
        print(width, height)
        upper_left = ((width // 2) - 200, (height // 2) + 200)
        bottom_right = ((width // 2) + 200, (height // 2) - 200)
        cv2.rectangle(image_anno, upper_left, bottom_right, (255, 255, 255), 2)
        cv2.putText(
            image_anno,
            self.response,
            upper_left,
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )
        # Publish image
        self.pub_img.publish(self.br.cv2_to_imgmsg(image_anno))

        # Make annotated image
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 500)
        fontScale = 1
        fontColor = (255, 0, 0)
        thickness = 2
        lineType = 2

        # Adding rectangle and text
        height, width, _ = image_anno.shape
        print(width, height)
        upper_left = ((width // 2) - 200, (height // 2) + 200)
        bottom_right = ((width // 2) + 200, (height // 2) - 200)
        cv2.rectangle(image_anno, upper_left, bottom_right, (255, 255, 255), 2)
        cv2.putText(
            image_anno,
            self.response,
            upper_left,
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )
        image_anno = cv2.cvtColor(image_anno, cv2.COLOR_BGR2RGB)

        # Publish image
        self.pub_img.publish(self.br.cv2_to_imgmsg(image_anno))

    def start(self):
        while not rospy.is_shutdown():
            self.loop_rate.sleep()


if __name__ == "__main__":
    # Start the node
    rospy.init_node("ingredient_validation_node", anonymous=True)
    node = IngredientValidation()
    node.start()
