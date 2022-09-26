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


class IngredientValidation:
    """
    This class ties together the methods needed for Ingredient Validation.
    """

    def __init__(self):
        # Params
        self.br = CvBridge()
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(1)

        # Publishers
        self.pub = rospy.Publisher("ingredient", String, queue_size=1)
        self.pub_img = rospy.Publisher("ingredient_img", Image, queue_size=1)

        # Subscribers
        rospy.Subscriber(
            "/camera/color/image_raw",
            Image,
            self.callback,
            queue_size=1,
            buff_size=2 ** 24,
        )

        # Class names
        self.class_names = [
            "blackolives",
            "blackpepper",
            "cabbage",
            "carrot",
            "cherrytomatoes",
            "chilliflakes",
            "corn",
            "cucumbers",
            "greenolives",
            "habaneropepper",
            "mushroom",
            "oregano",
            "peanuts",
            "redonion",
            "salt",
            "sugar",
            "vinegar",
            "water",
            "whiteonion",
        ]  # also "no_ingredient" class added manually

        rospack = rospkg.RosPack()
        weights_path = rospack.get_path("ingredient_validation")

        # Model
        self.model = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub",
            "nvidia_efficientnet_b0",
            verbose=False,
        )
        self.model.classifier.fc = nn.Linear(
            in_features=1280, out_features=len(self.class_names), bias=True
        )
        weights = torch.load(
            weights_path + "/model/efficientNet-b0-svd-for-plots-epoch25.pth"
        )
        self.model.load_state_dict(weights)
        self.model.eval()

    def callback(self, msg):
        """
        This callback is invoked whenever an image is obtained from the RGB Image subscriber
        """
        # Image preprocessing
        image = self.br.imgmsg_to_cv2(msg)
        image_anno = image
        image = np.asarray(image)
        image = PILImage.fromarray(image)
        torch_transform = T.Compose(
            [
                T.CenterCrop((512, 512)),
                T.Resize((512, 512)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        image = torch_transform(image)
        image = torch.unsqueeze(image, dim=0)

        # Obtaining model prediction
        outputs = self.model(image)
        outputs = F.softmax(outputs, dim=1)
        score = torch.max(outputs, 1)
        preds = torch.argmax(outputs, 1)
        if score[0].item() > 0.3:
            prediction = self.class_names[preds]
            pred_string = prediction + " " + str(round(score[0].item(), 2))
        else:
            prediction = "no_ingredient"
            pred_string = prediction
        self.pub.publish(prediction)
        print("Predicted ingredient: ", prediction)
        print("Confidence score: ", score[0].item())

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
            pred_string,
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
