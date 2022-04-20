#!/usr/bin/env python3
import rospy
import rospkg
import cv2
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import torch
import torchvision.transforms as T

from ingredient_validation.srv import ValidateIngredient, ValidateIngredientRequest, ValidateIngredientResponse

def handle_ingredient_validation(_ : ValidateIngredientRequest):
    try:
        # Params
        br = CvBridge()
        # Node cycle rate (in Hz).
        loop_rate = rospy.Rate(1)

        # Class names
        # class_names = ["bellpepper","blackolives","blackpepper","cabbage","carrot",
        #                 "cherrytomatoes","chilliflakes","corn","cucumbers","greenbeans",
        #                 "greenolives","habaneropepper","mushroom","oregano","peanuts",
        #                 "redonion","salt","sugar","vinegar","water","whiteonion","zucchini"]

        class_names = ["blackolives","blackpepper","cabbage","carrot",
                    "cherrytomatoes","chilliflakes","corn","cucumber",
                    "greenolives","habaneropepper","mushroom","oregano","peanuts",
                    "redonion","salt","sugar","vinegar","vinegar","whiteonion"]

        rospack = rospkg.RosPack()
        weights_path = rospack.get_path('ingredient_validation')

        # Load model & weights
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0',verbose=False)
        model.classifier.fc = nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
        # weights= torch.load(weights_path + "/model/efficientNet-b0-svd-strong-epoch5.pth")
        # weights= torch.load(weights_path + "/model/efficientNet-b0-svd-improved-epoch20.pth")
        weights= torch.load(weights_path + "/model/efficientNet-b0-svd-for-plots-epoch25.pth")
        model.load_state_dict(weights)
        model.eval()

        # Current image
        camera_rgb_topic = "/camera/color/image_raw"
        image = rospy.wait_for_message(camera_rgb_topic, Image)

        # Do a forward pass, get prediction and scores
        image = br.imgmsg_to_cv2(image)
        image = np.asarray(image)
        image = PILImage.fromarray(image)
        torch_transform = T.Compose([
                T.CenterCrop((512,512)),
                T.Resize((512,512)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        image = torch_transform(image)
        image = torch.unsqueeze(image, dim=0)

        outputs = model(image)
        outputs = F.softmax(outputs, dim=1)
        score = torch.max(outputs, 1)
        preds = torch.argmax(outputs, 1)

        # If score < 0.3, then say "No ingredient found"
        if score[0].item() > 0.3:
            prediction = class_names[preds]
        else:
            prediction = "No ingredient found"

        # print("Predicted ingredient: ", prediction)
        # print("Confidence score: ", score[0].item())
        temp = ValidateIngredientResponse()
        temp.found_ingredient = prediction
        return temp

    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def IngredientValidationServer():
    #Init node
    rospy.init_node('ingredient_validation_node', anonymous=True)
    service = rospy.Service('ingredient_validation', ValidateIngredient, handle_ingredient_validation)
    print("Ready to validate ingredient. ")
    rospy.spin()

if __name__ == '__main__':
    IngredientValidationServer()
