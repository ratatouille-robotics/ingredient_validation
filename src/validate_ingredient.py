#!/usr/bin/env python
import rospy
import cv2
import torch.nn as nn
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import os
import numpy as np
import torch
import torchvision.transforms as T

class IngredientValidation:
    def __init__(self):
        # Params
        self.br = CvBridge()
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(1)

        # Publishers
        self.pub = rospy.Publisher('ingredient', String, queue_size=10)

        # Subscribers
        rospy.Subscriber("/camera/color/image_raw",Image,self.callback)

        # Class names
        self.class_names = ["bellpepper","blackolives","blackpepper","cabbage","carrot",
                   "cherrytomatoes","chilliflakes","corn","cucumbers","greenbeans",
                   "greenolives","habaneropepper","mushroom","oregano","redonion",
                   "salt","whiteonion","zucchini"]

        # Model
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0',verbose=False)
        self.model.classifier.fc = nn.Linear(in_features=1280, out_features=len(self.class_names), bias=True)
        weights= torch.load("/home/shruthi/Workspace/pose/src/ingredient_validation/model/checkpoint-efficientNet-b0-epoch8.pth")
        self.model.load_state_dict(weights)
        self.model.eval()

    def callback(self, msg):
        image = self.br.imgmsg_to_cv2(msg)
        image = np.asarray(image)
        image = PILImage.fromarray(image)
        torch_transform = T.Compose([
                T.CenterCrop((720,720)),
                T.Resize((512,512)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        image = torch_transform(image)
        image = torch.unsqueeze(image, dim=0)
        
        outputs = self.model(image)
        preds = torch.argmax(outputs, 1)
        prediction = self.class_names[preds]
        self.pub.publish(prediction)
        #print("Predicted ingredient: ", prediction)

    def start(self):
        while not rospy.is_shutdown():
            # rospy.loginfo('Validating ingredient...')
            self.loop_rate.sleep()

if __name__ == '__main__':
    rospy.init_node('ingredient_validation_node', anonymous=True)
    node = IngredientValidation()
    node.start()