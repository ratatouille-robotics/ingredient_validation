#!/usr/bin/env python3

"""
Author: Sai Shruthi Balaji

This service, when called, reads the current snapshot of image from the camera,
passes it through the ingredient validation model and sends the response.

Note: It is currently used by the state machine.
Assumption: Fill level of container is high enough such that it is in the FOV of spectral camera
"""
import cv2
import os
import numpy as np
import pandas as pd
import pickle
import rospy
import rospkg
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import similaritymeasures
import matplotlib.pyplot as plt
from turtle import color
from datetime import datetime

from PIL import Image as PILImage
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ingredient_validation.srv import (
    ValidateIngredient,
    ValidateIngredientRequest,
    ValidateIngredientResponse,
)
class IngredientValidationService:
    """
    This class binds all the methods needed for the ingredient validation service
    """

    def __init__(self):
        # Initialize needed items
        self.br = CvBridge()
        self.loop_rate = rospy.Rate(1)
        self.unsure = False

        self.class_names = [
            "bell_pepper",
            "black_pepper",
            "cheese",
            "cherry_tomatoes",
            "chilli_flakes",
            "cumin_seeds",
            "garlic_powder",
            "ginger_garlic_paste",
            "kitchen_king",
            "mustard_seeds",
            "onion",
            "oregano",
            "paneer",
            "pasta",
            "rice",
            "salt",
            "sugar",
            "sunflower_oil",
            "turmeric",
            "water"
        ]

        self.visually_similar_classes = [
            "blackpepper",
            "oregano",
            "salt",
            "sugar",
        ]

        self.camera_rgb_topic = "/camera/color/image_raw"

        # Load model & weights
        rospack = rospkg.RosPack()
        weights_path = rospack.get_path("ingredient_validation")
        self.model = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub",
            "nvidia_efficientnet_b0",
            verbose=False,
        )
        self.model.classifier.fc = nn.Linear(
            in_features=1280, out_features=len(self.class_names), bias=True
        )
        weights = torch.load(
            weights_path + "/model/efficientNet-b0-dataset-v2-12-11-epoch10.pth"
        )
        self.model.load_state_dict(weights)

    def rgb_validation(self) -> str:
        """
        Validate ingredient from RGB image

        Returns:
            string: The ingredient present in the image
        """
        try:
            # Get current image
            image = rospy.wait_for_message(self.camera_rgb_topic, Image)

            # Do a forward pass, get prediction and scores
            self.model.eval()
            image = self.br.imgmsg_to_cv2(image)

            # Log the image
            rospack = rospkg.RosPack()
            package_path = rospack.get_path("ingredient_validation")
            log_path = package_path + "/logs"
            os.mkdir(log_path)
            now = datetime.now()
            timestamp = now.strftime("%m_%d_%y_%H_%M_%S")
            cv2.imwrite(image, now.strftime(log_path + "/ing_" + timestamp + ".jpg"))

            # Preprocess image
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

            # Forward passq
            outputs = self.model(image)
            outputs = F.softmax(outputs, dim=1)
            score = torch.max(outputs, 1)
            preds = torch.argmax(outputs, 1)

            # If score < 0.3, then say "No ingredient found"
            prediction = ""
            if score[0].item() > 0.3:
                prediction = self.class_names[preds]
            else:
                prediction = "no_ingredient"
            return prediction

        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def classify_spectra(self, test_sample: pd.DataFrame = None) -> str:
        """
        Method to perform classification by comparing Fretchet distance with existing dataset
        The spectral classification of ingredients is performed based on similarity of spectra.
        The distance metric used for comparison is Fretchet distance.

        Read more at:
        https://jekel.me/similarity_measures/similaritymeasures.html#similaritymeasures.similaritymeasures.frechet_dist
        """
        # Load existing data
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("ingredient_validation")
        data_path = os.path.join(package_path, 'data/spectral_absorbance')
        data_folders = os.listdir(os.path.join(os.getcwd(), data_path))

        # List of ingredients
        ingredient_pairs = [['blackpepper', 'oregano'], ['salt', 'sugar']]

        # Setting colors for plotting and visualization
        # colors = {'salt': 'blue', 'sugar': 'green', 'blackpepper': 'black', 'oregano': 'yellow', 'unknown': 'orange'}

        # Input test sample
        test_sample = test_sample.iloc[:,:2]
        test_sample = test_sample.to_numpy().astype(np.float64)

        # Code to plot
        # plt.plot(test_sample[:,0], test_sample[:,1], color=colors['unknown'], label='unknown')

        # Initialize minimum distance
        minimum_distance = float('inf')
        result = ""

        # For each ingredient in dataset, compute average frechet distance
        for folder in data_folders:
            for pair in ingredient_pairs:
                if folder in pair:
                    valid_folders = pair
            if folder in valid_folders:
                current_distance = 0
                dtw = 0
                csv_files = os.listdir(os.path.join(data_path, folder))
                for file in csv_files:
                    df = pd.read_csv(os.path.join(data_path, folder, file))
                    current_sample = df.iloc[28:,:2]
                    current_sample = current_sample.to_numpy().astype(np.float64)

                    # Code to plot
                    # plt.plot(current_sample[:,0], current_sample[:,1], color=colors[folder], label=folder)

                    # Compute frechet distance between current sample and test sample
                    current_distance += similaritymeasures.frechet_dist(current_sample, test_sample)
                    d, dist = similaritymeasures.dtw(current_sample, test_sample)
                    dtw += d

                # Average the distance
                current_distance = current_distance / len(csv_files)
                # print("Average distance from " + folder + " is " + str(current_distance))

                # Classified as the ingredient with minimum frechet curve distance
                if current_distance < minimum_distance:
                    minimum_distance = current_distance
                    result = folder

        return result

    def spectral_validation(self, current_ingredient) -> str:
        """
        This method invokes a spectral scan, identifies the ingredient and returns the corresponding class name.

        Returns:
            string: Ingredient class name
        """

        try:
            # We need a TCP connection with the windows server to which spectral camera is connected
            tcp_socket = socket.create_connection(('10.0.1.2', 65000), timeout=5)
            print("Connection established")

            # Send a test message to the windows server application
            data = "spectra"
            tcp_socket.sendall(str.encode(data))

            # Receive spectral data
            spectra = []
            packet = tcp_socket.recv(4096)
            while packet:
                data = pickle.loads(packet)
                spectra.append(data)
                packet = tcp_socket.recv(4096)

            # Store data in pandas df
            self.spectra_df = pd.DataFrame(spectra[1:], columns=spectra[0])

            # Get the prediction and return
            prediction = self.classify_spectra(self.spectra_df)

        except:
            print("Connection with windows machine failed!")
            prediction = current_ingredient

        finally:
            print("Closing socket")
            tcp_socket.close()

        return prediction

    def handle_ingredient_validation(
        self,
        req: ValidateIngredientRequest
    ) -> ValidateIngredientResponse:
        """
        Handler for the service

        :return: A ValidateIngredientResponse Msg that contains a string indicating the detected ingredient.
        """
        if req.mode == 'rgb':
            prediction = self.rgb_validation()

        elif req.mode == 'spectral':
            # Invoke spectral validation only if we have identified one of the visually similar ingredients,
            # or if confidence threshold is low
            if req.ingredient_name in self.visually_similar_classes or req.ingredient_name == "no_ingredient":
                prediction = self.spectral_validation(req.ingredient_name)
            else:
                prediction = req.ingredient_name

        if prediction:
            response = ValidateIngredientResponse()
            response.found_ingredient = prediction
            return response


def IngredientValidationServer():
    """
    Main server method
    """
    # Initialize node and call service handler
    rospy.init_node("ingredient_validation_node", anonymous=True)

    # Instantiate the service object and call the service
    ValidationServiceObj = IngredientValidationService()
    service = rospy.Service(
        "ingredient_validation",
        ValidateIngredient,
        ValidationServiceObj.handle_ingredient_validation,
    )
    print("Ready to validate ingredient. ")
    rospy.spin()


if __name__ == "__main__":
    IngredientValidationServer()
