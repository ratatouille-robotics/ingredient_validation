#!/usr/bin/env python3

"""
Author: Sai Shruthi Balaji

This file includes a method to perform classification of ingredients based on similarity of spectra.
The distance metric used for comparison is Fretchet distance.

Read more at:
https://jekel.me/similarity_measures/similaritymeasures.html#similaritymeasures.similaritymeasures.frechet_dist
"""

import os
import sys
print(sys.executable)
import rospkg
import numpy as np
import pandas as pd
import similaritymeasures
import matplotlib.pyplot as plt
from turtle import color

# List of ingredients
# ingredient_names = ['salt','sugar','blackpepper', 'oregano', 'bellpepper', 'cucumber']
ingredient_names = ['blackpepper', 'oregano', 'salt', 'sugar']
valid_folders = ['Salt','Sugar', 'Pepper', 'Oregano']

# Setting colors for plotting and visualization
colors = {'salt': 'blue', 'sugar': 'green', 'blackpepper': 'black', 'oregano': 'yellow', 'unknown': 'orange'}

# Load existing data
rospack = rospkg.RosPack()
package_path = rospack.get_path("ingredient_validation")
data_path = os.path.join(package_path, 'data/spectral_absorbance')
data_folders = os.listdir(os.path.join(os.getcwd(), data_path))

def classify_spectra(test_sample: pd.DataFrame = None) -> str:
    """
    Method to perform classification by comparing Fretchet distance with existing dataset
    """
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
        if folder in ingredient_names:
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

####################################### Test code ###############################################
# # To generate plot
# def legend_without_duplicate_labels(figure):
#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     figure.legend(by_label.values(), by_label.keys(), loc='upper right')

# # Test a sample
# for sample in os.listdir(os.path.join(package_path, 'data/spectral_absorbance/test/blackpepper')):
#     sample_path = os.path.join(package_path+'/'+'data/spectral_absorbance/test/blackpepper', sample)
#     test_sample = pd.read_csv(sample_path)
#     test_sample = test_sample.iloc[28:,:4]
#     result = classify_spectra(test_sample)

#     print("***Ingredient is identified to be " + result + "***")

# # Generate plot
# legend_without_duplicate_labels(plt)
# plt.show()