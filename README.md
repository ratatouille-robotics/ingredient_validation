# ingredient_validation
This repository contains the ROS Node to validate the ingredient and publish the information into a topic.

**To run:**

rosrun ingredient_validation validate_ingredient.py

**Topics published:**

**/ingredient**: 

String containing detected ingredient

**/ingredient-img**: 

Image containing bounding box indicating cropped image, and ingredient name on the image
