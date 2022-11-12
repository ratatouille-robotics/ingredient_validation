# ingredient_validation
This repository contains the ROS Node to validate the ingredient and publish the information into a topic.

**To run:**

rosrun ingredient_validation validate_ingredient.py

**To run data collection:**
rosrun ingredient_validation data_collection.py

1. Create a folder named "data" in ingredient_validation package.
2. Modify line:40 in the script with respecttive ingredient name. 
3. Run the script with above command and draw the bounding box to track across frames and press enter.

**Topics published:**

**/ingredient**:

String containing detected ingredient

**/ingredient-img**:

Image containing bounding box indicating cropped image, and ingredient name on the image
