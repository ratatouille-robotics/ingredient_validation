# Ingredient Validation Pipeline
This repository contains the ROS Service that validates the ingredient using an RGB image. If the ingredient is visually ambiguous (such as salt, sugar, oregano or black pepper), it is validated using its spectral absorbance reading. Finally, the identified ingredient name is sent as a response to the caller.

**To run the service:**

`rosrun ingredient_validation validate_ingredient_service.py`

**To call the service via a client:**

1. RGB Mode:

```
rospy.wait_for_service('ingredient_validation')
service_call = rospy.ServiceProxy('ingredient_validation', ValidateIngredient)
response = service_call(mode='rgb')
```


2. Spectral Mode:

```
rospy.wait_for_service('ingredient_validation')
service_call = rospy.ServiceProxy('ingredient_validation', ValidateIngredient)
response = service_call(mode='spectral', ingredient_name="no_ingredient")
```

**To run data collection:**

1. Create a folder named `data` in ingredient_validation package.
2. Modify `line 40` in the script with the name of the ingredient whose image data is being collected.
3. Run the script with the command:
`rosrun ingredient_validation data_collection.py`
4. Type `s` to draw the bounding box around the object that has to be tracked across frames and press enter.
