#!/usr/bin/env python3
import rospy
from ingredient_validation.srv import ValidateIngredient

# Example client node
def validate_ingredient_client():
    try:
        rospy.wait_for_service('ingredient_validation')
        service_call = rospy.ServiceProxy('ingredient_validation', ValidateIngredient)
        print("Calling service")
        response = service_call(mode='rgb', ingredient_name="no_ingredient")
        print(response)
        return response
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


if __name__ == "__main__":
    validate_ingredient_client()