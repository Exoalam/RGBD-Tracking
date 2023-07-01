#! usr/bin/env/python3

import cv2
import yaml
import json
import rospy
from std_msgs.msg import String

obj_info = []
position_data =[]
# Load the map image
map_image = cv2.imread("cropped_map2.png")
# Load the metadata from the YAML file
with open("testbedmap.yaml") as file:
    metadata = yaml.safe_load(file)
    print(metadata)
    
def add_points_to_map(map_image, metadata, point):
    resolution = metadata["resolution"]
    origin_x = metadata["origin"][0]
    origin_y = metadata["origin"][1]
    map_height, map_width = map_image.shape[:2]
    point_x_map = float(point['x'])
    point_y_map = float(point['y'])

    # point_x_image = int((point_x_map - origin_x) * resolution)
    # point_y_image = int((point_y_map - origin_y) * resolution)
    point_x_image = int((point_x_map) * resolution)
    point_y_image = int((point_y_map) * resolution)
    
    print(point_x_image)
    print(point_y_image)
    
        # Check if the point is within the bounds of the map image
    if 0 <= point_x_image < map_width and 0 <= point_y_image < map_height:
        point_color = (0, 0, 255)  # Red color
        point_radius = 5
        cv2.circle(map_image, (point_x_image, point_y_image), point_radius, point_color, -1)
        cv2.imwrite("the_modified_SHUVO254.png", map_image)
        


#callback funtion for object data info
def obj_info_callback(data):
    Json = json.loads(data.data)
    add_points_to_map(map_image, metadata, Json)
    

#function to extract positions
def position_data(data):
    #position information
    position_data.append(data)

   


#initialize a node
the_node = rospy.init_node("mapper")

#subscriber to the obj info topic
obj_info_sub = rospy.Subscriber("chatter", String, obj_info_callback)






# Add points to the map image
# print(obj_info)
# modified_map_image = add_points_to_map(map_image, metadata, obj_info)

# Display the modified map image
#cv2.imshow("Modified Map", modified_map_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the modified map image
# cv2.imwrite("the_modified_SHUVO254.png", modified_map_image)
# print("map succesfully saved")
rospy.spin()