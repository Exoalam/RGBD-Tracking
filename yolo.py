from ultralytics import YOLO
import cv2
import numpy
from ultralytics.yolo.utils.plotting import Annotator
import pyrealsense2 as rs
from scipy.signal import savgol_filter
from realsense_depth import *
import math
import matplotlib.pyplot as plt
import json
import threading
import time

import rospy
from std_msgs.msg import String

Global_point = (0,0)
pub_string = ""
def timer():
    global pub_string
    pub.publish(pub_string)

def call_function_periodically():
    while True:
        timer()
        time.sleep(5)

# Create a new thread and start it

def calculate_angle(depth_frame, x1, y1, x2, y2):
    depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

    # Get the depth values for the two points
    depth1 = depth_frame.get_distance(x1, y1)
    depth2 = depth_frame.get_distance(x2, y2)

    # Convert pixel coordinates to 3D coordinates
    point1 = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x1, y1], depth1)
    point2 = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x2, y2], depth2)
    # Calculate the angle between the two points
    angle = math.atan2((point2[1] - point1[1]), point2[0] - point1[0]) * 180.0 / math.pi

    return angle

def calc_distance(depth_info,x,y,depth):
    depth_intrinsics = depth_info.profile.as_video_stream_profile().intrinsics
    point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
    return point

model = YOLO('yolov8n.pt')
dc = DepthCamera()
# cap = cv2.VideoCapture(4)
# cap.set(3, 640)
# cap.set(4, 480)
hit_map = np.zeros((1000,1000))
detect_list = [39,41]
text = [39,41]
rospy.init_node('yolo_new', anonymous=True)
pub = rospy.Publisher('/object_info', String, queue_size=10)
thread = threading.Thread(target=call_function_periodically)
thread.start()

while True:
    
    ret, depth_frame, color_frame, depth_info, global_cod = dc.get_frame()
    img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    results = model.predict(img)
    for r in results:
        
        annotator = Annotator(color_frame)
        
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            pt1 = (int(b[0].detach().cpu().numpy()),int(b[1].detach().cpu().numpy()))
            x = box.xywh[0][0]
            y = box.xywh[0][1]
            w = box.xywh[0][2].detach().cpu().numpy()
            h = box.xywh[0][3].detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            x = x.detach().cpu().numpy()
            point = int(x), int(y)
            if c in detect_list:
                Global_point = point
                # distance = depth_frame[pt1[0]:pt1[0]+int(w),pt1[1]:pt1[1]+int(h)]
                # distance = np.average(distance)
                # idx = np.argwhere(depth_frame == distance)
                #print(global_cod[point[0],point[1]])
                # distance = savgol_filter(distance, window_length=10, polyorder=5)
                # distance = np.median(distance)
                #distance = numpy.abs(distance*numpy.cos((45/320)*numpy.abs(int(x)-320)))
                # x = int(x)-320
                # y = 240-int(y)
                # # x = distance*numpy.sin((45/320)*numpy.abs(int(x)-320))
                # # y = distance*numpy.sin((30/240)*numpy.abs(240-int(y)))
                depth = global_cod[0][2]
                print(depth)
                D_point = calc_distance(depth_info,x,y,depth)
                depth = numpy.abs(depth*numpy.cos((45/320)*numpy.abs(int(x)-320)))
                height = h*.8
                width = w*.8
                # angle = calculate_angle(depth_info, x, round(y+height/2), x, round(y-height/2))
                # depth1 = depth_info.get_distance(x, round(y+height/2))
                # depth2 = depth_info.get_distance(x, round(y-height/2))
                # height = np.sqrt(depth1 ** 2 + depth2 ** 2 - 2*depth1*depth2*np.cos(angle))
                # angle = calculate_angle(depth_info, round(x+width/2), y, round(x-width/2), y)
                # depth1 = depth_info.get_distance(round(x+width/2), y)
                # depth2 = depth_info.get_distance(round(x-width/2), y)
                # width = np.sqrt(depth1 ** 2 + depth2 ** 2 - 2*depth1*depth2*np.cos(angle))
                angle = calculate_angle(depth_info, point[0], point[1], 320, 240)
                # print(idx)
                # distance *= np.cos(angle)
                # if idx.shape[0] != 0:
                cv2.circle(color_frame, point, 4, (0, 0, 255))
                cv2.circle(color_frame, (320,240), 4, (0, 0, 255))
                #hit_map[round(D_point[2]*100),round(D_point[0]*100+320)] += 1 
                # annotator.box_label(b, model.names[int(c)]+" x:"+str(int(x))+" y:"+str(int(y))+" z:"+str(int(distance))+" Height:"+str(int(height))+" Width:"+str(int(width)))
                annotator.box_label(b, model.names[int(c)]+" x:"+str(round(D_point[0],2))+" y:"+str(round(D_point[1],2))+" z:"+str(round(np.abs(D_point[2]),2))+ " Height:"+str(round(height,2)))
                x =  '{ "name":"John", "age":30, "city":"New York"}'
                pub_string = '{"class":'+str(int(c))+',"model":'+str(model.names[int(c)])+',"x":'+str(round(D_point[0],2))+',"y":'+str(round(D_point[1],2))+',"z":'+str(round(D_point[2],2))+'}'
                print(pub_string)
                if c in text:
                    text.remove(c)
                    #f = open("/home/shuvo/RGBD-Tracking/Map.txt", "w")
                    #f.write(str(model.names[int(c)])+" x:"+str(round(D_point[0],2))+" y:"+str(round(D_point[1],2))+" z:"+str(round(D_point[2],2)) + '\n')
                    #f.close()
                    with open('Map.txt', 'a') as file:
                        # Move the file pointer to the end of the file
                        file.seek(0, 2)  # 2 indicates moving to the end of the file

                        # Write the new entry followed by a newline character
                        new_object = str(model.names[int(c)])+" x:"+str(round(D_point[0],2))+" y:"+str(round(D_point[1],2))+" z:"+str(round(D_point[2],2))
                        file.write(new_object + '\n')
                    

    color_frame = annotator.result()  
    cv2.imshow('YOLO V8 Detection', color_frame)     
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # plt.imshow(hit_map, cmap='viridis')
        # plt.colorbar()

        # # Add labels and title
        # plt.xlabel('X-axis')
        # plt.ylabel('Y-axis')
        # plt.title('2D Array Plot')

        # # Show the plot
        # plt.show()
        break
