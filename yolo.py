from ultralytics import YOLO
import cv2
import numpy
from ultralytics.yolo.utils.plotting import Annotator
import pyrealsense2 as rs
from scipy.signal import savgol_filter
from realsense_depth import *
from scipy.spatial import distance
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import threading
import time
import csv
import rospy
from std_msgs.msg import String

data = []

custom_dtype = np.dtype([
    ('hit', np.int8),       
    ('accuracy', np.int8),
    ('class', np.int8)       
])
map = np.zeros((200, 200, 200), dtype=custom_dtype)
pub_string = ""
def max_hit(points):
    final_list = []
    dis = 0
    for p in points:
        point_list = []
        value_list = []
        for point in points:
            dis = distance.euclidean(point, p)
            if dis < 20:
                point_list.append(point)
                value_list.append(map[point]['hit'])
        max_point = point_list[value_list.index(max(value_list))]
        if max_point not in final_list:        
            final_list.append(max_point)
    return final_list        
def timer():
    global pub_string
    pub.publish(pub_string)

def call_function_periodically():
    while True:
        timer()
        time.sleep(5)


def calculate_angle_2d_x_axis(P1, P2):

    theta = np.arctan2(P2[1] - P1[1], P2[0] - P1[0])


    theta = np.degrees(theta)

    return theta
def calculate_angle_2d(P1, P2):
    P1 = np.array(P1)
    P2 = np.array(P2)

    dot_product = np.dot(P1, P2)
    magnitude_P1 = np.linalg.norm(P1)
    magnitude_P2 = np.linalg.norm(P2)

    cos_theta = dot_product / (magnitude_P1 * magnitude_P2)

    theta = np.arccos(cos_theta)

    # Convert theta from radians to degrees
    theta = np.degrees(theta)

    return theta

def calc_distance(depth_info,x,y,depth):
    depth_intrinsics = depth_info.profile.as_video_stream_profile().intrinsics
    point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
    return point

model = YOLO('yolov8n.pt')
dc = DepthCamera()
target_dis = 1
# cap = cv2.VideoCapture(4)
# cap.set(3, 640)
# cap.set(4, 480)
hit_map = np.zeros((1000,1000))
data.append(['Angle','Static Z','Angled Z', 'Calculated Z', 'Static Accuracy', 'Angled Accuracy'])
detect_list = [39,41]
text = [39,41]
rospy.init_node('yolo_new', anonymous=True)
pub = rospy.Publisher('/object_info', String, queue_size=10)
thread = threading.Thread(target=call_function_periodically)
thread.start()

while True:
    
    ret, depth_frame, color_frame, depth_info = dc.get_frame()
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
            old_data = [(0,0,0),0]
            if c in detect_list:
                points = dc.Global_points(point[0],point[1])
                dis = dc.actual_depth(point[0],point[1])
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
                depth = points[0][2]
                D_point = calc_distance(depth_info,x,y,depth)
                
       
                #depth = numpy.abs(depth*numpy.cos((45/320)*numpy.abs(int(x)-320)))
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
                #angle = calculate_angle_2d((x,y),(320,240))
                angle = np.abs(90/640*x-45)
                dis = np.abs(dis * np.cos(angle))
                print(angle)
                print(dis)
                static_accuracy = 100 - np.abs(target_dis-D_point[2])*100/target_dis
                angled_accuracy = 100 - np.abs(dis-D_point[2])*100/dis
                data.append([round(angle,2), target_dis, round(dis,2), round(D_point[2],2), round(static_accuracy,2), round(angled_accuracy,2)])
                map[round(D_point[0]*100),round(D_point[1]*100),round(D_point[2]*100)]['hit'] += 1
                map[round(D_point[0]*100),round(D_point[1]*100),round(D_point[2]*100)]['class'] = c
                # distance = np.linalg.norm(D_point - old_data[0]) * 100
                
                # if angle > 0 and angle < 10:
                #     map[round(D_point[0]*100),round(D_point[1]*100),round(D_point[2]*100)]['accuracy'] = 99
                # if angle > 10 and angle < 20:
                #     map[round(D_point[0]*100),round(D_point[1]*100),round(D_point[2]*100)]['accuracy'] = 98   
                # if angle > 30 and angle < 40:
                #     map[round(D_point[0]*100),round(D_point[1]*100),round(D_point[2]*100)]['accuracy'] = 97 
                # if angle > 40:
                #     map[round(D_point[0]*100),round(D_point[1]*100),round(D_point[2]*100)]['accuracy'] = 95                                                       
                # print(idx)
                # distance *= np.cos(angle)
                # if idx.shape[0] != 0:
                cv2.circle(color_frame, point, 4, (0, 0, 255))
                cv2.circle(color_frame, (320,240), 4, (0, 0, 255))
                #hit_map[round(D_point[2]*100),round(D_point[0]*100+320)] += 1 
                # annotator.box_label(b, model.names[int(c)]+" x:"+str(int(x))+" y:"+str(int(y))+" z:"+str(int(distance))+" Height:"+str(int(height))+" Width:"+str(int(width)))
                annotator.box_label(b, model.names[int(c)]+" x:"+str(round(D_point[0],2))+" y:"+str(round(D_point[1],2))+" z:"+str(round(D_point[2],2))+ " Angle:"+str(round(angle,2)))
                x =  '{ "name":"John", "age":30, "city":"New York"}'
                pub_string = '{"class":'+str(int(c))+',"model":'+str(model.names[int(c)])+',"x":'+str(round(D_point[0],2))+',"y":'+str(round(D_point[1],2))+',"z":'+str(round(D_point[2],2))+'}'
                if c in text:
                    text.remove(c)
                    #f = open("/home/shuvo/RGBD-Tracking/Map.txt", "w")
                    #f.write(str(model.names[int(c)])+" x:"+str(round(D_point[0],2))+" y:"+str(round(D_point[1],2))+" z:"+str(round(D_point[2],2)) + '\n')
                    #f.close()
                    with open('Map.txt', 'a') as file:
                        # Move the file pointer to the end of the file
                        file.seek(0, 2)  # 2 indicates moving to the end of the file


                        new_object = str(model.names[int(c)])+" x:"+str(round(D_point[0],2))+" y:"+str(round(D_point[1],2))+" z:"+str(round(D_point[2],2))
                        file.write(new_object + '\n')
                    

    color_frame = annotator.result()  
    cv2.imshow('YOLO V8 Detection', color_frame)     
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break  

indices = np.where(map['hit'] > 1)

points = list(zip(indices[0], indices[1], indices[2]))
points = max_hit(points)
datax = []
with open('hitmap.txt', 'a') as file:
    file.seek(0, 2) 
    for i, point in enumerate(points):
        x= f"Point {i+1}: {point} {map[point]}"
        datax.append([point[0],point[1],point[2],map[point]['hit']])
        file.write(x + '\n')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs, ys, zs, hits = zip(*datax)
ax.scatter(xs, ys, zs, s=hits)
plt.show()
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for row in data:
        writer.writerow(row)
