from ultralytics import YOLO
import cv2
import numpy
from ultralytics.utils.plotting import Annotator
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
# import rospy
# from std_msgs.msg import String


data = []

def percentage(frame_h, frame_w, h, w):
    return int(((h*w)/(frame_h*frame_w))*100)

custom_dtype = np.dtype([
    ('hit', np.int8),       
    ('accuracy', np.int8),
    ('class', np.int8)       
])
map = np.zeros((1000, 1000, 1000), dtype=custom_dtype)
pub_string = ""
def max_hit(points):
    final_list = []
    dis = 0
    for p in points:
        point_list = []
        value_list = []
        for point in points:
            dis = distance.euclidean(point, p)
            if dis < 10:
                point_list.append(point)
                value_list.append(map[point]['hit'])
        max_point = point_list[value_list.index(max(value_list))]
        if max_point not in final_list:        
            final_list.append(max_point)
    return final_list        


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
hit_map = np.zeros((1000,1000))
data.append(['Angle','Static Z','Angled Z', 'Calculated Z', 'Static Accuracy', 'Angled Accuracy'])
detect_list = [39,41,-99]
text = [39,41]
# rospy.init_node('yolo_new', anonymous=True)
# pub = rospy.Publisher('/object_info', String, queue_size=10)
# thread = threading.Thread(target=call_function_periodically)
# thread.start()

while True:
    
    ret, depth_frame, color_frame, depth_info = dc.get_frame()
    img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    img_shape = img.shape
    robot = (0,0,0)
    map[robot]['hit'] = 0
    results = model.predict(img)
    for r in results:
        
        annotator = Annotator(color_frame)
        
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            _b = box.xywh[0].detach().cpu().numpy()
            pt1 = (int(b[0].detach().cpu().numpy()),int(b[1].detach().cpu().numpy()))
            x = box.xywh[0][0]
            y = box.xywh[0][1]
            w = box.xywh[0][2].detach().cpu().numpy()
            h = box.xywh[0][3].detach().cpu().numpy()
            _c = box.conf.detach().cpu().numpy()
            ratio = w/h
            y = y.detach().cpu().numpy()
            x = x.detach().cpu().numpy()          
            point = int(x), int(y)
            if c in detect_list:
                if cv2.waitKey(1) & 0xFF == ord('m'):
                    x = int(input('X: '))
                    y = int(input('Y: '))
                    z = int(input('Z: '))
                    map[robot]['hit'] = 0
                    robot = (x,y,z)
                    map[robot]['hit'] = 100
                points = dc.Global_points(point[0],point[1])
                dis = dc.actual_depth(point[0],point[1])
                depth = points[0][2]
                D_point = calc_distance(depth_info,x,y,depth)
                D_point[1] = D_point[1]*-1
                angle = np.abs(90/640*x-45)
                dis = np.abs(dis * np.cos(angle))
                static_accuracy = 100 - np.abs(target_dis-D_point[2])*100/target_dis
                angled_accuracy = 100 - np.abs(dis-D_point[2])*100/dis
                data.append([round(angle,2), target_dis, round(dis,2), round(D_point[2],2), round(static_accuracy,2), round(angled_accuracy,2)])
                dx = 0
                dy = 0
                dx = 500+D_point[0]*100
                dy = 500+D_point[1]*100
                map[round(dx),round(dy),round(D_point[2]*100)]['hit'] += 1
                map[round(dx),round(dy),round(D_point[2]*100)]['class'] = c
                cv2.circle(color_frame, point, 4, (0, 0, 255))
                cv2.circle(color_frame, (320,240), 4, (0, 0, 255))
                p = percentage(img_shape[0],img_shape[1],_b[3],_b[2])
                #annotator.box_label(b, model.names[int(c)] + " " + str(round(float(_c), 2)) + " " + str(p) +" "+str(ratio))
                annotator.box_label(b, model.names[int(c)]+" x:"+str(round(D_point[0],2))+" y:"+str(round(D_point[1],2))+" z:"+str(round(D_point[2],2))+ " Pb:" + str(p) +" Rt:"+str(ratio))
                x =  '{ "name":"John", "age":30, "city":"New York"}'
                #pub_string = '{"class":'+str(int(c))+',"model":'+str(model.names[int(c)])+',"x":'+str(round(D_point[0],2))+',"y":'+str(round(D_point[1],2))+',"z":'+str(round(D_point[2],2))+'}'
                if c in text:
                    text.remove(c)
                    with open('Map.txt', 'a') as file:
                        file.seek(0, 2)
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
data_actual = []
for i in detect_list:
    count = 1
    for x in points:
        if map[x]['class'] == i:
            cls = map[x]['class']
            data_actual.append(['class:'+str(cls)+'/'+model.names[int(cls)],' Object: '+str(count), ' Pos: ',x])
            count += 1

with open('hitlist.txt', 'a') as file:
    file.seek(0, 2) 
    for i in data_actual:
        file.write(str(i)+'\n')

datax = []
for i, point in enumerate(points):
    datax.append([point[0],point[1],point[2],map[point]['hit']])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs, ys, zs, hits = zip(*datax)
ax.scatter(xs, ys, zs, s=hits)
plt.show()
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for row in data:
        writer.writerow(row)
csv_file_path = "data.csv"

with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    for row in data:
        csv_writer.writerow(row)
