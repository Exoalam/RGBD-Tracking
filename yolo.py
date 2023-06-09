from ultralytics import YOLO
import cv2
import numpy
from ultralytics.yolo.utils.plotting import Annotator
import pyrealsense2 as rs
from scipy.signal import savgol_filter
from realsense_depth import *
import math
import matplotlib.pyplot as plt

def calculate_angle(depth_frame, x1, y1, x2, y2):
    depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

    # Get the depth values for the two points
    depth1 = depth_frame.get_distance(x1, y1)
    depth2 = depth_frame.get_distance(x2, y2)

    # Convert pixel coordinates to 3D coordinates
    point1 = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x1, y1], depth1)
    point2 = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x2, y2], depth2)

    # Calculate the angle between the two points
    angle = math.atan2(point2[1] - point1[1], point2[0] - point1[0]) * 180.0 / math.pi
    if angle < 0:
        angle += 360.0

    return angle

def calc_distance(depth_info,x,y):
    depth_intrinsics = depth_info.profile.as_video_stream_profile().intrinsics
    depth = depth_info.get_distance(x, y)
    point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
    return point

model = YOLO('yolov8n.pt')
dc = DepthCamera()
# cap = cv2.VideoCapture(4)
# cap.set(3, 640)
# cap.set(4, 480)
hit_map = np.zeros((1000,1000))
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
            r_x = box.xywh[0][0]
            r_y = box.xywh[0][1]
            x = box.xywh[0][0]
            y = box.xywh[0][1]
            w = box.xywh[0][2].detach().cpu().numpy()
            h = box.xywh[0][3].detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            x = x.detach().cpu().numpy()
            point = int(x), int(y)
            if c == 41:
                
                # distance = depth_frame[point[1]-5:point[1]+5, point[0]-5:point[0]+5].flatten()
                
                # # print(distance.shape)
                
                
                # distance = savgol_filter(distance, window_length=10, polyorder=5)
                # distance = np.median(distance)
                # distance = numpy.abs(distance*numpy.cos((45/320)*numpy.abs(int(x)-320)))
                # x = int(x)-320
                # y = 240-int(y)
                # # x = distance*numpy.sin((45/320)*numpy.abs(int(x)-320))
                # # y = distance*numpy.sin((30/240)*numpy.abs(240-int(y)))
                depth = depth_info.get_distance(x, y)
                D_point = calc_distance(depth_info,x,y)
                depth = numpy.abs(depth*numpy.cos((45/320)*numpy.abs(int(x)-320)))
                height = h*.8
                width = w*.8
                angle = calculate_angle(depth_info, x, int(y+height/2), x, int(y-height/2))
                depth1 = depth_info.get_distance(x, int(y+height/2))
                depth2 = depth_info.get_distance(x, int(y-height/2))
                height = np.sqrt(depth1 ** 2 + depth2 ** 2 - 2*depth1*depth2*np.cos(angle))
                cv2.circle(color_frame, point, 4, (0, 0, 255))
                hit_map[round(D_point[2]*100),round(D_point[0]*100+320)] += 1 
                # annotator.box_label(b, model.names[int(c)]+" x:"+str(int(x))+" y:"+str(int(y))+" z:"+str(int(distance))+" Height:"+str(int(height))+" Width:"+str(int(width)))
                annotator.box_label(b, model.names[int(c)]+" x:"+str(round(D_point[0],2))+" y:"+str(round(D_point[1],2))+" z:"+str(round(D_point[2],2))+ " Height:"+str(round(height,2)))

    color_frame = annotator.result()  
    cv2.imshow('YOLO V8 Detection', color_frame)     
    if cv2.waitKey(1) & 0xFF == ord('q'):
        plt.imshow(hit_map, cmap='viridis')
        plt.colorbar()

        # Add labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('2D Array Plot')

        # Show the plot
        plt.show()
        break
