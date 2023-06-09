from __future__ import print_function
from ultralytics import YOLO
import cv2
import numpy
from ultralytics.yolo.utils.plotting import Annotator
import pyrealsense2 as rs
from realsense_depth import *
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import threading
import time
import csv
import rospy
from std_msgs.msg import String
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
pointx = []


      
      
def image_callback(img_msg):
  try:
    cv_image = dbridge.imgmsg_to_cv2(img_msg, "16UC1")
    print(cv_image[pointx])  
  except CvBridgeError as e:
    print(e)  
dbridge = CvBridge()
def depth_data(point):
  global pointx
  pointx = point[1],point[0]
  sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, image_callback)        
class Vision_Engine:

  def __init__(self):
    #self.image_pub = rospy.Publisher("image_topic_2",Image)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback)
    
    self.model = YOLO('yolov8n.pt')
    

      
  def callback(self,data):
    try:
      color_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
      img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
      detect_list = [39,41] # Class of detected objects
      results = self.model.predict(img)
      for r in results:   
        annotator = Annotator(color_frame)
        boxes = r.boxes

        for box in boxes:
            
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls # Class of the Object
            pt1 = (int(b[0].detach().cpu().numpy()),int(b[1].detach().cpu().numpy()))
            x = box.xywh[0][0].detach().cpu().numpy()
            y = box.xywh[0][1].detach().cpu().numpy()
            point = int(x), int(y)
            if c in detect_list:
                
                depth_data(point)
                if cv2.waitKey(1) == ord('s'):
                    try:
                        # thread = threading.Thread(target=qr_scanner(color_frame))
                        # thread.start()
                        color_frame = qr_scanner(color_frame)
                    except:
                        print("QR Error")
                    
                if cv2.waitKey(1) == ord('x'):                   
                    try:
                        thread = threading.Thread(target=solve(color_frame))
                        thread.start()
                        #color_frame = solve(color_frame)
                    except:
                        print("Circle Error")
                if cv2.waitKey(1) == ord('m'):                   
                    try:
                        thread = threading.Thread(target=motion(color_frame))
                        thread.start()
                        #color_frame = solve(color_frame)
                    except:
                        print("Motion Error")        
                      
                # points = dc.Global_points(point[0],point[1]) 
                # depth = points[0][2] # Get Z value
                # D_point = calc_distance(depth_info,x,y,depth) # Get x,y,z values
                D_point = (0,0,0)
                angle = np.abs(90/640*x-45) # Get Angle from the center
                cv2.circle(color_frame, point, 4, (0, 0, 255)) # Center of the camera
                cv2.circle(color_frame, (320,240), 4, (0, 0, 255)) # Center of the detected object
                annotator.box_label(b, self.model.names[int(c)]+" x:"+str(round(D_point[0],2))+" y:"+str(round(D_point[1],2))+" z:"+str(round(D_point[2],2))+ " Angle:"+str(round(angle,2)))

      color_frame = annotator.result()  
      cv2.imshow('YOLO V8 Detection', color_frame)  
    except CvBridgeError as e:
      print(e)
      
def motion(frame):
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame2 = cv2.threshold(frame2, 30, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5),np.uint8)
    frame3 = cv2.morphologyEx(frame2, cv2.MORPH_CLOSE, kernel, iterations=2)
    # frame3 = cv2.erode(frame2, kernel, iterations=2)
    # frame3 = 255 - frame3
    contours, _ = cv2.findContours(frame3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)

        cx = x + w // 2
        cy = y + h // 2

    frame = cv2.circle(frame, (cx, cy), 5, (255, 0, 0), 2)
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)

# Circle Detection
def calculate_line_angle(line_endpoint):
    ydiff = line_endpoint[0][1] - line_endpoint[1][1] # Row diff
    xdiff = line_endpoint[0][0] - line_endpoint[1][0] # Col diff

    angle = int(np.round(np.degrees(np.arctan2(ydiff, -xdiff)))) 

    angle = (angle + 360) % 360

    return angle

def get_position(angle):

    if 80 <= angle and angle <= 100:
        return 'T'
    elif 35 <= angle and angle <= 55:
        return 'TR'
    elif 350 <= angle or angle <= 10:
        return 'R'
    elif 325 >= angle and angle >= 305:
        return 'BR'
    elif 260 <= angle and angle <= 280:
        return 'B'
    elif 215 <= angle and angle <= 235:
        return 'BL'
    elif 170 <= angle and angle <= 190:
        return 'L'
    elif 125 <= angle and angle <= 145:
        return 'TL'
    
    return ''

def yolo_result(image, weights):

    image = cv2.resize(image, (480, 640), interpolation = cv2.INTER_AREA)
    
    model = YOLO(weights)
    results = model.predict(image)

    crops = []
    for r in results:
        boxes = r.boxes
        for box_id, box in enumerate(boxes):
            b = box.xyxy[0].detach().cpu().numpy()
            top_left = (int(b[0]), int(b[1]))
            bottom_right = (int(b[2]), int(b[3]))
            cropped = image[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
            cropped = cv2.resize(cropped, (1024, 1024), interpolation = cv2.INTER_AREA)
            crops.append(cropped)
    
    return crops

def find_circle(cropped_frames):

    list_of_circles = []

    for cropped_frame in cropped_frames:
        temp_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        # temp_frame = cv2.resize(temp_frame, (1024, 1024), interpolation = cv2.INTER_AREA)
        temp_frame = cv2.GaussianBlur(temp_frame, (11, 11), 0)
        _, temp_frame = cv2.threshold(temp_frame, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(temp_frame, cv2.HOUGH_GRADIENT_ALT, 1, minDist=50,
                            param1=10, param2=0.2, minRadius=1, maxRadius=10000000)
        circles = np.round(circles[0, :]).astype("int")
        list_of_circles.append(circles)
    
    return list_of_circles
        

def find_line_points(list_of_frames, list_of_circles):

    dimension = (list_of_frames[0].shape[0], list_of_frames[0].shape[1])
    result = []
    result_circles = []

    for (frame, circle_info) in zip(list_of_frames, list_of_circles):
        
        ret, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_result = []
        frame_result_circle = []
        last_angle = -1
        for circle_ in circle_info:
            (x, y, r) = circle_
            mask = np.zeros(dimension, dtype = frame.dtype)
            mask2 = mask.copy()

            if r <= 20:
                mask = cv2.circle(mask, (x, y), r - 10, 255, 2)
                mask2 = cv2.circle(mask2, (x, y), r + 10, 255, 2)
            else:
                mask = cv2.circle(mask, (x, y), r - 10, 255, 2)
                mask2 = cv2.circle(mask2, (x, y), r + 10, 255, 2)


            mask_result = cv2.bitwise_and(mask, frame)
            mask2_result = cv2.bitwise_and(mask2, frame)
            locations = np.where(mask_result == 255)
            locations2 = np.where(mask2_result == 255)

            avg_y, avg_x = int(np.round(np.mean(locations[0][:]))), int(np.round(np.mean(locations[1][:])))
            avg_y2, avg_x2 = int(np.round(np.mean(locations2[0][:]))), int(np.round(np.mean(locations2[1][:])))

            distance1 = np.hypot(avg_y - y, avg_x - x)
            distance2 = np.hypot(avg_y2 - y, avg_x2 - x)
            avg_pt = None
            if distance2 > distance1:
                avg_pt = (avg_x2, avg_y2)
            else:
                avg_pt = (avg_x, avg_y)
            angle = calculate_line_angle(((x, y), (avg_pt[0], avg_pt[1])))
            if np.abs(last_angle - angle) <= 20:
                continue
            else:
                frame_result.append(((x, y), avg_pt))
                frame_result_circle.append((x, y, r))
                last_angle = angle
        
        result_circles.append(frame_result_circle)
        result.append(frame_result)
    
    return result, result_circles

def draw_shapes(list_of_frames, list_of_circles, list_of_endpoints, list_of_angles):

    result_frames = []
    
    for (frame, circles, endpoints, angles) in zip(list_of_frames, list_of_circles, list_of_endpoints, list_of_angles):
        output_frame = frame.copy()
        
        # Draw circles.
        for circle in circles:
            (x, y, r) = circle
            cv2.circle(output_frame, (x, y), r, (0, 255, 0), 2)
        
        # Draw lines
        for endpoint, angle in zip(endpoints, angles):
            ((x_1, y_1), (x_2, y_2)) = endpoint
            cv2.line(output_frame, (x_1, y_1), (x_2, y_2), (0, 0, 255), 2)
            cv2.putText(output_frame, str(angle), (x_2 + 10, y_2 + 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA, False)
        
        result_frames.append(output_frame)

    return result_frames

def get_angle(line_endpoints):

    result = []
    for endpoints in line_endpoints:
        temp_result = []
        for endpoint in endpoints:
            angle = calculate_line_angle(endpoint)
            temp_result.append(angle)
        result.append(temp_result)
    return result

def get_positions(angles):

    result = []

    for angle_ in angles:
        temp_result = []
        for angle__ in angle_:
            temp_result.append(get_position(angle__))
        result.append(temp_result)
    return result

def stringify(circle_infos, angles, positions):

    result = {
        "crops": []
    }

    for idx, (frame_circles, frame_angles, frame_positions) in enumerate(zip(circle_infos, angles, positions)):
        temp_dict = {}
        temp_dict["crop_no"] = idx
        temp_dict["circles"] = []
        for idx2, (circle, angle, position) in enumerate(zip(frame_circles, frame_angles, frame_positions)):
            temp_dict2 = {}
            temp_dict2["angle"] = str(angle)
            temp_dict2["position"] = str(position)
            temp_dict2["circle_no"] = idx2
            temp_dict["circles"].append(temp_dict2)
        result["crops"].append(temp_dict)
    
    result_str = json.dumps(result, indent=4)
    return result_str

def save_info(save_data):
    f = open('saved_data.txt', 'w')
    f.write(save_data)
    f.close()

    

def solve(frame, weights = 'yolov8n.pt', save_file = False):

    circle_infos = find_circle([frame])
    line_endpoints, circle_infos = find_line_points([frame], circle_infos)
    line_angles = get_angle(line_endpoints)
    cropped_frames_drawn = draw_shapes([frame], circle_infos, line_endpoints, line_angles)
    print(stringify(circle_infos, line_angles, get_positions(line_angles)))
    if save_file:
        save_info(stringify(circle_infos, line_angles, get_positions(line_angles)))

    string1 = stringify(circle_infos, line_angles, get_positions(line_angles))
    # return cropped_frames_drawn[0]
    cv2.imshow('circle', cropped_frames_drawn[0])
    cv2.waitKey(0)  


# QR Scanner
def qr_scanner(image):
    d = cv2.QRCodeDetector()
    val,p,s_q = d.detectAndDecode(image)
    cv2.putText(image, str(val), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)
    return image
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)

def main(args):
  ve = Vision_Engine()
  rospy.init_node('Vision_Engine', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)