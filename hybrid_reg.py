from ultralytics import YOLO
import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator
from realsense_depth import *
from scipy.spatial import distance
import csv
from serial_connector_1 import *

model = YOLO('best.pt')
dc = DepthCamera()
ser_con = SerialConnector()
ser_con.connect('/dev/ttyUSB0')

detect_list = [0]
Train_data = []
while True:
    
    ret, depth_frame, color_frame, depth_info = dc.get_frame()
    img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    img_shape = img.shape
    results = model.predict(img)
    for r in results:
        
        annotator = Annotator(color_frame)
        
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0] 
            _b = box.xywh[0].detach().cpu().numpy()
            c = box.cls
            x = box.xywh[0][0]
            y = box.xywh[0][1]
            w = box.xywh[0][2].detach().cpu().numpy()
            h = box.xywh[0][3].detach().cpu().numpy()
            _c = box.conf.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            x = x.detach().cpu().numpy()          
            point = int(x), int(y)
            if c in detect_list:    
                points = dc.Global_points(point[0],point[1])
                dis = dc.actual_depth(point[0],point[1])
                distance = np.sqrt(points[0]**2+points[1]**2+points[2]**2)
                cor = ser_con.get_orientation()
                Train_data.append([distance,cor[0],cor[1],cor[2],h,w])
                annotator.box_label(b, model.names[int(c)]+" x:"+str(round(points[0][0],2))+" y:"+str(round(points[0][1],2))+" z:"+str(round(points[0][2],2)))
                
                    

    color_frame = annotator.result()  
    cv2.imshow('YOLO V8 Detection', color_frame)     
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break  


with open('hybrid_reg.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for row in Train_data:
        writer.writerow([row[0],row[1],row[2],row[3]])
