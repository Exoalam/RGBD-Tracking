import numpy as np
from transforms3d.axangles import axangle2aff
from transforms3d.affines import compose
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
from realsense_depth import *
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


model = YOLO('yolov8n.pt')
dc = DepthCamera()
hit_map = np.zeros((1000,1000))
detect_list = [39,41,-99]
datax = []
while True:
    while True: 
        ret, depth_frame, color_frame, depth_info = dc.get_frame()
        img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        results = model.predict(img)
        for r in results:
            
            annotator = Annotator(color_frame)
            
            boxes = r.boxes
            for box in boxes:
                
                b = box.xyxy[0] 
                c = box.cls
                pt1 = (int(b[0].detach().cpu().numpy()),int(b[1].detach().cpu().numpy()))
                x = box.xywh[0][0].detach().cpu().numpy()
                y = box.xywh[0][1].detach().cpu().numpy()       
                point = int(x), int(y)
                if c in detect_list:
                            
                    points = dc.Global_points(point[0],point[1])
                    cv2.circle(color_frame, point, 4, (0, 0, 255))
                    annotator.box_label(b, model.names[int(c)]+" x:"+str(round(points[0][0],2))+" y:"+str(round(points[0][1],2))+" z:"+str(round(points[0][2],2)))
                

        color_frame = annotator.result()  
        cv2.imshow('Detection', color_frame)     
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break  


