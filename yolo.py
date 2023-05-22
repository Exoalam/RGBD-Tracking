from ultralytics import YOLO
import cv2
import numpy
from ultralytics.yolo.utils.plotting import Annotator
import pyrealsense2
from realsense_depth import *

model = YOLO('yolov8n.pt')
dc = DepthCamera()
# cap = cv2.VideoCapture(4)
# cap.set(3, 640)
# cap.set(4, 480)

while True:

    ret, depth_frame, color_frame = dc.get_frame()
    img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)

    results = model.predict(img)

    for r in results:
        
        annotator = Annotator(color_frame)
        
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            x = box.xywh[0][0]
            y = box.xywh[0][1]
            y = y.detach().cpu().numpy()
            x = x.detach().cpu().numpy()
            point = int(x), int(y)
            if c == 41:
                distance = depth_frame[point[1], point[0]]
                distance = numpy.abs(distance*numpy.cos((45/320)*numpy.abs(int(x)-320)))
                x = int(x)-320
                y = 240-int(y)
                x = x*numpy.sin((45/320)*numpy.abs(int(x)-320))
                y = y*numpy.sin((30/240)*numpy.abs(240-int(y)))
                cv2.circle(color_frame, point, 4, (0, 0, 255))
                annotator.box_label(b, model.names[int(c)]+" x:"+str(int(x))+" y:"+str(int(y))+" z:"+str(int(distance)))
          
    color_frame = annotator.result()  
    cv2.imshow('YOLO V8 Detection', color_frame)     
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
