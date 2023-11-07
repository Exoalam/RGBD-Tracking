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

model = YOLO('yolov8n.pt')
dc = DepthCamera()
target_dis = 1
hit_map = np.zeros((1000,1000))
detect_list = [39,41,-99]
robot = (0,0,0)
map[robot]['hit'] = 100

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
            if c in detect_list:
                if cv2.waitKey(1) & 0xFF == ord('m'):
                    x = int(input('X: '))
                    y = int(input('Y: '))
                    z = int(input('Z: '))
                    map[robot]['hit'] = 0
                    robot = (x,y,z) 
                    map[robot]['hit'] = 100
                           
                points = dc.Global_points(point[0],point[1])
                map[round(points[0]*100),round(points[1]*100),round(points[2]*100)]['hit'] += 1
                map[round(points[0]*100),round(points[1]*100),round(points[2]*100)]['class'] = c
                cv2.circle(color_frame, point, 4, (0, 0, 255))
                annotator.box_label(b, model.names[int(c)]+" x:"+str(round(points[0],2))+" y:"+str(round(points[1],2))+" z:"+str(round(points[2],2)))
             

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
            hits = map[x]['hit']
            data_actual.append(['class:'+str(cls)+'/'+model.names[int(cls)],' Object: '+str(count), ' Pos: ',x, ' hits: ',hits])
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
# with open('output.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     for row in data:
#         writer.writerow(row)

translation = (1, 2, 3)  
axis = (0, 1, 0)         
angle = np.radians(45)   

rotation_matrix = axangle2aff(axis, angle)
affine_matrix = compose(T=translation, R=rotation_matrix[:3, :3], Z=np.ones(3))

point = np.array([4, 5, 6, 1])
transformed_point = affine_matrix.dot(point)

print(transformed_point[:3])
