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

def filter_xy_pairs_by_depth(xy_pairs, dc):
    # Convert list of xy pairs to a NumPy array for vectorized operations
    xy_pairs_np = xy_pairs

    # Assuming batch_global_points has been modified to return a NumPy array of depths
    # If xs and ys need to be separate for batch_global_points, adjust as necessary
    batch_depths = dc.batch_global_points(xy_pairs_np[:, 0], xy_pairs_np[:, 1])
    
    # Convert batch_depths to a NumPy array if it's not already one
    batch_depths_np = np.array(batch_depths)

    # Filter to keep only those pairs where the corresponding depth is greater than 0
    # Assuming depth information is in the third column ([2]) of batch_depths_np
    filtered_indices = np.where(batch_depths_np[:, 2] > 0)[0]

    # Use filtered indices to select xy pairs
    filtered_xy_pairs_np = xy_pairs_np[filtered_indices]


    return filtered_xy_pairs_np

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
hit_map = np.zeros((1000,1000))
detect_list = [39,41,-99]
datax = []
while True:
    map = np.zeros((1000, 1000, 1000), dtype=custom_dtype)
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
                x = box.xywh[0][0].detach().cpu().numpy()
                y = box.xywh[0][1].detach().cpu().numpy()       
                point = int(x), int(y)
                if c in detect_list:
                    points = dc.Global_points(point[0],point[1])
                    points[0][1] *= -1
                    dx = 500+points[0][0]*100
                    dy = 500+points[0][1]*100
                    map[round(dx),round(dy),round(points[0][2]*100)]['hit'] += 1
                    map[round(dx),round(dy),round(points[0][2]*100)]['class'] = c
                    depth_threshold = 0.05  # Depth threshold to filter out background, adjust based on your requirements

                    top = int(b[1].detach().cpu().numpy())
                    left = int(b[0].detach().cpu().numpy())
                    bottom = int(b[3].detach().cpu().numpy())
                    right = int(b[2].detach().cpu().numpy())
                    # Calculate the average depth of the object
                    # Collect all (x, y) pairs first
                    step_size = 1
                    xs = np.arange(left, right, step_size)
                    ys = np.arange(top, bottom, step_size)

                    # Create a meshgrid from xs and ys
                    X, Y = np.meshgrid(xs, ys)

                    # Stack and reshape to get a list of (x, y) pairs
                    xy_pairs = np.stack([X, Y], axis=-1).reshape(-1, 2)
                    xy_pairs_list = filter_xy_pairs_by_depth(xy_pairs,dc)
                    for index, (x, y) in enumerate(xy_pairs_list):
                        if index % 10 == 0:  
                            cv2.circle(color_frame, (x, y), 1, (0, 0, 255), -1)
                    #cv2.circle(color_frame, point, 4, (0, 0, 255))
                    annotator.box_label(b, model.names[int(c)]+" x:"+str(round(points[0][0],2))+" y:"+str(round(points[0][1],2))+" z:"+str(round(points[0][2],2)))
                

        color_frame = annotator.result()  
        cv2.imshow('Detection', color_frame)     
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break  

    map[500,500,0]['hit'] = 0
    indices = np.where(map['hit'] > 1)
    points = list(zip(indices[0], indices[1], indices[2]))
    points = max_hit(points)
    print(points)
    for i, point in enumerate(points):
        datax.append([point[0]-500,point[1]-500,point[2],map[point]['hit']])

    if input() == 'x':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs, ys, zs, hits = zip(*datax)
        ax.scatter(xs, ys, zs, s=hits)
        plt.show()
        
        x1, y1, z1, h1= datax[0]
        x2, y2, z2, h2 = datax[1]
        translation = (-x2, -y2, -z2)  
        axis = (0, 1, 0)         
        angle = np.radians(0)   

        rotation_matrix = axangle2aff(axis, angle)
        affine_matrix = compose(T=translation, R=rotation_matrix[:3, :3], Z=np.ones(3))

        point = np.array([x1, y1, z1, 1])
        transformed_point = affine_matrix.dot(point)

        print(transformed_point[:3])
        break
    else:
        continue


