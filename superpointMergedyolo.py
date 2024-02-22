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
from superpoint_pytorch import SuperPoint
import torch

custom_dtype = np.dtype([
    ('hit', np.int8),       
    ('accuracy', np.int8),
    ('class', np.int8)       
])


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

model = SuperPoint()
weights_path = 'weights/superpoint_v6_from_tf.pth'
model.load_state_dict(torch.load(weights_path))
model.eval()  # Set the model to evaluation mode
img = cv2.imread('cup_ref.jpg', cv2.IMREAD_GRAYSCALE)
img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float() / 255.0
with torch.no_grad():
    output = model({'image': img_tensor})
    keypoints = output['keypoints'][0]  # Assuming single image (batch size = 1)
    descriptors = output['descriptors'][0]

desc1_np = descriptors.cpu().numpy()
desc1_np = desc1_np.astype(np.float32)

while True:
    map = np.zeros((1000, 1000, 1000), dtype=custom_dtype)
    while True: 
        ret, depth_frame, color_frame, depth_info = dc.get_frame()
        img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        results = model.predict(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
                    img_tensor2 = torch.from_numpy(img_gray).unsqueeze(0).unsqueeze(0).float() / 255.0
                    with torch.no_grad():
                        output2 = model({'image': img_tensor2})
                        keypoints2 = output2['keypoints'][0]  # Assuming single image (batch size = 1)
                        descriptors2 = output2['descriptors'][0]
                    desc2_np = descriptors2.cpu().numpy()
                    desc2_np = desc2_np.astype(np.float32)
                    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
                    matches = bf.match(desc1_np, desc2_np)
                    matches = sorted(matches, key=lambda x: x.distance)

                    # Convert keypoints tensors to numpy arrays
                    keypoints_np = keypoints.cpu().detach().numpy()
                    keypoints2_np = keypoints2.cpu().detach().numpy()

                    # # Convert numpy arrays of keypoints to lists of cv2.KeyPoint objects
                    keypoints_cv = [cv2.KeyPoint(float(kp[0]), float(kp[1]), 1) for kp in keypoints_np]
                    keypoints2_cv = [cv2.KeyPoint(float(kp[0]), float(kp[1]), 1) for kp in keypoints2_np]


                    kp1 = [cv2.KeyPoint(kp[0], kp[1], 1) for kp in keypoints_np]
                    kp2 = [cv2.KeyPoint(kp[0], kp[1], 1) for kp in keypoints2_np]

                    # Convert the matches to point arrays for findHomography
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                    # Find homography matrix using RANSAC
                    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
                    theta = -np.arctan2(H[0,1], H[0,0]) * (180 / np.pi)

                    print(theta)
                    if cv2.waitKey(1) & 0xFF == ord('m'):
                        x = int(input('X: '))
                        y = int(input('Y: '))
                        z = int(input('Z: '))
                        map[robot]['hit'] = 0
                        robot = (x,y,z) 
                        map[robot]['hit'] = 100
                            
                    points = dc.Global_points(point[0],point[1])
                    points[0][1] *= -1
                    dx = 500+points[0][0]*100
                    dy = 500+points[0][1]*100
                    map[round(dx),round(dy),round(points[0][2]*100)]['hit'] += 1
                    map[round(dx),round(dy),round(points[0][2]*100)]['class'] = c
                    cv2.circle(color_frame, point, 4, (0, 0, 255))
                    annotator.box_label(b, model.names[int(c)]+ " " + theta + " "+" x:"+str(round(points[0][0],2))+" y:"+str(round(points[0][1],2))+" z:"+str(round(points[0][2],2)))
                

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


