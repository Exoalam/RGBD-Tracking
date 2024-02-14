from superpoint_pytorch import SuperPoint
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

camera = cv2.VideoCapture(0)
model = SuperPoint()
weights_path = 'weights\superpoint_v6_from_tf.pth'
model.load_state_dict(torch.load(weights_path))
model.eval()  # Set the model to evaluation mode
img = cv2.imread('bottle_ref.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.GaussianBlur(img, (5, 5), 0)
img = cv2.Canny(img, threshold1=100, threshold2=200)
img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float() / 255.0  
with torch.no_grad():
    output = model({'image': img_tensor})
    keypoints = output['keypoints'][0]  # Assuming single image (batch size = 1)
    descriptors = output['descriptors'][0]
desc1_np = descriptors.cpu().numpy()
desc1_np = desc1_np.astype(np.float32)

while(True):
    ret, frame = camera.read()
    img2 = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.GaussianBlur(img2, (5, 5), 0)
    img2 = cv2.Canny(img2, threshold1=100, threshold2=200)
    img_tensor2 = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).float() / 255.0  
    with torch.no_grad():
        output2 = model({'image': img_tensor2})
        keypoints2 = output2['keypoints'][0]  # Assuming single image (batch size = 1)
        descriptors2 = output2['descriptors'][0]

    # Convert PyTorch tensor descriptors to NumPy arrays
    desc2_np = descriptors2.cpu().numpy()
    desc2_np = desc2_np.astype(np.float32)
    # Now use the NumPy arrays for matching

    # Assuming desc1 and desc2 are descriptors from image 1 and image 2 respectively
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(desc1_np, desc2_np)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 4:
        src_pts = np.float32([keypoints[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx] for m in matches]).reshape(-1, 1, 2)

        # Find homography matrix
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Assuming planar motion, extract the rotation angle
        theta = -np.arctan2(H[0,1], H[0,0]) * (180 / np.pi)

        print(theta)
    else:
        print("Not enough matches are found - {}/{}".format(len(matches), 4))
        H = None
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
  
camera.release() 

cv2.destroyAllWindows() 