from superpoint_pytorch import SuperPoint
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

model = SuperPoint()
weights_path = 'weights\superpoint_v6_from_tf.pth'
model.load_state_dict(torch.load(weights_path))
model.eval()  # Set the model to evaluation mode

img = cv2.imread('cup_ref.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('title2.jpg', cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (640, 640))
img2 = cv2.resize(img2, (640, 640))
# img = cv2.GaussianBlur(img, (5, 5), 0)

# # Apply Canny edge detector
# img = cv2.Canny(img, threshold1=100, threshold2=200)

# img2 = cv2.GaussianBlur(img2, (5, 5), 0)

# # Apply Canny edge detector
# img2 = cv2.Canny(img2, threshold1=100, threshold2=200)
# if img is None:
#     raise ValueError("Image not found")
img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float() / 255.0  # Normalize if necessary
img_tensor2 = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).float() / 255.0  # Normalize if necessary

# Forward pass to get keypoints and descriptors
with torch.no_grad():
    output = model({'image': img_tensor})
    keypoints = output['keypoints'][0]  # Assuming single image (batch size = 1)
    descriptors = output['descriptors'][0]

with torch.no_grad():
    output2 = model({'image': img_tensor2})
    keypoints2 = output2['keypoints'][0]  # Assuming single image (batch size = 1)
    descriptors2 = output2['descriptors'][0]

# Convert PyTorch tensor descriptors to NumPy arrays
desc1_np = descriptors.cpu().numpy()
desc2_np = descriptors2.cpu().numpy()
desc1_np = desc1_np.astype(np.float32)
desc2_np = desc2_np.astype(np.float32)
# Now use the NumPy arrays for matching

# Assuming desc1 and desc2 are descriptors from image 1 and image 2 respectively
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
# Filter matches using the status output from findHomography
good_matches = [m for m, s in zip(matches, status) if s]

# Draw the filtered matches
img_matches = cv2.drawMatches(img, kp1, img2, kp2, good_matches, None, flags=2)

# Display the matches
cv2.imshow('Filtered Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

