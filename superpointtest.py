from superpoint_pytorch import SuperPoint
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

model = SuperPoint()
weights_path = 'weights\superpoint_v6_from_tf.pth'
model.load_state_dict(torch.load(weights_path))
model.eval()  # Set the model to evaluation mode

img = cv2.imread('bottle_ref.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('bottle_ref - Copy.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Image not found")
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

# # # Display keypoints on the image
# for point in keypoints2:
#     x, y = map(int, point)
#     cv2.circle(img_resized2, (x, y), 1, (0, 255, 0), -1)

# cv2.imshow('Keypoints', img_resized2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
    
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

N = 10  # You can adjust N to inspect more matches
good_matches = matches[:N]  # Assuming 'matches' is a list of cv2.DMatch objects and is sorted by distance

# Convert keypoints tensors to numpy arrays
keypoints_np = keypoints.cpu().detach().numpy()
keypoints2_np = keypoints2.cpu().detach().numpy()

# Convert numpy arrays of keypoints to lists of cv2.KeyPoint objects
keypoints_cv = [cv2.KeyPoint(float(kp[0]), float(kp[1]), 1) for kp in keypoints_np]
keypoints2_cv = [cv2.KeyPoint(float(kp[0]), float(kp[1]), 1) for kp in keypoints2_np]

# Now you can draw the matches
# Make sure the images are read correctly before this step
img_matches = cv2.drawMatches(img, keypoints_cv, img2, keypoints2_cv, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matches
cv2.imshow("Matches", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()


