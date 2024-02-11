import cv2
import numpy as np
# Load the images
reference_image = cv2.imread('test/test_source.png')
target_image = cv2.imread('test/test_target.png')

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# Detect keypoints and compute descriptors
kp1, des1 = orb.detectAndCompute(reference_image, None)
kp2, des2 = orb.detectAndCompute(target_image, None)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort matches by distance (best first)
matches = sorted(matches, key=lambda x:x.distance)

# Extract location of good matches
points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Find homography
H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

if H is not None:
    # Assuming a pure rotation, extract the rotation part of the homography matrix
    # This simplification works under the assumption of no skew and uniform scale
    cos_theta = H[0, 0]
    sin_theta = H[1, 0]

    # Calculate the rotation angle in radians
    angle_rad = np.arctan2(sin_theta, cos_theta)

    # Convert angle from radians to degrees
    angle_deg = np.degrees(angle_rad)

    print(f"Rotation Angle (degrees): {angle_deg}")
else:
    print("Homography matrix not found.")