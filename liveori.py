import cv2
import numpy as np

camera = cv2.VideoCapture(0)
reference_image = cv2.imread('sour.jpg')
#target_image = cv2.imread('test/test_target.png')

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# Detect keypoints and compute descriptors
kp1, des1 = orb.detectAndCompute(reference_image, None)

while(True):
    ret, frame = camera.read()

    cv2.imshow('frame', frame)
    kp2, des2 = orb.detectAndCompute(frame, None)

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
        font = cv2.FONT_HERSHEY_SIMPLEX 
  
        # org 
        org = (frame.shape[0]//2, frame.shape[1]//2) 
        
        # fontScale 
        fontScale = 1
        
        # Blue color in BGR 
        color = (255, 0, 0) 
        
        # Line thickness of 2 px 
        thickness = 2
        
        # Using cv2.putText() method 
        frame = cv2.putText(frame, f"Rotation Angle (degrees): {angle_deg}", org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 
        print(f"Rotation Angle (degrees): {angle_deg}")
    else:
        print("Homography matrix not found.")
        frame = cv2.putText(frame, "Homography matrix not found.", org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
camera.release() 

cv2.destroyAllWindows() 