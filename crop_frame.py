from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import numpy as np

model = YOLO('yolov8n.pt')  # Ensure the model path is correct
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
reference_image = cv2.imread('bottle_ref.jpg')
reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2GRAY)
#target_image = cv2.imread('test/test_target.png')

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# Detect keypoints and compute descriptors
kp1, des1 = orb.detectAndCompute(reference_image, None)
# Example additions to the video processing loop
frame_counter = 0
n_skip_frames = 5  # Process every 5th frame for example

while True:
    ret, frame = cap.read()
    if not ret or frame_counter % n_skip_frames != 0:
        frame_counter += 1
        continue  # Skip this frame
    
    # Your existing processing logic here...

    frame_counter += 1

    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.predict(img)

    annotator = Annotator(frame)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            if c == 39:  # Check if the detected class is 41
                annotator.box_label(b, model.names[int(c)])
                # Crop the frame based on the bounding box
                # Note: OpenCV uses (x, y, w, h) format for cropping
                x1, y1, x2, y2 = map(int, b)
                cropped_frame = frame[y1:y2, x1:x2]
                cv2.imshow(f'Cropped - Class {int(c)}', cropped_frame)  # Display cropped frame
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
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
                    annotator.box_label(b, model.names[int(c)]+" x:"+str(angle_deg))
                    print(f"Rotation Angle (degrees): {angle_deg}")
                else:
                    print("Homography matrix not found.")
    frame = annotator.result()
    cv2.imshow('YOLO V8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
