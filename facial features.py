import cv2
import numpy as np
import os
from tqdm import tqdm
from imutils import face_utils
import dlib

import cv2
import dlib
import numpy as np

# Load the pre-trained dlib face detector and facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to enhance facial features in the depth image
def enhance_facial_features(depth_image_path):
    # Load the depth image (assuming grayscale depth image)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)
    depth_image_rgb = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)  # Convert to RGB for visualization

    # Detect faces in the image
    gray = cv2.cvtColor(depth_image_rgb, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Get landmarks for the detected face
        landmarks = predictor(gray, face)

        # Convert landmarks to numpy array
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Enhance Nose
        nose_tip = landmarks[30]  # Nose tip
        cv2.circle(depth_image_rgb, tuple(nose_tip), 7, (0, 255, 0), -1)  # Larger circle for enhancement
        # Add more depth to the nose region
        cv2.line(depth_image_rgb, tuple(landmarks[27]), tuple(nose_tip), (0, 255, 0), 2)

        # Enhance Eyes (left and right)
        left_eye = landmarks[36:42]  # Left eye landmarks
        right_eye = landmarks[42:48]  # Right eye landmarks
        for (x, y) in left_eye:
            cv2.circle(depth_image_rgb, (x, y), 4, (255, 0, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(depth_image_rgb, (x, y), 4, (255, 0, 0), -1)

        # Enhance Mouth (larger mouth area and lips)
        for i in range(48, 68):
            cv2.circle(depth_image_rgb, tuple(landmarks[i]), 3, (0, 0, 255), -1)

        # Add synthetic shading to enhance the features (darker regions for shadow effect)
        for i in range(36, 48):  # Add shadow around the eyes
            cv2.circle(depth_image_rgb, tuple(landmarks[i]), 5, (0, 0, 0), 2)  # Black circle for shadow

        # Apply enhanced depth to simulate volume in features (nose, eyes, mouth)
        nose_region = depth_image[nose_tip[1] - 15:nose_tip[1] + 15, nose_tip[0] - 15:nose_tip[0] + 15]
        if nose_region.size > 0:
            depth_image[nose_tip[1] - 15:nose_tip[1] + 15, nose_tip[0] - 15:nose_tip[0] + 15] = \
                np.clip(nose_region + 15, 0, 255)

    return depth_image_rgb

# Function to process a folder of images
def process_depth_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith(('.png', '.jpg', '.jpeg'))]

    for image_path in image_paths:
        # Enhance facial features in the depth image
        enhanced_image = enhance_facial_features(image_path)

        # Save the enhanced image
        output_path = os.path.join(output_folder, f"enhanced_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, enhanced_image)
        print(f"Processed {image_path}, saved to {output_path}")


# Example usage
input_folder = 'evolved/Fear'  # Folder with depth images (2D depth images)
output_folder = 'facial features evolved/Fear'  # Folder to save images with added features

process_depth_folder(input_folder, output_folder)
