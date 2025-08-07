import os
import cv2
from tqdm import tqdm

# Function to apply OpenCV's denoising
def opencv_denoise(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    
    # Use OpenCV's Non-Local Means denoising
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised_image

# Function to process all images in a folder
def denoise_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_paths = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_path in tqdm(image_paths):
        # Denoise the image using OpenCV
        denoised_image = opencv_denoise(image_path)
        
        # Convert to the correct format and save the image
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        denoised_image_bgr = cv2.cvtColor(denoised_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
        cv2.imwrite(output_path, denoised_image_bgr)  # Save the denoised image
    
    print(f"Processed {len(image_paths)} images and saved to {output_folder}")

# Example usage
input_folder = 'non evolved/Fear'  # Path to folder with noisy images
output_folder = 'denoised non evolved/Fear'  # Path where want the cleaned images saved

denoise_folder(input_folder, output_folder)
