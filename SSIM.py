import os
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

# Config
BASELINE_DIR = "SDG Depth Three"
SYNTH_DIR = "denoised evolved"
LABELS = {'Fear': 0, 'Happiness': 1, 'Neutral': 2}
IMG_SIZE = (256, 256)  # width, height

def calculate_ssim(img1, img2):
    # img1 and img2 should be grayscale numpy arrays
    return ssim(img1, img2, data_range=img2.max() - img2.min())

def load_and_process_image(path):
    img = Image.open(path).convert("L")  # convert to grayscale
    img = img.resize(IMG_SIZE)
    img_np = np.array(img)
    return img_np

def main():
    print("========================================")
    print("Starting SSIM evaluation...")
    print("========================================")

    all_scores = []
    print("Class-wise SSIM Results:")
    for cls in LABELS.keys():
        baseline_class_dir = os.path.join(BASELINE_DIR, cls)
        synth_class_dir = os.path.join(SYNTH_DIR, cls)

        if not os.path.isdir(baseline_class_dir):
            print(f"[❌] Missing class directory: {baseline_class_dir}")
            continue
        if not os.path.isdir(synth_class_dir):
            print(f"[❌] Missing class directory: {synth_class_dir}")
            continue

        baseline_images = sorted(os.listdir(baseline_class_dir))
        synth_images = sorted(os.listdir(synth_class_dir))

        # We assume no matched filenames, so compare in order
        min_len = min(len(baseline_images), len(synth_images))
        if min_len == 0:
            print(f"[❌] No images found for class: {cls}")
            continue

        class_ssim_scores = []
        for i in range(min_len):
            baseline_path = os.path.join(baseline_class_dir, baseline_images[i])
            synth_path = os.path.join(synth_class_dir, synth_images[i])

            img1 = load_and_process_image(baseline_path)
            img2 = load_and_process_image(synth_path)

            if img1.shape != img2.shape:
                print(f"[❌] Shape mismatch at index {i} in class '{cls}', skipping pair.")
                continue

            score = calculate_ssim(img1, img2)
            class_ssim_scores.append(score)

        if class_ssim_scores:
            avg_class_ssim = np.mean(class_ssim_scores)
            print(f" - {cls}: {avg_class_ssim:.2f} SSIM")
            all_scores.extend(class_ssim_scores)
        else:
            print(f"[❌] No valid SSIM scores for class: {cls}")

    if all_scores:
        avg_ssim = np.mean(all_scores)
        print("========================================")
        print(f"✅ Overall Average SSIM: {avg_ssim:.2f}")
    else:
        print("❌ No valid SSIM results found.")

if __name__ == "__main__":
    main()
