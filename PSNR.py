import os
from PIL import Image
import numpy as np
import math

def calculate_psnr(img1, img2):
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return float('inf')  # Identical images
    max_pixel = 255.0
    psnr = 30 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

# Paths
baseline_dir = 'SDG Depth Three'
generated_dir = 'denoised evolved'
image_size = (256, 256)

all_psnr_values = []
class_psnr_dict = {}

# Get common subfolders (classes)
classes = sorted(os.listdir(baseline_dir))
print(f"Found classes: {classes}\n")

for class_name in classes:
    base_path = os.path.join(baseline_dir, class_name)
    gen_path = os.path.join(generated_dir, class_name)

    base_images = sorted(os.listdir(base_path))
    gen_images = sorted(os.listdir(gen_path))

    class_psnrs = []

    print(f"Processing class: {class_name}")
    
    for base_img_name, gen_img_name in zip(base_images, gen_images):
        base_img_path = os.path.join(base_path, base_img_name)
        gen_img_path = os.path.join(gen_path, gen_img_name)

        try:
            img_base = Image.open(base_img_path).convert('RGB').resize(image_size)
            img_gen = Image.open(gen_img_path).convert('RGB').resize(image_size)

            psnr = calculate_psnr(img_base, img_gen)
            class_psnrs.append(psnr)
            all_psnr_values.append(psnr)

        except Exception as e:
            print(f"Error processing {base_img_name} vs {gen_img_name}: {e}")

    avg_psnr = np.mean(class_psnrs)
    class_psnr_dict[class_name] = avg_psnr
    print(f"[{class_name}] Average PSNR: {avg_psnr:.2f} dB\n")

overall_psnr = np.mean(all_psnr_values)
print("="*40)
print("Class-wise PSNR Results:")
for cls, val in class_psnr_dict.items():
    print(f" - {cls}: {val:.2f} dB")

print("="*40)
print(f"✅ Overall Average PSNR: {overall_psnr:.2f} dB")
