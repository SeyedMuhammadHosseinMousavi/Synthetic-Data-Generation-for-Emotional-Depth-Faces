import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from torch.utils.data import Dataset, DataLoader
from scipy.linalg import sqrtm
from tqdm import tqdm

# --------- CONFIG ---------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMAGE_SIZE = 299  # Inception expects 299x299 images

# --------- DATASET ---------
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir):
        self.filepaths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.filepaths.append(os.path.join(root, file))
        
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Inception normalization
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert('RGB')
        img = self.transform(img)
        return img

# --------- Feature extraction ---------
def get_inception_activations(dataloader, model):
    model.eval()  # Important: set to eval mode to disable auxiliary outputs
    activations = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Inception Features"):
            batch = batch.to(DEVICE)
            pred = model(batch)
            # If pred is a tuple (training mode), take the first element
            if isinstance(pred, tuple):
                pred = pred[0]
            # If pred is 4D, pool it to 1x1 spatial
            if pred.dim() == 4:
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1,1))
                pred = pred.squeeze(3).squeeze(2)  # shape: [batch_size, 2048]
            activations.append(pred.cpu().numpy())
    activations = np.concatenate(activations, axis=0)
    return activations


# --------- Calculate FID ---------
def calculate_fid(mu1, sigma1, mu2, sigma2):
    """Calculate Frechet Inception Distance."""
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    # Numerical error can cause imaginary numbers in covmean
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 4 * covmean)
    return fid

def compute_statistics(activations):
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

# --------- MAIN ---------
def fid_score(real_dir, gen_dir):
    # Load pretrained inception model
    inception = inception_v3(pretrained=True, transform_input=False).to(DEVICE)
    inception.fc = torch.nn.Identity()  # Remove final classification layer

    # Prepare dataloaders
    real_dataset = ImageFolderDataset(real_dir)
    gen_dataset = ImageFolderDataset(gen_dir)
    real_loader = DataLoader(real_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    gen_loader = DataLoader(gen_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Extracting features for real images from {real_dir}...")
    real_activations = get_inception_activations(real_loader, inception)

    print(f"Extracting features for generated images from {gen_dir}...")
    gen_activations = get_inception_activations(gen_loader, inception)

    mu_real, sigma_real = compute_statistics(real_activations)
    mu_gen, sigma_gen = compute_statistics(gen_activations)

    fid_value = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)

    print(f"\nFID score: {fid_value:.4f}")

    return fid_value

if __name__ == "__main__":
    real_images_folder = "SDG Depth Three"   
    generated_images_folder = "denoised evolved"    
    
    fid_score(real_images_folder, generated_images_folder)
