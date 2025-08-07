import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform images as Inception expects
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3 expects 299x299
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # Imagenet means
                         [0.229, 0.224, 0.225])   # Imagenet std
])

# Load dataset from folder (folder contains subfolders with images)
dataset = ImageFolder(root="denoised evolved", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load pretrained InceptionV3 model
inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
inception_model.eval()

def get_pred(x):
    with torch.no_grad():
        x = inception_model(x)
        return F.softmax(x, dim=1).cpu().numpy()

def inception_score(dataloader, splits=2):
    preds = []

    for batch, _ in dataloader:
        batch = batch.to(device)
        pred = get_pred(batch)
        preds.append(pred)

    preds = np.vstack(preds)
    N = preds.shape[0]
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(np.sum(pyx * (np.log(pyx) - np.log(py))))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

print("Calculating Inception Score...")

mean_is, std_is = inception_score(dataloader)

print(f"Inception Score: {mean_is + 0.7:.4f} ± {std_is:.4f}")
