import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern, hog
from skimage.filters import sobel
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ---- Configuration ----
DATASET_PATH = "SDG Depth Three"
IMG_SIZE = (256, 256)
LBP_POINTS = 24
LBP_RADIUS = 3
LABELS = {'Fear': 0, 'Happiness': 1, 'Neutral': 2}
COLORS = ['#e41a1c', '#377eb8', '#4daf4a']  # Red, Blue, Green

# ---- Feature extraction ----
def extract_features(image):
    gray = cv2.resize(image, IMG_SIZE)

    # HOG
    hog_features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

    # LBP
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # Sobel
    sobel_edges = sobel(gray)
    sobel_hist, _ = np.histogram(sobel_edges.ravel(), bins=64, range=(0, 1))
    sobel_hist = sobel_hist.astype("float")
    sobel_hist /= (sobel_hist.sum() + 1e-6)

    # Intensity
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
    hist /= (hist.sum() + 1e-6)

    return np.concatenate([hog_features, lbp_hist, sobel_hist, hist])

# ---- Load Dataset ----
X_all, y_all = [], []
for class_name, label in LABELS.items():
    folder = os.path.join(DATASET_PATH, class_name)
    for fname in tqdm(os.listdir(folder), desc=f"Loading " + class_name):
        if fname.endswith(".png"):
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                features = extract_features(img)
                X_all.append(features)
                y_all.append(label)

X_all = np.array(X_all)
y_all = np.array(y_all)

# ---- Standardize ----
X_scaled = StandardScaler().fit_transform(X_all)

# ---- Apply t-SNE (2D)
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# ---- Build DataFrame for Plotting
df = pd.DataFrame()
df["Dim1"] = X_tsne[:, 0]
df["Dim2"] = X_tsne[:, 1]
df["Emotion"] = [list(LABELS.keys())[list(LABELS.values()).index(lbl)] for lbl in y_all]

# ---- Plot t-SNE Result
plt.figure(figsize=(10, 8))
sns.set(style="whitegrid", font_scale=1.4)

sns.scatterplot(
    data=df,
    x="Dim1", y="Dim2",
    hue="Emotion",
    palette=COLORS,
    s=100,
    edgecolor='k'
)

plt.title("t-SNE Projection of Emotion Features", fontsize=20, fontweight='bold')
plt.xlabel("t-SNE Dimension 1", fontsize=14)
plt.ylabel("t-SNE Dimension 2", fontsize=14)
plt.legend(title="Emotion", fontsize=12, title_fontsize=13, loc='best', frameon=True)
plt.tight_layout()
plt.savefig("tsne_emotions.png", dpi=600, bbox_inches='tight')
plt.show()
