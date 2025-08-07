import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern, hog
from skimage.filters import sobel
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ---- Configuration ----
DATASET_PATH = "evolved and baseline"

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

# ---- Apply LDA ----
lda = LDA(n_components=1)
X_lda1 = lda.fit_transform(X_scaled, y_all).flatten()

# ---- Create DataFrame ----
df = pd.DataFrame()
df["LDA1"] = X_lda1
df["Emotion"] = [list(LABELS.keys())[list(LABELS.values()).index(lbl)] for lbl in y_all]

# ---- Plot KDE of LDA1 per Emotion ----
plt.figure(figsize=(7, 6))
sns.set(style="whitegrid", font_scale=1.4)

for i, emotion in enumerate(LABELS.keys()):
    subset = df[df["Emotion"] == emotion]
    sns.kdeplot(
        data=subset,
        x="LDA1",
        fill=True,
        alpha=0.5,
        linewidth=2.5,
        label=emotion,
        color=COLORS[i],
        bw_adjust=1.0
    )

plt.title("LDA1 KDE Distribution per Emotion", fontsize=18, fontweight='bold')
plt.xlabel("LDA Component 1", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(title="Class", fontsize=12, title_fontsize=13, loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig("LDA1_KDE_Per_Emotion.png", dpi=1200, bbox_inches='tight')
plt.show()
