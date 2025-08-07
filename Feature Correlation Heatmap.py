import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern, hog
from skimage.filters import sobel
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# ---- Configuration ----
DATASET_PATH = "evolved and baseline"
IMG_SIZE = (256, 256)
LBP_POINTS = 24
LBP_RADIUS = 3
LABELS = {'Fear': 0, 'Happiness': 1, 'Neutral': 2}

# ---- Prepare feature lists ----
hog_features_all = []
lbp_features_all = []
sobel_features_all = []
intensity_features_all = []

# ---- Load and extract features ----
for class_name in LABELS.keys():
    folder = os.path.join(DATASET_PATH, class_name)
    for fname in os.listdir(folder):
        if fname.endswith(".png"):
            img_path = os.path.join(folder, fname)
            print(f"Loading image: {img_path}")  

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            gray = cv2.resize(img, IMG_SIZE)

            # HOG
            hog_feat = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
            hog_features_all.append(hog_feat)

            # LBP
            lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
            lbp_hist = lbp_hist.astype("float")
            lbp_hist /= (lbp_hist.sum() + 1e-6)
            lbp_features_all.append(lbp_hist)

            # Sobel
            sobel_edges = sobel(gray)
            sobel_hist, _ = np.histogram(sobel_edges.ravel(), bins=32, range=(0, 1))
            sobel_hist = sobel_hist.astype("float")
            sobel_hist /= (sobel_hist.sum() + 1e-6)
            sobel_features_all.append(sobel_hist)

            # Intensity Histogram
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)
            intensity_features_all.append(hist)


# ---- Convert and standardize ----
hog_arr = StandardScaler().fit_transform(np.array(hog_features_all))
lbp_arr = StandardScaler().fit_transform(np.array(lbp_features_all))
sobel_arr = StandardScaler().fit_transform(np.array(sobel_features_all))
intensity_arr = StandardScaler().fit_transform(np.array(intensity_features_all))

# ---- Aggregate by mean for correlation ----
hog_mean = np.mean(hog_arr, axis=1)
lbp_mean = np.mean(lbp_arr, axis=1)
sobel_mean = np.mean(sobel_arr, axis=1)
intensity_mean = np.mean(intensity_arr, axis=1)

# ---- Correlation DataFrame ----
df = pd.DataFrame({
    'HOG': hog_mean,
    'LBP': lbp_mean,
    'Sobel Edge': sobel_mean,
    'Intensity Histogram': intensity_mean
})
corr = df.corr()

# ---- Plot heatmap ----
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Purples", square=True, cbar=True)
plt.title("Correlation Heatmap of Feature Types", fontsize=16, fontweight='bold')
plt.tight_layout()

# import ace_tools as tools; tools.display_dataframe_to_user(name="Feature Type Correlation", dataframe=corr)
plt.savefig("feature_correlation_heatmap.png", dpi=1200, bbox_inches='tight')
plt.show()
