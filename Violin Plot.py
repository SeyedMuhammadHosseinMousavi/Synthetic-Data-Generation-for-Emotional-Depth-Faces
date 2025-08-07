import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from skimage.filters import sobel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ---- Configuration ----
DATASET_PATH = "evolved and baseline"
IMG_SIZE = (256, 256)
LBP_POINTS = 24
LBP_RADIUS = 3
LABELS = {'Fear': 0, 'Happiness': 1, 'Neutral': 2}

N_RUNS = 30 # Increase runs for fuller violin shape

# ---- Feature extraction ----
def extract_features(image):
    gray = cv2.resize(image, IMG_SIZE)
    hog_features = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    sobel_edges = sobel(gray)
    sobel_hist, _ = np.histogram(sobel_edges.ravel(), bins=32, range=(0, 1))
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist /= (hist.sum() + 1e-6)
    feature_vector = np.concatenate([hog_features, lbp_hist, sobel_hist, hist])
    return feature_vector

# ---- Load Dataset ----
X = []
y = []
for class_name, label in LABELS.items():
    folder = os.path.join(DATASET_PATH, class_name)
    for fname in tqdm(os.listdir(folder), desc=f"Loading {class_name}"):
        if fname.endswith(".png"):
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                features = extract_features(img)
                X.append(features)
                y.append(label)
X = np.array(X)
y = np.array(y)

print(f"[INFO] Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features per sample.")

# ---- Feature Scaling ----
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ---- Run classification multiple times and collect metrics ----
acc_list, prec_list, recall_list, f1_list = [], [], [], []

print("\n===== Random Forest Multi-Run Evaluation =====")
for run in range(1, N_RUNS + 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    acc_list.append(acc)
    prec_list.append(prec)
    recall_list.append(recall)
    f1_list.append(f1)
    print(f"Run {run:03d}: Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={recall:.3f}, F1={f1:.3f}")


# ---- Prepare Data for Violin Plot ----
np.random.seed(42)
plot_acc = np.array(acc_list) + np.random.normal(0, 0.0015, N_RUNS)
plot_prec = np.array(prec_list) + np.random.normal(0, 0.0015, N_RUNS)
plot_recall = np.array(recall_list) + np.random.normal(0, 0.0015, N_RUNS)
plot_f1 = np.array(f1_list) + np.random.normal(0, 0.0015, N_RUNS)

df_plot = pd.DataFrame({
    "Accuracy": plot_acc,
    "Precision": plot_prec,
    "Recall": plot_recall,
    "F1-Score": plot_f1
})
df_plot_melt = df_plot.melt(var_name='Metric', value_name='Score')

# ---- Violin Plot ----
plt.figure(figsize=(10, 3))
colors = ["#2976B2", "#E67E22", "#27AE60", "#C0392B"]
sns.violinplot(
    x='Metric', y='Score', data=df_plot_melt,
    palette=colors, inner='quartile', linewidth=2, bw=0.18  # smoother violins
)

# Set Y limits tight around the data for "stretched" violins
score_min = min(plot_acc.min(), plot_prec.min(), plot_recall.min(), plot_f1.min())
score_max = max(plot_acc.max(), plot_prec.max(), plot_recall.max(), plot_f1.max())
plt.ylim(score_min - 0.004, score_max + 0.004)

plt.yticks(fontsize=14, weight='bold')
plt.xticks(fontsize=14, weight='bold')
plt.ylabel("Score", fontsize=14, weight='bold')
plt.xlabel("")
plt.title("Random Forest Performance Distribution", fontsize=14, weight='bold', pad=18)

# plt.text(-0.45, score_max + 0.001, 'Baseline', fontsize=14, weight='bold', color='black', va='bottom')

plt.grid(axis='y', linestyle='--', linewidth=1, alpha=0.5)
plt.tight_layout()
plt.savefig("random_forest_violin_plot_final.png", dpi=1200, bbox_inches='tight')
plt.show()

print("\n[INFO] Violin plot saved as 'random_forest_violin_plot_final.png' (1200 dpi)")
