import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from skimage.filters import sobel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ---- Configuration ----
DATASET_PATH = "evolved"
# DATASET_PATH = "evolved and baseline"

IMG_SIZE = (256, 256)
LBP_POINTS = 24
LBP_RADIUS = 3
LABELS = {'Fear': 0, 'Happiness': 1, 'Neutral': 2}

# ---- Feature extraction ----
def extract_features(image):
    gray = cv2.resize(image, IMG_SIZE)

    # HOG
    hog_features = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)

    # LBP
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # Sobel edge
    sobel_edges = sobel(gray)
    sobel_hist, _ = np.histogram(sobel_edges.ravel(), bins=32, range=(0, 1))

    # Intensity histogram
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist /= (hist.sum() + 1e-6)

    # Concatenate all features
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

# ---- Train/Test Split ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Train Classifiers ----
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# ---- Train Classifiers ----
print("\n--- Random Forest ---")
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=LABELS.keys()))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("Precision (macro): {:.2f}%".format(precision_score(y_test, y_pred, average='macro') * 100))
print("Recall (macro): {:.2f}%".format(recall_score(y_test, y_pred, average='macro') * 100))
print("F1 Score (macro): {:.2f}%".format(f1_score(y_test, y_pred, average='macro') * 100))

print("\n--- Decision Tree ---")
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(classification_report(y_test, y_pred, target_names=LABELS.keys()))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("Precision (macro): {:.2f}%".format(precision_score(y_test, y_pred, average='macro') * 100))
print("Recall (macro): {:.2f}%".format(recall_score(y_test, y_pred, average='macro') * 100))
print("F1 Score (macro): {:.2f}%".format(f1_score(y_test, y_pred, average='macro') * 100))

print("\n--- XGBoost ---")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
print(classification_report(y_test, y_pred, target_names=LABELS.keys()))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("Precision (macro): {:.2f}%".format(precision_score(y_test, y_pred, average='macro') * 100))
print("Recall (macro): {:.2f}%".format(recall_score(y_test, y_pred, average='macro') * 100))
print("F1 Score (macro): {:.2f}%".format(f1_score(y_test, y_pred, average='macro') * 100))

# ---- Cross-Validation Results ----
# print("\n--- Cross-Validation (5-fold) ---")
# rf_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
# print(f"Random Forest CV Accuracy: {rf_scores.mean():.2f} ± {rf_scores.std():.2f}")

# dt_scores = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
# print(f"Decision Tree CV Accuracy: {dt_scores.mean():.2f} ± {dt_scores.std():.2f}")

# xgb_scores = cross_val_score(xgb, X, y, cv=5, scoring='accuracy')
# print(f"XGBoost CV Accuracy: {xgb_scores.mean():.2f} ± {xgb_scores.std():.2f}")
