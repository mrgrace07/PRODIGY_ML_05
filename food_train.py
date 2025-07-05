import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATASET_DIR = "dataset"
MODEL_FILE = "food_model.pkl"
IMAGE_SIZE = (100, 100)

def extract_hog_features(img):
    img = cv2.resize(img, IMAGE_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    return features

X = []
y = []

# Load images
for label in os.listdir(DATASET_DIR):
    label_dir = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    for file in os.listdir(label_dir):
        path = os.path.join(label_dir, file)
        img = cv2.imread(path)
        if img is None:
            continue
        features = extract_hog_features(img)
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
# # Create pipeline: PCA + SVM
# model = make_pipeline(PCA(n_components=50), SVC(kernel='linear', probability=True))
from sklearn.preprocessing import StandardScaler

model = make_pipeline(
    StandardScaler(),
    PCA(n_components=50),
    SVC(kernel='linear', probability=True)
)

from collections import Counter
print(Counter(y))

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc * 100:.2f}%")

# Optional: Print detailed performance
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)

print(f"\n📦 Trained HOG+PCA+SVM model saved to: {MODEL_FILE}")
