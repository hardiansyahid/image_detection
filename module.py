import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import os

# Konversi ke grayscale
def convert_to_grayscale(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

# Ekstrak fitur LBP
def extract_texture_lbp(gray):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist  # return histogram sebagai fitur

# Ekstrak fitur GLCM
def extract_texture_glcm(gray):
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return [contrast, correlation]  # bisa ditambah fitur lain juga

# Klasifikasi
def classification(data_features, labels):
    X_train, X_test, y_train, y_test = train_test_split(data_features, labels, test_size=0.2, random_state=42)
    model = SVC()
    model.fit(X_train, y_train)
    print("Akurasi:", model.score(X_test, y_test))
    return model

def predict_image(model, image_path):
    gray = convert_to_grayscale(image_path)
    lbp_feat = extract_texture_lbp(gray)
    glcm_feat = extract_texture_glcm(gray)
    combined_features = np.concatenate([lbp_feat, glcm_feat]).reshape(1, -1)
    return model.predict(combined_features)[0]

def load_dataset(dataset_dir):
    data_features = []
    labels = []

    for label_name in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label_name)
        if not os.path.isdir(label_path):
            continue

        for filename in os.listdir(label_path):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(label_path, filename)
                try:
                    gray = convert_to_grayscale(image_path)
                    lbp_feat = extract_texture_lbp(gray)
                    glcm_feat = extract_texture_glcm(gray)
                    combined_features = np.concatenate([lbp_feat, glcm_feat])
                    data_features.append(combined_features)
                    labels.append(label_name)
                except Exception as e:
                    print(f"Gagal proses {image_path}: {e}")

    return data_features, labels