import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report

# Kullanım:
#   python3 test_dis.py                  -> "Test Dis" klasörünü kullanır
#   python3 test_dis.py ~/Desktop/test   -> verdiğin klasörü kullanır
# Klasör yapısı: <klasör>/no ve <klasör>/yes
TEST_DIR = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else "Test Dis"
THRESHOLD = 0.4


def crop_brain(img):  # RGB girdi, crop_dataset.py ile aynı mantık
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w * h < 0.2 * img.shape[0] * img.shape[1]:
        return img
    pad = 5
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, img.shape[1]), min(y + h + pad, img.shape[0])
    return img[y0:y1, x0:x1]


if not os.path.isdir(TEST_DIR):
    raise SystemExit(f"Test klasörü bulunamadı: {TEST_DIR}\n"
                     "Beklenen yapı: <klasör>/no ve <klasör>/yes")

model = tf.keras.models.load_model("best_model.keras")

X, y, names = [], [], []
for label, cat in enumerate(["no", "yes"]):
    folder = os.path.join(TEST_DIR, cat)
    if not os.path.isdir(folder):
        raise SystemExit(f"Alt klasör bulunamadı: {folder}")
    for f in os.listdir(folder):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = cv2.imread(os.path.join(folder, f))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(crop_brain(img), (224, 224))
        X.append(preprocess_input(img.astype("float32")))
        y.append(label)
        names.append(os.path.join(cat, f))

X, y = np.array(X), np.array(y)
probs = model.predict(X).ravel()
pred = (probs > THRESHOLD).astype(int)

print("Görsel sayısı:", len(y))
print("Dış test accuracy:", (pred == y).mean())
print(confusion_matrix(y, pred))
print(classification_report(y, pred, target_names=["no", "yes"]))

print("\nYanlış bilinenler:")
for i in np.where(pred != y)[0]:
    print(names[i], "gerçek:", y[i], "skor:", round(float(probs[i]), 3))
