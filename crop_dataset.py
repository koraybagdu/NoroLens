import os
import cv2
import numpy as np

SRC = "Veri Seti"
DST = "Veri Seti Kirpilmis"
PAD = 5  # kenarlara bırakılacak boşluk (piksel)

def crop_brain(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Siyah arka planı ayır (Otsu eşikleme)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Çok küçük bir alan bulunduysa (hatalı tespit) orijinali koru
    if w * h < 0.2 * img.shape[0] * img.shape[1]:
        return img

    x0, y0 = max(x - PAD, 0), max(y - PAD, 0)
    x1, y1 = min(x + w + PAD, img.shape[1]), min(y + h + PAD, img.shape[0])
    return img[y0:y1, x0:x1]

count = 0
for cat in ["no", "yes"]:
    src_dir = os.path.join(SRC, cat)
    dst_dir = os.path.join(DST, cat)
    os.makedirs(dst_dir, exist_ok=True)

    for f in os.listdir(src_dir):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = cv2.imread(os.path.join(src_dir, f))
        if img is None:
            continue
        cv2.imwrite(os.path.join(dst_dir, f), crop_brain(img))
        count += 1

print("Kırpılan görsel sayısı:", count)