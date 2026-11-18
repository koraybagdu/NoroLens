import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from gradcam import make_gradcam_heatmap, overlay_heatmap

THRESHOLD = 0.4
CAM_LAYER = "block6a_expand_activation"  # alternatifler: "top_activation", "block4a_expand_activation"


def crop_brain(img):
    """crop_dataset.py ile aynı kırpma (RGB girdi)."""
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


def predict(model, rgb_img):
    resized = cv2.resize(rgb_img, (224, 224))
    tensor = preprocess_input(np.expand_dims(resized.astype("float32"), axis=0))
    score = float(model.predict(tensor, verbose=0)[0][0])
    return resized, tensor, score


st.set_page_config(page_title="Beyin Tümörü Tespiti & Grad-CAM", layout="wide")

st.title("🧠 Beyin Tümörü Tespiti ve Açıklanabilir AI (XAI)")
st.write("MRI taramasını yükleyin; model analiz etsin ve karar verirken baktığı bölgeyi Grad-CAM ile göstersin.")
st.caption("Bu uygulama eğitim amaçlıdır; tıbbi tanı için kullanılamaz.")


@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("best_model.keras")


model = load_trained_model()

uploaded_file = st.sidebar.file_uploader("Bir Beyin MRI Görüntüsü Yükleyin (JPG/PNG)", type=["jpg", "jpeg", "png"])
use_crop = st.sidebar.checkbox("Kırpmayı kullan (Grad-CAM ve karar için)", value=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # İki yolu da hesapla (teşhis için)
    cropped_np = crop_brain(img_np)
    res_crop, ten_crop, score_crop = predict(model, cropped_np)
    res_raw, ten_raw, score_raw = predict(model, img_np)

    if use_crop:
        resized_img, input_tensor, prediction = res_crop, ten_crop, score_crop
    else:
        resized_img, input_tensor, prediction = res_raw, ten_raw, score_raw

    is_tumor = prediction > THRESHOLD
    confidence = prediction if is_tumor else (1.0 - prediction)

    heatmap = make_gradcam_heatmap(input_tensor, model, last_conv_layer_name=CAM_LAYER)
    cam_image = overlay_heatmap(heatmap, resized_img)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Orijinal MRI")
        st.image(image, use_container_width=True)
        st.image(resized_img, caption="Modele giren görsel", use_container_width=True)

    with col2:
        st.subheader("Analiz Sonucu")
        if is_tumor:
            st.error("⚠️ **Tümör Tespit Edildi**")
        else:
            st.success("✅ **Sağlıklı / Tümör Yok**")

        st.metric(label="Model Güven Skoru", value=f"{confidence * 100:.2f}%")
        st.progress(float(confidence))

        st.markdown("**Teşhis: ham skorlar**")
        st.write(f"Kırpmalı: {score_crop:.3f} → {'Tümör' if score_crop > THRESHOLD else 'Tümörsüz'}")
        st.write(f"Kırpmasız: {score_raw:.3f} → {'Tümör' if score_raw > THRESHOLD else 'Tümörsüz'}")
        st.caption(f"Eşik: {THRESHOLD}")

    with col3:
        st.subheader("Grad-CAM Karar Haritası")
        st.image(
            cam_image,
            caption="Sıcak renkler modelin kararına en çok katkı yapan bölgelerdir. Tümörün kesin konumu anlamına gelmez.",
            use_container_width=True,
        )