import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

IMG_SIZE = 224
BATCH = 32
DATA_DIR = "Veri Seti Kirpilmis"
THRESHOLD = 0.4  # düşük eşik = daha az false negative

# 1) Sadece dosya yollarını topla
paths, labels = [], []
for label, cat in enumerate(["no", "yes"]):
    folder = os.path.join(DATA_DIR, cat)
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith("aug_"):
            paths.append(os.path.join(folder, f))
            labels.append(label)

paths, labels = np.array(paths), np.array(labels, dtype="float32")

# 2) Train / Val / Test ayrımı
p_train, p_tmp, y_train, y_tmp = train_test_split(
    paths, labels, test_size=0.3, random_state=42, stratify=labels)
p_val, p_test, y_val, y_test = train_test_split(
    p_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print(f"Eğitim: {len(p_train)} | Val: {len(p_val)} | Test: {len(p_test)}")

# 3) tf.data pipeline
def load_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img.set_shape((IMG_SIZE, IMG_SIZE, 3))
    img = preprocess_input(img)
    return img, label

def make_ds(p, y, training):
    ds = tf.data.Dataset.from_tensor_slices((p, y))
    if training:
        ds = ds.shuffle(len(p), seed=42)
    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(p_train, y_train, True)
val_ds = make_ds(p_val, y_val, False)
test_ds = make_ds(p_test, y_test, False)

# 4) Model
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom((-0.2, 0.2)),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomContrast(0.3),
], name="augment")

base_model = EfficientNetB0(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                            include_top=False, weights="imagenet")
base_model.trainable = False  # Aşama 1: tamamen dondur

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = augment(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = models.Model(inputs, outputs)

cw = compute_class_weight("balanced", classes=np.array([0., 1.]), y=y_train)
class_weight = {0: cw[0], 1: cw[1]}

# Tek checkpoint nesnesi: iki aşama boyunca en iyi val_accuracy korunur
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1)

def make_callbacks():
    return [
        checkpoint,
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                             patience=2, min_lr=1e-6),
    ]

# AŞAMA 1: sadece başlığı eğit
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
print("\nAşama 1: Başlık eğitimi")
model.fit(train_ds, validation_data=val_ds, epochs=8,
          class_weight=class_weight, callbacks=make_callbacks())

# AŞAMA 2: ince ayar (BatchNorm dondurulmuş)
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="binary_crossentropy", metrics=["accuracy"])
print("\nAşama 2: İnce ayar")
model.fit(train_ds, validation_data=val_ds, epochs=25,
          class_weight=class_weight, callbacks=make_callbacks())

# 5) Test seti değerlendirmesi
best = tf.keras.models.load_model("best_model.keras")
print("\nTest sonucu:", best.evaluate(test_ds))

probs = best.predict(test_ds).ravel()
for th in [0.5, THRESHOLD]:
    y_pred = (probs > th).astype(int)
    print(f"\n=== Eşik: {th} ===")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["no", "yes"]))


    # Hatalı tahminleri yazdır
    wrong = np.where(((probs > 0.4).astype(int)) != y_test)[0]
print("\nYanlış bilinen görseller:")
for i in wrong:
    print(p_test[i], "gerçek:", int(y_test[i]), "skor:", round(float(probs[i]), 3))