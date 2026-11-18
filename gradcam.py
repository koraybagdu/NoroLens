import numpy as np
import tensorflow as tf
import cv2


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block6a_expand_activation",
                         plus=True):
    base_model = None
    classifier_layers = []
    found_base = False

    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            base_model = layer
            found_base = True
            continue
        if found_base:
            classifier_layers.append(layer)

    if base_model is None:
        return np.zeros((224, 224))

    grad_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[base_model.get_layer(last_conv_layer_name).output,
                 base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, features = grad_model(img_array)

        x = features
        for layer in classifier_layers[:-1]:
            x = layer(x)

        last = classifier_layers[-1]
        logit = tf.matmul(x, last.kernel) + last.bias   # sigmoid öncesi
        loss = logit[:, 0]

    grads = tape.gradient(loss, conv_output)[0]   # (H, W, C)
    A = conv_output[0]                            # (H, W, C)

    if plus:
        # Grad-CAM++ ağırlıkları
        g2 = tf.square(grads)
        g3 = g2 * grads
        sum_A = tf.reduce_sum(A, axis=(0, 1), keepdims=True)
        denom = 2.0 * g2 + sum_A * g3
        denom = tf.where(denom != 0.0, denom, tf.ones_like(denom))
        alpha = g2 / denom
        weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(0, 1))
    else:
        weights = tf.reduce_mean(grads, axis=(0, 1))

    heatmap = tf.reduce_sum(A * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0).numpy()

    # Uç değerlere dayanıklı normalizasyon
    hi = np.percentile(heatmap, 99)
    if hi > 0:
        heatmap = np.clip(heatmap / hi, 0, 1)
    return heatmap


def overlay_heatmap(heatmap, original_img, alpha=0.6, threshold=0.5, power=2.0,
                    colormap=cv2.COLORMAP_JET):
    """
    threshold: bu değerin altındaki bölgeler hiç boyanmaz (0.5 = sadece en güçlü yarı)
    power: >1 olursa zayıf bölgeler daha da bastırılır
    """
    h, w = original_img.shape[:2]
    hm = cv2.resize(heatmap.astype("float32"), (w, h), interpolation=cv2.INTER_CUBIC)
    hm = np.clip(hm, 0, 1)

    # Eşik altını sıfırla ve kalanı yeniden ölçekle
    hm = np.clip((hm - threshold) / (1.0 - threshold), 0, 1)
    hm = hm ** power

    jet = cv2.applyColorMap(np.uint8(255 * hm), colormap)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB).astype("float32")

    # Boyama şiddeti ısı değerine bağlı: zayıf bölge şeffaf kalır
    mask = (hm * alpha)[..., None]
    out = original_img.astype("float32") * (1 - mask) + jet * mask
    return np.clip(out, 0, 255).astype("uint8")