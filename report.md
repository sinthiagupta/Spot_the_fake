# Spot the Fake Photo — Submission Note

## Approach

I used **Transfer Learning with MobileNetV2** as a frozen feature extractor combined with **Logistic Regression** as the classifier.

**How it works:**
1. Each image is resized to 224x224 and passed through MobileNetV2 (pretrained on ImageNet, weights frozen). The final classification head is removed, giving a **1280-dimensional feature vector** per image.
2. These feature vectors are used to train a **Logistic Regression classifier** (LogisticRegressionCV with cross-validation to pick the best regularization).
3. At prediction time, predict.py runs the same pipeline and returns a score from **0.0 (real photo) to 1.0 (screen recapture)**.

The intuition: MobileNetV2's deep features capture subtle differences in texture, colour fidelity, sharpness, and glare patterns that distinguish a real camera photo from a photo of a screen.

---

## Accuracy

| Split | Images | Accuracy |
|---|---|---|
| Training set | 80 images | 100% |
| **Held-out test set** | **20 images** | **95%** |

The honest number is **95% on 20 held-out images** that the model never saw during training (stratified 80/20 split, random_state=42). One screen image was misclassified as real — it was a very clean, bright screen shot that produced features similar to a real photo.

---

## Latency

| Metric | Value | Notes |
|---|---|---|
| Min latency | ~47 ms | Warmed-up speed (model already in memory) |
| Avg latency | ~90 ms | Includes some cold-start overhead |
| **Production latency** | **~47 ms** | In a real app, model loads once at startup |

**Device:** Windows laptop, Intel CPU (no GPU).

The bottleneck is the MobileNetV2 forward pass (~75ms). The Logistic Regression prediction itself takes less than 1ms.

---

## Cost Per Image

| Deployment | Cost |
|---|---|
| **On-device (phone)** | **Free** — runs entirely on the user's device. No server needed. |
| Cloud (AWS t3.micro, $0.0104/hr) | Approx. 40,000 images/hr → **Approx. $0.26 per million images** |
| Cloud (AWS c6i.large, $0.085/hr) | Approx. $1.50 per million images, 3x faster throughput |

On-device deployment is the best choice — runs free, works offline, and is instant after warmup.

---

## What I Would Improve With More Time

1. **More diverse training data** — Photos from different lighting, angles, and screen types (OLED vs LCD) to improve generalisation.
2. **Fine-tune MobileNetV2** — Unfreeze the last 2-3 layers and fine-tune on training data to push accuracy above 97%.
3. **Convert to TFLite / CoreML** — For phone deployment to drop latency under 10ms on modern phones.
4. **Adaptive threshold** — Use a higher threshold (e.g., 0.7) to reduce false positives in fraud detection context.
5. **Anti-cheating robustness** — Test against adversarial cases: screen glare removed in post-processing, printed photos, or high-resolution OLED screens.
