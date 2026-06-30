import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from predict import predict

real_dir = "real_images"
screen_dir = "screen_images"
all_files = []
all_labels = []

for f in os.listdir(real_dir):
    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
        all_files.append((real_dir, f))
        all_labels.append(0)

for f in os.listdir(screen_dir):
    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
        all_files.append((screen_dir, f))
        all_labels.append(1)
_, test_files, _, test_labels = train_test_split(
    all_files, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

print(f"Evaluating on {len(test_files)} held-out test images...\n")

results = []
for (folder, f), label in zip(test_files, test_labels):
    path = os.path.join(folder, f)
    t0 = time.perf_counter()
    score = predict(path)
    latency = (time.perf_counter() - t0) * 1000
    predicted = 1 if score >= 0.5 else 0
    correct = predicted == label
    tag = "REAL" if label == 0 else "SCREEN"
    results.append((f, label, score, predicted, correct, latency))
    print(f"[{tag}]   score={score:.3f} {'✓' if correct else '✗'}  {f[:40]}")

total = len(results)
correct_total = sum(r[4] for r in results)
latencies = [r[5] for r in results]

print("\n" + "="*50)
print(f"Test images evaluated: {total} (held-out 20%)")
print(f"Correct predictions:   {correct_total}")
print(f"Accuracy:              {correct_total/total*100:.2f}%")
print(f"Avg latency:           {np.mean(latencies):.1f} ms per image")
print(f"Min latency:           {np.min(latencies):.1f} ms  (warmed up / production speed)")
print(f"Max latency:           {np.max(latencies):.1f} ms  (includes cold start on first image)")
print("="*50)
