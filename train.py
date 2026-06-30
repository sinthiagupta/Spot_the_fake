import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from predict import extract_features
def main():
    real_dir = os.path.join(os.path.dirname(__file__), "real_images")
    screen_dir = os.path.join(os.path.dirname(__file__), "screen_images")

    if not os.path.exists(real_dir) or not os.path.exists(screen_dir):
        print("Error: Make sure 'real_images/' and 'screen_images/' folders exist.")
        return

    X = []
    y = []
    filenames = []

    def process_folder(folder_path, label):
        print(f"Processing folder: {folder_path}...")
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        for f in files:
            full_path = os.path.join(folder_path, f)
            try:
                feats = extract_features(full_path)
                X.append(feats)
                y.append(label)
                filenames.append(f)
            except Exception as e:
                print(f"  Skipping {f} due to error: {e}")

    process_folder(real_dir, 0)
    process_folder(screen_dir, 1)

    X = np.array(X)
    y = np.array(y)
    filenames = np.array(filenames)

    if len(X) < 6:
        print("Error: Please collect at least 6 total images.")
        return
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    df.insert(0, "filename", filenames)
    df.insert(1, "label", y)
    df.to_csv("features_labels.csv", index=False)
    print(f"\nSaved full features CSV to 'features_labels.csv' ({len(df)} rows, {X.shape[1]} feature columns)")
    X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(
        X, y, filenames, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTraining Set ({len(f_train)} images):")
    for fname, label in zip(f_train, y_train):
        print(f"  [{'REAL' if label == 0 else 'SCREEN'}] {fname}")

    print(f"\nTest Set ({len(f_test)} images):")
    for fname, label in zip(f_test, y_test):
        print(f"  [{'REAL' if label == 0 else 'SCREEN'}] {fname}")
    cv_folds = min(5, np.min(np.bincount(y_train)))
    clf = LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0], cv=cv_folds, max_iter=1000, random_state=42)
    print("\nTraining Logistic Regression classifier...")
    clf.fit(X_train, y_train)

    print(f"\nTraining Complete!")
    print(f"  Best Regularization C: {clf.C_[0]}")
    print(f"  Train Accuracy:        {clf.score(X_train, y_train) * 100:.2f}%")
    print(f"  Validation Accuracy:   {clf.score(X_test, y_test) * 100:.2f}%")

    clf_path = os.path.join(os.path.dirname(__file__), "classifier.pkl")
    joblib.dump(clf, clf_path)
    print(f"\nSaved trained classifier to '{clf_path}'!")

if __name__ == "__main__":
    main()

