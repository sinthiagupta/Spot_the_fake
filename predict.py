"""Fill this in. That's the whole interface.

Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)
A hard 0 or 1 is fine if your method gives a yes/no answer.
"""

import sys
from PIL import Image
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import joblib
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(pretrained=True)
model.classifier = torch.nn.Identity()
model.eval()
model.to(device)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
clf = joblib.load(os.path.join(os.path.dirname(__file__), "classifier.pkl"))
def extract_features(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(tensor).squeeze(0).cpu().numpy()
    return features

def predict(image_path: str) -> float:
    features = extract_features(image_path)
    prob = clf.predict_proba(features.reshape(1, -1))[0][1]
    return float(prob)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    print(f"{predict(sys.argv[1]):.4f}")
