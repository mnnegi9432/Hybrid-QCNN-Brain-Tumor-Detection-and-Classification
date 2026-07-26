import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from model_loader import scaler, pca


CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def extract_features(model, image):
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy()


def prepare_features(features):
    features = scaler.transform(features)
    features = pca.transform(features)
    return torch.tensor(features, dtype=torch.float32)


def hybrid_predict(feature_model, hybrid_model, image_path):

    image = preprocess_image(image_path)

    features = extract_features(feature_model, image)

    features = prepare_features(features)

    with torch.no_grad():
        output = hybrid_model(features)

        prediction = torch.argmax(output, dim=1).item()

    return CLASS_NAMES[prediction]
