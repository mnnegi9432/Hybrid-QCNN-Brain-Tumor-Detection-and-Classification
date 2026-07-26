import os
import joblib
import torch

MODEL_DIR = "."

# Load preprocessing objects
scaler = joblib.load(os.path.join(MODEL_DIR, "standard_scaler.pkl"))
pca = joblib.load(os.path.join(MODEL_DIR, "pca16.pkl"))

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model, filename):
    """
    Load a PyTorch checkpoint into the supplied model.
    """
    checkpoint = torch.load(
        os.path.join(MODEL_DIR, filename),
        map_location=device
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model
