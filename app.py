from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

from predict import hybrid_predict
from model_loader import load_checkpoint
from models import HybridQCNN

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------------------------------------
# TODO:
# Replace these placeholders with your actual models.
# This depends on how your EfficientNet feature extractor
# and quantum layer are instantiated in your training code.
# -------------------------------------------------

feature_model = None
hybrid_model = None

# Example:
# feature_model = load_checkpoint(feature_model, "best_efficientnet.pth")
# hybrid_model = load_checkpoint(hybrid_model, "best_hybrid_qcnn.pth")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template(
            "index.html",
            prediction="No image uploaded."
        )

    file = request.files["image"]

    if file.filename == "":
        return render_template(
            "index.html",
            prediction="Please select an image."
        )

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        result = hybrid_predict(
            feature_model,
            hybrid_model,
            filepath
        )
    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=result,
        image=filename
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
