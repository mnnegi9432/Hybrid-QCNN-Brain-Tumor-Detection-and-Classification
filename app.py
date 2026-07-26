import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from predict import hybrid_predict

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


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

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            prediction="Unsupported file type."
        )

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    try:
        prediction = hybrid_predict(filepath)

        return render_template(
            "index.html",
            prediction=prediction,
            image=filename
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction=f"Error: {e}"
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
