from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from utils.preprocess import preprocess_image
from utils.feedback import get_pose_feedback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the pose detection model
model = tf.keras.models.load_model('/home/hghosh/Desktop/CODING/Python/Internship/flask/models/best_pose_detection_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)

        # Process image and extract features
        image = Image.open(filepath).convert('RGB')
        features = preprocess_image(image).reshape(1, -1)

        prediction = model.predict(features)
        feedback = get_pose_feedback(prediction)

        return jsonify({"prediction": prediction.tolist(), "feedback": feedback})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/realtime', methods=['POST'])
def realtime_prediction():
    try:
        data = request.get_json()
        image_base64 = data.get("image")
        pose = data.get("pose", "Unknown pose")

        if not image_base64:
            return jsonify({"error": "No image data provided"}), 400

        # Decode the base64 image
        header, encoded = image_base64.split(",", 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_data)).convert('RGB')

        # Resize/process the image
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        feedback = get_pose_feedback(prediction, pose)

        return jsonify({"prediction": prediction.tolist(), "feedback": feedback})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
