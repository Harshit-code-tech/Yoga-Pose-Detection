import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2

# Load your pre-trained model (modify the path as needed)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("/home/hghosh/Desktop/CODING/Python/Internship/flask/models/best_pose_detection_model.h5")

model = load_model()

# Define a video transformer class for real-time prediction
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Preprocess the frame (resize and normalize as per your model's requirements)
        resized_img = cv2.resize(img, (28, 28))  # Example: resizing to 28x28 if the model requires it
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if needed
        normalized_img = gray_img / 255.0  # Normalize pixel values
        input_data = normalized_img.reshape(1, -1)  # Flatten or reshape as required by the model

        # Predict
        prediction = self.model.predict(input_data)
        predicted_class = np.argmax(prediction)

        # Display the prediction on the frame
        cv2.putText(
            img,
            f"Predicted: {predicted_class}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return img

# Sidebar Options
st.sidebar.header("Options")
data_upload = st.sidebar.file_uploader("Upload Test Data (CSV or Numpy File)", type=["csv", "npy"])
real_time_input = st.sidebar.text_input("Real-Time Input (Comma-separated 99 features)")

# Main Interface
st.title("Model Real-Time Feedback and Analysis")

if data_upload:
    st.subheader("Uploaded Data")

    # Load data
    if data_upload.name.endswith("csv"):
        data = pd.read_csv(data_upload)
    else:
        data = np.load(data_upload)
        data = pd.DataFrame(data)

    st.write(data.head())

    # Assuming the last column is the label
    X_test = data.iloc[:, :-1].values
    y_test = data.iloc[:, -1].values

    # Make predictions
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # Display Metrics
    accuracy = np.mean(y_pred_classes == y_test) * 100
    st.write(f"### Model Accuracy: {accuracy:.2f}%")

    # Classification Report
    st.subheader("Classification Report")
    class_report = classification_report(y_test, y_pred_classes, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    st.dataframe(class_report_df)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # User-selectable range for confusion matrix visualization
    st.write("### Confusion Matrix Heatmap")
    class_range = st.slider("Select Class Range for Visualization", 0, len(conf_matrix)-1, (0, 20))
    subset_matrix = conf_matrix[class_range[0]:class_range[1]+1, class_range[0]:class_range[1]+1]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(subset_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(class_range[0], class_range[1]+1), yticklabels=range(class_range[0], class_range[1]+1))
    plt.title("Confusion Matrix (Subset)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    st.pyplot(fig)

    # Top Misclassifications
    st.subheader("Top Misclassifications")
    misclassifications = [
        (i, j, conf_matrix[i, j])
        for i in range(conf_matrix.shape[0])
        for j in range(conf_matrix.shape[1]) if i != j and conf_matrix[i, j] > 0
    ]
    misclassifications = sorted(misclassifications, key=lambda x: x[2], reverse=True)

    top_misclassifications = pd.DataFrame(
        misclassifications[:10],
        columns=["True Label", "Predicted Label", "Count"]
    )
    st.write(top_misclassifications)

# Real-Time Single Input Prediction
if real_time_input:
    st.subheader("Real-Time Prediction")

    # Convert input to array
    try:
        real_time_features = np.array([float(i) for i in real_time_input.split(",")]).reshape(1, -1)
        prediction = model.predict(real_time_features)
        predicted_class = np.argmax(prediction)

        st.write(f"### Predicted Class: {predicted_class}")
        st.bar_chart(prediction[0])

    except ValueError:
        st.error("Invalid input! Please ensure it is comma-separated numerical values with 99 features.")

# Real-Time Camera Input
st.subheader("Real-Time Camera Input")
st.write("Start the camera and hold objects/gestures for real-time classification.")
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
