import pandas as pd
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from mediapipe import solutions
import tempfile

# Load the class names dynamically from the Excel file
df = pd.read_excel("yoga_pose_classes.xlsx")
pose_list = df["Pose Name"].tolist()

# Load the model
model = tf.keras.models.load_model('best_pose_detection_model.h5')

# Sidebar for pose selection
selected_pose = st.sidebar.selectbox("Select a Pose", pose_list)

# File upload
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "png", "mp4", "avi", "mov", "mkv","webm"])

# Real-time camera feedback
use_camera = st.checkbox("Use Camera for Real-Time Feedback")

# Feedback function
def provide_feedback(keypoints, selected_pose):
    input_data = np.array([keypoints])
    prediction = model.predict(input_data)
    predicted_pose = pose_list[np.argmax(prediction)]
    confidence = np.max(prediction)

    feedback = []
    if predicted_pose != selected_pose:
        feedback.append(f"Adjust your pose closer to {selected_pose}.")
    feedback.append(f"Confidence in detected pose: {confidence:.2f}")

    return predicted_pose, confidence, feedback

# Keypoint extraction function
def extract_keypoints(image):
    with solutions.pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
            return keypoints, results.pose_landmarks
    return None, None

# Draw keypoints and connections
def draw_keypoints_and_connections(image, landmarks, connections):
    h, w, _ = image.shape
    annotated_image = image.copy()

    for i, landmark in enumerate(landmarks.landmark):
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

    for start, end in connections:
        x1, y1 = int(landmarks.landmark[start].x * w), int(landmarks.landmark[start].y * h)
        x2, y2 = int(landmarks.landmark[end].x * w), int(landmarks.landmark[end].y * h)
        cv2.line(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return annotated_image

# Mediapipe keypoint connections
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (0, 8),
    (8, 9), (9, 10), (10, 11), (11, 12), (12, 24), (8, 24), (24, 23),
    (23, 11), (23, 25), (25, 26), (26, 27), (27, 28)
]

# Initialize frame counter
frame_count = 0

# Real-Time Processing
if use_camera:
    st.write("Initializing camera...")
    cap = cv2.VideoCapture(0)
    st_frame = st.empty()
    feedback_text = st.empty()  # Placeholder for feedback

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break

        # Extract keypoints
        keypoints, landmarks = extract_keypoints(frame)

        if keypoints:
            # Predict pose
            predicted_pose, confidence, feedback = provide_feedback(keypoints, selected_pose)

            # Draw skeleton
            annotated_frame = draw_keypoints_and_connections(frame, landmarks, POSE_CONNECTIONS)

            # Update feedback in the single placeholder with a unique key
            feedback_text.text_area(
                "Improvement Feedback",
                "\n".join(feedback),
                key=f"camera_feedback_{frame_count}"
            )

            # Display updated frame
            st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

        frame_count += 1  # Increment frame count for unique key

        # Stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# File Processing
elif uploaded_file:
    file_type = uploaded_file.type
    st.write("Processing uploaded file...")

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    if "video" in file_type:
        cap = cv2.VideoCapture(temp_file_path)
        st_frame = st.empty()
        feedback_text = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Extract keypoints
            keypoints, landmarks = extract_keypoints(frame)

            if keypoints:
                # Predict pose
                predicted_pose, confidence, feedback = provide_feedback(keypoints, selected_pose)

                # Draw skeleton
                annotated_frame = draw_keypoints_and_connections(frame, landmarks, POSE_CONNECTIONS)

                # Update feedback
                feedback_text.text_area("Improvement Feedback", "\n".join(feedback), key=f"video_feedback_{frame_count}")

                # Display updated frame
                st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            frame_count += 1
        cap.release()

    elif "image" in file_type:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Extract keypoints
        keypoints, landmarks = extract_keypoints(image)

        if keypoints:
            # Predict pose
            predicted_pose, confidence, feedback = provide_feedback(keypoints, selected_pose)

            # Draw skeleton
            annotated_image = draw_keypoints_and_connections(image, landmarks, POSE_CONNECTIONS)

            # Display feedback
            st.text_area("Improvement Feedback", "\n".join(feedback), key="image_feedback")

            # Display updated frame
            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
                     caption=f"Pose: {predicted_pose} | Confidence: {confidence:.2f}")

st.write("\nPress 'q' to stop the camera feed.")
