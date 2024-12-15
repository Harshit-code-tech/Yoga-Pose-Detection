# Yoga Pose Detection

This project implements a real-time yoga pose detection feature using MediaPipe Pose and a custom-trained neural network. Designed for yoga practitioners, the application provides actionable feedback to improve pose alignment and accuracy.

## Features

- **Pose Detection**: Predicts yoga poses using a TensorFlow-based MLP model.
- **Real-Time Feedback**: Analyze poses via webcam and provide corrective feedback instantly.
- **File Upload Support**: Works with uploaded images and videos.
- **Customizable Pose Selection**: Users can select a target pose for specific guidance.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/yoga-pose-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd yoga-pose-detection
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Application**:
   ```bash
   streamlit run yoga_pose_app.py
   ```
2. **Features**:
   - Select a target pose from the sidebar.
   - Upload an image or video file (supported formats: `.jpg`, `.png`, `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`).
   - Enable real-time feedback using your webcam.
3. **Output**:
   - Display of the predicted pose with confidence score.
   - Corrective feedback based on the selected pose.

## File Descriptions

- `yoga_pose_app.py`: Main application script.
- `best_pose_detection_model.h5`: Pre-trained MLP model for pose classification.
- `yoga_pose_classes.xlsx`: Excel file containing yoga pose names.

## Model Details

The model is an MLP trained on extracted pose keypoints:
- **Input**: Normalized 3D keypoints from MediaPipe Pose.
- **Architecture**: Three dense layers with ReLU activations and softmax for classification.
- **Accuracy**: Achieved 81.4% accuracy on the validation dataset.
- **Output**: Yoga pose classification.

## Key Areas of Focus

1. **Problem Understanding & Creativity**:
   - The problem of pose misalignment is addressed through an innovative pipeline combining MediaPipe for keypoint extraction and a custom MLP for pose classification. This ensures lightweight, efficient, and accurate pose detection.

2. **Model Performance**:
   - The model achieves 81.4% accuracy, providing reliable predictions. 
   - Visual feedback through annotated skeletons ensures users can easily understand and correct their poses.

3. **User-Centric Application**:
   - Designed with yoga practitioners in mind, the app emphasizes ease of use and practical feedback.
   - Real-time and file-based processing options cater to diverse user preferences.

4. **Scalability**:
   - The architecture can scale to include more poses and adapt to varied datasets.
   - The modular design ensures ease of integration with other fitness applications.

## Contributing

We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature/bugfix.
3. Commit your changes and push to your fork.
4. Submit a pull request with a detailed explanation.

## Future Enhancements

- Expand the dataset to include more yoga poses.
- Add multi-language support for global accessibility.
- Enhance real-time performance with optimized models.
- Integrate with fitness tracking platforms.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [MediaPipe Pose](https://google.github.io/mediapipe/) for keypoint extraction.
- TensorFlow for model training and inference.
- Streamlit for building an interactive UI.

## Contact

For any inquiries or suggestions, please contact [Harshit Ghosh](https://www.linkedin.com/in/harshit-ghosh-026622272/).

