#%%
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
#%%
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

#%%
# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Last point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle
#%%
# Load an example image
image_path = "/kaggle/input/82yogaclasses/yoga_images/images/Tree_Pose_or_Vrksasana_/0_42.jpg"  # Update with a valid path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#%%
# Process the image with MediaPipe
results = pose.process(image_rgb)
#%%
# Extract landmarks
angle = None  # Initialize the variable to avoid NameError

if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark

    # Get coordinates for keypoints (example: left shoulder, elbow, and wrist)
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    # Calculate angle
    angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    print(f"Left Arm Angle: {angle}")

    # Visualize keypoints and skeleton
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Display image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
else:
    print("No landmarks detected. Please ensure the input image shows a person performing a yoga pose.")

#%%
# Provide feedback only if angle is calculated
if angle is not None:
    if angle < 170:
        print("Straighten your arm for better alignment.")
        feedback = "Straighten your arm for better alignment."
    else:
        print("Great job maintaining arm alignment!")
        feedback = "Great job maintaining arm alignment."
else:
    feedback = "No landmarks detected. Unable to provide feedback."

#%%
# Store session insights
import json
session_data = {
    "pose": "Tree Pose",
    "angles": {"left_arm": angle if angle is not None else "Not detected"},
    "feedback": feedback,
    "time_spent": "15 seconds"
}
#%%
# Save insights to a file
with open("session_insights.json", "w") as file:
    json.dump(session_data, file, indent=4)
#%% md
# # trial
#%%
import cv2
import mediapipe as mp
import numpy as np
import os
import json
from tqdm import tqdm

#%%
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Define function to extract keypoints
def extract_keypoints(image_path):
    with mp_pose.Pose(static_image_mode=True) as pose:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            # Flatten landmarks into a single array
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
            return keypoints
        else:
            return None




#%%
def process_dataset(dataset_path, output_file):
    data = {}  # Initialize a dictionary to hold the processed data
    classes = os.listdir(dataset_path)
    
    valid_extensions = [".jpg", ".jpeg", ".png"]
    
    for class_name in tqdm(classes, desc="Processing classes"):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
        
        # Initialize an empty list for the class if not already done
        if class_name not in data:
            data[class_name] = []
        
        for img_name in os.listdir(class_path):
            if not any(img_name.lower().endswith(ext) for ext in valid_extensions):
                print(f"Skipping non-image file: {img_name}")
                continue
            
            img_path = os.path.join(class_path, img_name)
            
            # Debugging: Check if file exists
            if not os.path.exists(img_path):
                print(f"File not found: {img_path}")
                continue
            
            # Debugging: Print file path
            print(f"Processing: {img_path}")
            
            # Attempt to read the image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to read image: {img_path}")
                continue
            
            keypoints = extract_keypoints(img_path)
            if keypoints is not None:
                data[class_name].append({"image": img_name, "keypoints": keypoints.tolist()})
    
    # Save keypoints to JSON
    with open(output_file, "w") as file:
        json.dump(data, file, indent=4)

#%%
import warnings

# Suppress MediaPipe specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*feedback manager requires.*")

#%%
# Run the processing
dataset_path = "/kaggle/input/82yogaclasses/yoga_images/images"  # Change to your dataset path
output_file = "keypoints.json"
process_dataset(dataset_path, output_file)
#%%
import numpy as np
import tensorflow as tf
import json
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
import gc
from tensorflow.keras.backend import clear_session
#%%
# Load keypoints data
with open("/kaggle/input/keypoints/keypoints (1).json", "r") as file:
    keypoint_data = json.load(file)
#%%
# Prepare feature matrix and labels
X = []  # Features
y = []  # Labels
#%%
for class_name, samples in keypoint_data.items():
    for sample in samples:
        X.append(sample["keypoints"])
        y.append(class_name)
#%%

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
#%%
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

#%%
# Convert to numpy arrays
X_train, X_test = np.array(X_train), np.array(X_test)
y_train, y_test = np.array(y_train), np.array(y_test)

#%%
# Data generator
def data_generator(X, y, batch_size=16):
    while True:
        indices = np.random.permutation(len(X))
        for i in range(0, len(X), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_X = np.array([np.resize(X[j], (128, 128, 3)) for j in batch_indices])
            batch_y = y[batch_indices]
            yield batch_X, batch_y

train_gen = data_generator(X_train, y_train, batch_size=16)
val_gen = data_generator(X_test, y_test, batch_size=16)
#%%
# Define models to compare
def build_mlp(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(num_classes, activation="softmax")
    ])
    return model
#%%
def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    return model
#%%
def build_mobilenet(input_shape, num_classes):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model
#%%
def build_efficientnet(input_shape, num_classes):
    base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model
#%%
def build_resnet(input_shape, num_classes):
    base_model = ResNet50(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model
#%%
models = {
    "MLP": build_mlp(len(X_train[0]), len(label_encoder.classes_)),
    "CNN": build_cnn((128, 128, 3), len(label_encoder.classes_)),
    "MobileNetV2": build_mobilenet((128, 128, 3), len(label_encoder.classes_)),
    "EfficientNet": build_efficientnet((128, 128, 3), len(label_encoder.classes_)),
    "ResNet50": build_resnet((128, 128, 3), len(label_encoder.classes_))
}

#%%
X_train_resized = np.array([np.resize(x, (128, 128, 3)) for x in X_train])
X_test_resized = np.array([np.resize(x, (128, 128, 3)) for x in X_test])

#%%
# Train and evaluate models
best_model = None
best_model_name = ""
best_accuracy = 0
results = {}
model_histories = {}
#%%
for model_name, model in models.items():
    print(f"Training {model_name}...")

    reset_memory()  # Clear memory before training

    if model_name == "MLP":
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16, verbose=1)
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        history = model.fit(train_gen, steps_per_epoch=len(X_train) // 16,
                            validation_data=val_gen, validation_steps=len(X_test) // 16, epochs=50, verbose=1)

    # Store training history
    model_histories[model_name] = history

    # Evaluate the model
    if model_name == "MLP":
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(np.array([np.resize(x, (128, 128, 3)) for x in X_test]))

    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred_classes)
    results[model_name] = accuracy

    print(f"{model_name} Accuracy: {accuracy}")

    # Check if this model is the best
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = model_name
#%%
# Fine-tune the best model (if it is a pre-trained model)
if isinstance(best_model, Model):
    print(f"Fine-tuning the best model ({best_model.name})...")
    best_model.layers[0].trainable = True  # Unfreeze the base model layers
    best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                       loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    best_model.fit(X_train_resized, y_train, validation_data=(X_test_resized, y_test), epochs=10, batch_size=32, verbose=1)

#%%
# Save the best model
best_model.save("best_pose_detection_model.h5")
#%%
# Print results
print("Model Comparison Results:")
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.4f}")

print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")