import json
import csv

# Load the JSON file
with open('/home/hghosh/Desktop/CODING/Python/Internship/flask/models/keypoints (1).json', 'r') as file:
    data = json.load(file)

# Define output CSV file
output_csv = "yoga_pose_keypoints.csv"

# Extract and write keypoints to CSV
with open(output_csv, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write header
    csv_writer.writerow(["Pose Name", "Keypoints"])

    # Iterate over the dictionary
    for pose_name, pose_data in data.items():
        for pose in pose_data:
            keypoints = pose.get("keypoints", [])
            csv_writer.writerow([pose_name, keypoints])

print(f"Keypoints successfully extracted and saved to {output_csv}")
