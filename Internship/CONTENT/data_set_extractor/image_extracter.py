import os
import requests

# Define paths
base_folder = "/path/to/file"  # Change to your desired download location
txt_folder = "/path/to/saving_folder"  # Folder containing all the pose .txt files

# Create base folder if it doesn't exist
os.makedirs(base_folder, exist_ok=True)

# Iterate through all .txt files in the txt_folder
for txt_file in os.listdir(txt_folder):
    # Process only .txt files
    if txt_file.endswith(".txt") and txt_file != "yoga_train.txt" and txt_file != "yoga_test.txt":
        pose_file_path = os.path.join(txt_folder, txt_file)
        print(f"Processing file: {txt_file}")

        with open(pose_file_path, "r") as file:
            for line in file:
                # Split into relative path and URL
                try:
                    relative_path, url = line.strip().split("\t")
                except ValueError:
                    print(f"Skipping malformed line in {txt_file}: {line.strip()}")
                    continue

                # Create subdirectories as per the relative path
                full_path = os.path.join(base_folder, relative_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)

                # Download the image
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        with open(full_path, "wb") as img_file:
                            img_file.write(response.content)
                        print(f"Downloaded: {relative_path}")
                    else:
                        print(f"Failed (HTTP {response.status_code}): {url}")
                except Exception as e:
                    print(f"Error downloading {url}: {e}")

print("Download process complete.")
