# Check a sample file
file_path = "/home/hghosh/Desktop/CODING/Python/Internship/training/Yoga-82/yoga_dataset_links/Akarna_Dhanurasana.txt"

with open(file_path, 'r') as file:
    lines = file.readlines()

# Print the first few lines to understand the structure
print(lines[:5])