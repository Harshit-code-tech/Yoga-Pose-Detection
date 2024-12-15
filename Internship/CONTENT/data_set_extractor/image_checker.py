# Check a sample file
file_path = "/path/to/file.txt"

with open(file_path, 'r') as file:
    lines = file.readlines()

# Print the first few lines to understand the structure
print(lines[:5])
