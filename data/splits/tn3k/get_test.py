import os

def list_files_in_directory(directory):
    """
    Recursively lists all files in a given directory.
    
    Args:
        directory (str): Path to the directory to search.

    Returns:
        list: A list of file paths found in the directory and its subdirectories.
    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list

directory_path = 'airs/tn3k/test-image/'
file_paths = list_files_in_directory(directory_path)
list = []

for path in file_paths:
    path = path.split('/')
    name = path[6] + '/' + path[7] + '/' + path[8]
    list.append(name)

# Write the formatted paths to a text file
with open('airs/tn3k/test.txt', 'w') as file:
    for path in list:
        file.write(path + '\n')