import os
import sys


def find_most_recent_model(folder):
    """Recursively finds the most recently modified model.json file in a folder."""

    most_recent_time = 0
    most_recent_file = ""

    # Search for model.json in the current directory
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isdir(file_path):
            # Recursively search the subdirectory
            subdirectory_file = find_most_recent_model(file_path)
            if subdirectory_file:
                # Update most_recent_file if a more recent file was found in the subdirectory
                subdirectory_time = os.path.getmtime(subdirectory_file)
                if subdirectory_time > most_recent_time:
                    most_recent_time = subdirectory_time
                    most_recent_file = subdirectory_file
        elif file == "model.json":
            # Check if model.json is in the current directory
            modified_time = os.path.getmtime(file_path)
            if modified_time > most_recent_time:
                most_recent_time = modified_time
                most_recent_file = file_path

    return most_recent_file


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python find_metadata.py folder")
        sys.exit(1)

    folder = sys.argv[1]

    if not os.path.isdir(folder):
        print(folder, "is not a valid directory")
        sys.exit(1)

    most_recent_file = find_most_recent_model(folder)

    if not most_recent_file:
        print("No model.json files found in", folder)
    else:
        parent_folder = os.path.abspath(
            os.path.join(most_recent_file, os.pardir))

        print("Most recently modified model.json file found in:", parent_folder)
