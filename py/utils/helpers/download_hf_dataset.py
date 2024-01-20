import os
import random
import zipfile
from io import BytesIO
import json
from utils.helpers.get_dataset_from_hf import ChessDataset


def create_folder(path):
    os.makedirs(path, exist_ok=True)


def create_subfolders(root_folder, num_folders):
    for i in range(num_folders):
        create_folder(os.path.join(root_folder, f"{i:03}"))


def dict_to_string(data):
    moves = ';'.join(data["Moves"])
    termination = data["Termination"]
    result = data["Result"]
    return f"{moves},{termination},{result}"


def zip_data(buffer):
    string_buffer = [dict_to_string(item) for item in buffer]
    with zipfile.ZipFile(BytesIO(), 'w', zipfile.ZIP_DEFLATED) as zip_buffer:
        zip_buffer.writestr('data.csv', '\n'.join(string_buffer))
        return zip_buffer.fp.getvalue()


def write_zip(data, path):
    with open(path, 'wb') as f:
        f.write(data)


def process_buffer(folder, buffer, file_index):
    if not buffer:
        return
    data = zip_data(buffer)
    file_path = os.path.join(folder, f"{file_index}.zip")
    write_zip(data, file_path)
    print(f"Written file {file_index}.zip in folder {folder}")


def main():
    root_folder = '/Volumes/Elements/dataset'  # Set your root folder path
    num_folders = 1000
    max_lines_per_file = 500

    chess_dataset = ChessDataset()

    create_subfolders(root_folder, num_folders)
    buffers = {i: [] for i in range(num_folders)}
    file_indices = {i: 0 for i in range(num_folders)}

    for line in chess_dataset.iterable:
        folder_index = random.randint(0, num_folders - 1)
        buffers[folder_index].append(line)
        if len(buffers[folder_index]) >= max_lines_per_file:
            folder_path = os.path.join(root_folder, f"{folder_index:03}")
            process_buffer(
                folder_path, buffers[folder_index], file_indices[folder_index])
            file_indices[folder_index] += 1
            buffers[folder_index] = []

    # Write remaining data in buffers
    for folder_index, buffer in buffers.items():
        if buffer:
            folder_path = os.path.join(root_folder, f"{folder_index:03}")
            process_buffer(folder_path, buffer, file_indices[folder_index])


if __name__ == "__main__":
    main()
