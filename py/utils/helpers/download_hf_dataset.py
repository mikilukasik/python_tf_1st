import os
import random
# import zipfile
from io import BytesIO
# import json
from utils.helpers.get_dataset_from_hf import ChessDataset, game_to_csv
import gzip


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


def gzip_data(buffer):
    string_buffer = [dict_to_string(item) for item in buffer]
    with BytesIO() as bytes_io:
        with gzip.GzipFile(fileobj=bytes_io, mode='w') as gz_buffer:
            gz_buffer.write('\n'.join(string_buffer).encode('utf-8'))
        return bytes_io.getvalue()


def write_gzip(data, path):
    with open(path, 'wb') as f:
        f.write(data)


def process_buffer(folder, buffer, file_index):
    if not buffer:
        return
    data = gzip_data(buffer)
    file_path = os.path.join(folder, f"{file_index}.gz")
    write_gzip(data, file_path)  # Change function name
    print(f"Written file {file_index}.gz in folder {folder}")


def main():
    root_folder = '/Volumes/Elements/processed_dataset'  # Set your root folder path
    # root_folder = '__try_this'  # Set your root folder path
    num_folders = 1000
    max_lines_per_file = 500

    chess_dataset = ChessDataset()

    create_subfolders(root_folder, num_folders)
    buffers = {i: [] for i in range(num_folders)}
    file_indices = {i: 0 for i in range(num_folders)}

    for game in chess_dataset.iterable:
        for line in game_to_csv(game):
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
