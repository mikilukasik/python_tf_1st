import os
import random
from utils.helpers.get_dataset_from_hf import ChessDataset, game_to_moves
import pandas as pd

root_folder = '/Volumes/Elements/processed_dataset'
num_folders = 1000
max_lines_per_file = 1000


class ParquetWriter:
    def __init__(self):
        self.data = []

    def add_row(self, row):
        self.data.append(row)

    def write_to_parquet(self, file_name, engine='pyarrow', compression='snappy'):
        df = pd.DataFrame(self.data)
        df.columns = df.columns.map(str)
        df.to_parquet(file_name, engine=engine, compression=compression)


def create_folder(path):
    os.makedirs(path, exist_ok=True)


def create_subfolders(root_folder, num_folders):
    for i in range(num_folders):
        create_folder(os.path.join(root_folder, f"{i:03}"))


def process_buffer(folder, writer, file_index):
    if not writer.data:
        return
    file_path = os.path.join(folder, f"{file_index:05}.parquet")
    writer.write_to_parquet(file_path)
    print(f"Written file {file_index}.parquet in folder {folder}")


def main():

    chess_dataset = ChessDataset()

    create_subfolders(root_folder, num_folders)
    writers = {i: ParquetWriter() for i in range(num_folders)}
    file_indices = {i: 0 for i in range(num_folders)}

    for game in chess_dataset.iterable:
        for line in game_to_moves(game):
            # print(line)
            folder_index = random.randint(0, num_folders - 1)
            writers[folder_index].add_row(line)
            if len(writers[folder_index].data) >= max_lines_per_file:
                folder_path = os.path.join(root_folder, f"{folder_index:03}")
                process_buffer(
                    folder_path, writers[folder_index], file_indices[folder_index])
                file_indices[folder_index] += 1
                writers[folder_index] = ParquetWriter()

    # Write remaining data in writers
    for folder_index, writer in writers.items():
        if writer.data:
            folder_path = os.path.join(root_folder, f"{folder_index:03}")
            process_buffer(folder_path, writer, file_indices[folder_index])


if __name__ == "__main__":
    main()
