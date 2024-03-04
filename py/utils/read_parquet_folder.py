import os
import pandas as pd


def read_parquet_folder(folder_path):
    """
    Generator function to yield rows from Parquet files in the given folder.
    """
    row_count = 0

    # List all files in the folder
    files = os.listdir(folder_path)

    print(f"Reading files from {folder_path}")
    print(f"Total files: {len(files)}")

    # Filter for Parquet files and yield rows
    for file in files:
        print("file:", file)
        if file.endswith('.parquet'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_parquet(file_path)
            print("df:", df)
            for _, row in df.iterrows():
                row_count += 1
                if row_count % 5000 == 0:
                    print(f"Read {row_count} rows from {folder_path}")
                yield row


# Usage example
# folder_path = 'your_folder_path'  # Replace with the path to your folder
# for row in read_parquet_folder(folder_path):
#     # Process each row as needed
#     print(row)
