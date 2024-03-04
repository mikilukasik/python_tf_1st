import pandas as pd


class ParquetWriter:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = []

    def add_row(self, row):
        """Add a single row to the data."""
        self.data.append(row)

    def write_to_parquet(self, engine='pyarrow', compression='snappy'):
        """Write the data to a Parquet file."""
        df = pd.DataFrame(self.data)
        df.to_parquet(self.file_name, engine=engine, compression=compression)
