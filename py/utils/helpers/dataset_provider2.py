import time
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
import logging
import requests
from . import prefetch
from .get_dataset_from_hf import ChessDataset
from ..print_large import print_large
import io
import sys
chess_dataset = ChessDataset()


class DatasetProvider:
    def __init__(self, model_meta, batch_size, ys_format='default', xs_format='default', dataset_reader_version='16', filter='default', evaluateOnly=False, fresh_reader=False):
        self.dataset_reader_id = 'eval2' if evaluateOnly else model_meta.get(
            "dataseReaderId")
        self.batch_size = batch_size
        self.ys_format = ys_format
        self.xs_format = xs_format
        self.dataset_reader_version = dataset_reader_version
        self.filter = filter

        # unfortunately our huggingface implementation always starts a new reader (currently)

        # if fresh_reader or not self.dataset_reader_id:
        #     dataset_reader_response = requests.get(
        #         "http://localhost:3550/datasetReader?ysformat="+self.ys_format+'&xsformat='+self.xs_format+'&readerVersion='+self.dataset_reader_version+'&filter='+self.filter)
        #     self.dataset_reader_id = dataset_reader_response.json().get("id")
        #     model_meta["dataseReaderId"] = self.dataset_reader_id
        #     print_large("", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "", "New dataset_reader_id retrieved:",
        #                 self.dataset_reader_id, "", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "")

        prefetch.set_data_getter(self.get_dataset)
        prefetch.prefetch_data()

    def get_next_batch(self):
        data = prefetch.get_data()
        prefetch.prefetch_data()
        return data

    def get_dataset(self, url, max_retries=50, retry_interval=5):
        retries = 0
        while retries < max_retries:
            # try:
            start_time = time.monotonic()
            print('getting data...')

            # dataset_csv = pd.read_csv(io.StringIO(
            #     chess_dataset.get_dataset_as_csv(100000)), header=None)

            # dataset_csv.fillna(value=0, inplace=True)

            # dataset_features = np.array(
            #     dataset_csv.drop(columns=[896]))
            # dataset_labels = to_categorical(
            #     dataset_csv[896], num_classes=1837)

            # datasetTensor = tf.data.Dataset.from_tensor_slices(
            #     (tf.reshape(tf.constant(dataset_features), [-1, 8, 8, 14]), dataset_labels))

            # Get dataset
            dataset = chess_dataset.get_dataset_as_csv(
                100000)

            # Extracting features (xs) and labels (ys)
            features = [item['xs'] for item in dataset]
            labels = [item['ys'] for item in dataset]

            # Converting features and labels into numpy arrays
            features_array = np.array(features)
            labels_array = np.array(labels)

            # One-hot encode the labels
            labels_onehot = to_categorical(labels_array, num_classes=1837)

            # Creating a TensorFlow dataset
            datasetTensor = tf.data.Dataset.from_tensor_slices(
                (tf.reshape(tf.constant(features_array), [-1, 8, 8, 14]), labels_onehot))

            datasetTensor = datasetTensor.shuffle(
                100).batch(self.batch_size)
            end_time = time.monotonic()
            logging.info(
                f"http GET {end_time - start_time:.3f}s")
            return datasetTensor
            # except Exception as e:
            #     logging.warning(
            #         f"Error while getting data: {e}. Retrying in {retry_interval} seconds...")
            #     retries += 1
            #     time.sleep(retry_interval)
        logging.error(f"Failed to get data after {max_retries} retries.")
        return None
