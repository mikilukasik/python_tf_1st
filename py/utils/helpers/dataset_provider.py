import time
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
import logging
import requests
from . import prefetch
from ..print_large import print_large


class DatasetProvider:
    def __init__(self, model_meta, batch_size, ys_format='default', xs_format='default', dataset_reader_version='16'):
        self.dataset_reader_id = model_meta.get("dataseReaderId")
        self.batch_size = batch_size
        self.ys_format = ys_format
        self.xs_format = xs_format
        self.dataset_reader_version = dataset_reader_version

        if not self.dataset_reader_id:
            dataset_reader_response = requests.get(
                "http://localhost:3500/datasetReader?ysformat="+self.ys_format+'&xsformat='+self.xs_format+'&readerVersion='+self.dataset_reader_version)
            self.dataset_reader_id = dataset_reader_response.json().get("id")
            model_meta["dataseReaderId"] = self.dataset_reader_id
            print_large("", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "", "New dataset_reader_id retrieved:",
                        self.dataset_reader_id, "", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "")

        prefetch.set_data_getter(self.get_dataset)
        prefetch.prefetch_data()

    def get_next_batch(self):
        data = prefetch.get_data()
        prefetch.prefetch_data()
        return data

    def get_dataset(self, url, max_retries=50, retry_interval=5):
        retries = 0
        while retries < max_retries:
            try:
                start_time = time.monotonic()
                print('calling API')
                dataset_csv = pd.read_csv("http://localhost:3500/datasetReader/" +
                                          self.dataset_reader_id + "/dataset?format=csv&ysformat="+self.ys_format+'&xsformat='+self.xs_format+'&readerVersion='+self.dataset_reader_version, header=None, na_values=[''])
                dataset_csv.fillna(value=0, inplace=True)

                if self.ys_format == '1966':
                    # TODO: xs_format needs doing here

                    dataset_features = np.array(
                        dataset_csv.drop(columns=[896, 897, 898, 899]))

                    class_labels_one_hot = to_categorical(
                        dataset_csv[896], num_classes=1837)

                    from_labels_one_hot = to_categorical(
                        dataset_csv[897], num_classes=64)
                    to_labels_one_hot = to_categorical(
                        dataset_csv[898], num_classes=64)

                    dataset_labels = np.concatenate(
                        [class_labels_one_hot, from_labels_one_hot, to_labels_one_hot, dataset_csv[899].values.reshape(-1, 1)], axis=1)

                else:
                    if self.xs_format == '39':
                        dataset_features = np.array(
                            dataset_csv.drop(columns=[2496]))
                        dataset_labels = to_categorical(
                            dataset_csv[2496], num_classes=1837)
                    else:
                        dataset_features = np.array(
                            dataset_csv.drop(columns=[896]))
                        dataset_labels = to_categorical(
                            dataset_csv[896], num_classes=1837)

                datasetTensor = tf.data.Dataset.from_tensor_slices(
                    (tf.reshape(tf.constant(dataset_features), [-1, 8, 8, 39 if self.xs_format == '39' else 14]), dataset_labels))
                datasetTensor = datasetTensor.shuffle(
                    100).batch(self.batch_size)
                end_time = time.monotonic()
                logging.info(
                    f"http GET {end_time - start_time:.3f}s")
                return datasetTensor
            except Exception as e:
                logging.warning(
                    f"Error while getting data: {e}. Retrying in {retry_interval} seconds...")
                retries += 1
                time.sleep(retry_interval)
        logging.error(f"Failed to get data after {max_retries} retries.")
        return None
