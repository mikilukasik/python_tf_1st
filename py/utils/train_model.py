
from .save_model import save_model
from .load_model import load_model
from . import prefetch
from collections import deque
import shutil
import pandas as pd
import numpy as np
import json
from urllib.request import urlopen
from keras.utils import to_categorical
import tensorflow as tf
import os
# from utils import load_model, print_large, save_model
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)


iterations_with_no_improvement = 0
lastSavedAvg = 9999


def save_model_and_delete_last(model, avg, model_dest, is_temp=False):
    foldername = os.path.join(model_dest, str(
        avg) + ('_temp' if is_temp else ''))
    save_model(model, foldername)

    if lastSavedAvg < 9999 and not is_temp:
        old_foldername = os.path.join(model_dest, str(lastSavedAvg))
        shutil.rmtree(old_foldername)
        print('deleted old:', old_foldername)


avgQ10 = deque(maxlen=10)
avgQ50 = deque(maxlen=50)
avgQ250 = deque(maxlen=250)


def get_avg(avgQ):
    if len(avgQ) == 0:
        return 0
    else:
        return sum(avgQ)/len(avgQ)


def appendToAvg(val):
    avgQ10.append(val)
    avgQ50.append(val)
    avgQ250.append(val)


def saveIfShould(model, val, model_dest):
    global iterations_with_no_improvement
    global lastSavedAvg

    appendToAvg(val)

    iterations_with_no_improvement += 1

    if not hasattr(saveIfShould, "counter"):
        saveIfShould.counter = 0  # Initialize the counter
    saveIfShould.counter = (saveIfShould.counter + 1) % 5

    if saveIfShould.counter > 0 or len(avgQ10) < 6:
        return

    avg10 = get_avg(avgQ10)
    avg50 = get_avg(avgQ50)
    avg250 = get_avg(avgQ250)

    avg = (avg10 + avg50 + avg250) / 3

    print('(avg, 10, 50, 250)', avg, avg10, avg50, avg250)

    if avg < lastSavedAvg:
        save_model_and_delete_last(model, avg, model_dest=model_dest)
        iterations_with_no_improvement = 0
        lastSavedAvg = avg

    if (iterations_with_no_improvement > 50):
        save_model_and_delete_last(model, avg, model_dest, True)
        iterations_with_no_improvement = 0


def train_model(model_source, model_dest, BATCH_SIZE=256, learning_rate=0.0003):
    model = load_model(model_source)

    datasetReaderId = json.loads(
        urlopen("http://localhost:3500/datasetReader").read())["id"]

    def data_getter(url):
        dataset_csv = pd.read_csv("http://localhost:3500/datasetReader/" +
                                  datasetReaderId + "/dataset?format=csv", header=None, na_values=[''])
        dataset_csv.fillna(value=0, inplace=True)
        dataset_features = np.array(dataset_csv.drop(columns=[896]))
        dataset_labels = to_categorical(dataset_csv[896], num_classes=1837)
        datasetTensor = tf.data.Dataset.from_tensor_slices(
            (tf.reshape(tf.constant(dataset_features), [-1, 8, 8, 14]), dataset_labels))
        datasetTensor = datasetTensor.shuffle(100).batch(BATCH_SIZE)
        end_time = time.monotonic()
        return datasetTensor

    prefetch.set_data_getter(data_getter)
    prefetch.prefetch_data()

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate), loss='categorical_crossentropy',
                  metrics=['categorical_crossentropy'])

    model.summary()

    while True:
        # Load the dataset from the server
        start_time = time.time()

        datasetTensor = prefetch.get_data()
        prefetch.prefetch_data()

        logging.info(
            f"Loaded dataset in {time.time() - start_time:.2f} seconds")

        # Train the model on the dataset
        start_time = time.time()
        val = model.fit(datasetTensor, epochs=1).history["loss"][0]
        logging.info(
            f"Trained model on dataset in {time.time() - start_time:.2f} seconds")

        # Save the model if necessary
        saveIfShould(model, val, model_dest=model_dest)
