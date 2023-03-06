from .print_large import print_large
from .save_model import save_model
from .load_model import load_model, load_model_meta

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
import time
import logging
import requests


# Set up logging
logging.basicConfig(level=logging.INFO)

model_meta = {}
iterations_with_no_improvement = 0
lastSavedAvg = 9999

needs_lr_reduced = False
last_lr_reduce_time = 0
min_time_between_lr_reduces = 1 * 60 * 60  # 2 hours in seconds


def lr_schedule(epoch, lr):
    global needs_lr_reduced, last_lr_reduce_time

    current_time = time.time()
    time_since_last_run = current_time - last_lr_reduce_time

    if needs_lr_reduced and time_since_last_run >= min_time_between_lr_reduces:
        needs_lr_reduced = False
        last_lr_reduce_time = current_time

        new_lr = lr * 0.5
        print_large(f"Learning rate goes from {lr} to {new_lr}")
        model_meta['lr'] = new_lr
        return new_lr

    return lr


def save_model_and_delete_last(model, avg, model_dest, is_temp=False):
    foldername = os.path.join(model_dest, str(
        avg) + ('_temp' if is_temp else ''))
    save_model(model, foldername, model_meta)

    if lastSavedAvg < 9999 and not is_temp:
        old_foldername = os.path.join(model_dest, str(lastSavedAvg))
        shutil.rmtree(old_foldername)
        print('deleted old:', old_foldername)


avgQ10 = deque(maxlen=10)
avgQ50 = deque(maxlen=50)
avgQ51to100 = deque(maxlen=50)
avgQ250 = deque(maxlen=250)


def get_avg(avgQ):
    if len(avgQ) == 0:
        return 0
    else:
        return sum(avgQ)/len(avgQ)


def appendToAvg(val):
    global needs_lr_reduced
    global avgQ51to100
    global avgQ10
    global avgQ50
    global avgQ250

    avgQ10.append(val)
    avgQ50.append(val)
    avgQ250.append(val)

    if len(avgQ50) == 50:
        old_val = avgQ50.popleft()
        avgQ51to100.append(old_val)

        if len(avgQ51to100) == 50:
            past_50_diff = get_avg(avgQ50) - get_avg(avgQ51to100)
            print(f"Past 50 iterations loss diff: {past_50_diff}")

            if past_50_diff > 0:
                avgQ51to100 = deque(maxlen=50)
                needs_lr_reduced = True


def saveIfShould(model, val, model_dest):
    global iterations_with_no_improvement
    global lastSavedAvg
    global needs_lr_reduced

    appendToAvg(val)

    iterations_with_no_improvement += 1

    if not hasattr(saveIfShould, "counter"):
        saveIfShould.counter = 0  # Initialize the counter
    saveIfShould.counter = (saveIfShould.counter + 1) % 3

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
        # needs_lr_reduced = True
        iterations_with_no_improvement = 0


def train_model(model_source, model_dest, BATCH_SIZE=256, initial_learning_rate=0.0003):
    global model_meta

    # Load the Keras model and its metadata
    model = load_model(model_source)
    model_meta = load_model_meta(model_source)
    model_meta['lr'] = initial_learning_rate

    model.summary()

    # Get the dataset reader ID from the API
    dataset_reader_id = model_meta.get("dataseReaderId")
    if not dataset_reader_id:
        dataset_reader_response = requests.get(
            "http://localhost:3500/datasetReader")
        dataset_reader_id = dataset_reader_response.json().get("id")
        model_meta["dataseReaderId"] = dataset_reader_id
        print_large("", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "", "New dataset_reader_id retrieved:",
                    dataset_reader_id, "", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "")

    def data_getter(url, max_retries=120, retry_interval=5):
        retries = 0
        while retries < max_retries:
            try:
                start_time = time.monotonic()
                dataset_csv = pd.read_csv("http://localhost:3500/datasetReader/" +
                                          dataset_reader_id + "/dataset?format=csv", header=None, na_values=[''])
                dataset_csv.fillna(value=0, inplace=True)
                dataset_features = np.array(dataset_csv.drop(columns=[896]))
                dataset_labels = to_categorical(
                    dataset_csv[896], num_classes=1837)
                datasetTensor = tf.data.Dataset.from_tensor_slices(
                    (tf.reshape(tf.constant(dataset_features), [-1, 8, 8, 14]), dataset_labels))
                datasetTensor = datasetTensor.shuffle(100).batch(BATCH_SIZE)
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

    prefetch.set_data_getter(data_getter)
    prefetch.prefetch_data()

    # Initialize the optimizer
    optimizer = tf.keras.optimizers.legacy.Adam(initial_learning_rate)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['categorical_crossentropy'])

    # Define a LambdaCallback to log the learning rate
    log_lr_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: logging.info(
            f"Learning rate: {tf.keras.backend.get_value(model.optimizer.lr)}")
    )

    while True:
        # Load the dataset from the server
        start_time = time.time()

        datasetTensor = prefetch.get_data()
        prefetch.prefetch_data()

        logging.info(
            f"Loaded dataset in {time.time() - start_time:.2f} seconds")

        # Train the model on the dataset
        start_time = time.time()
        val = model.fit(datasetTensor, epochs=1, callbacks=[
                        tf.keras.callbacks.LearningRateScheduler(lr_schedule),
                        log_lr_callback]).history["loss"][0]
        logging.info(
            f"Trained model on dataset in {time.time() - start_time:.2f} seconds")

        # Save the model if necessary
        saveIfShould(model, val, model_dest=model_dest)
