from .print_large import print_large
from .save_model import save_model
from .load_model import load_model, load_model_meta
from .helpers.training_manager import TrainingManager
from .helpers.dataset_provider import DatasetProvider

from collections import deque
import shutil
from urllib.request import urlopen
from keras.utils import to_categorical
import tensorflow as tf
import os
import time
import logging
from keras.callbacks import LearningRateScheduler

logging.basicConfig(level=logging.INFO)

model_meta = {}
iterations_with_no_improvement = 0
lastSavedAvg = 9999
batch_size = 256
next_lr = 0.0001
training_manager = None


def save_model_and_delete_last(model, avg, model_dest, is_temp=False):
    foldername = os.path.join(model_dest, str(
        avg) + ('_temp' if is_temp else ''))
    save_model(model, foldername, model_meta)
    training_manager.save_stats(foldername, True)

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
    global avgQ10, avgQ50, avgQ250

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
        iterations_with_no_improvement = 0


def train_model(model_source, model_dest, initial_batch_size=256, initial_lr=0.0003, gpu=True, force_lr=False):
    device = '/gpu:0' if gpu else '/cpu:0'
    with tf.device(device):
        return train_model_v3(model_source, model_dest, initial_batch_size, initial_lr, gpu, force_lr)


def train_model_v3(model_source, model_dest, initial_batch_size=256, initial_lr=0.0003, gpu=True, force_lr=False):
    global model_meta, batch_size, next_lr, training_manager

    batch_size = initial_batch_size

    # Load the Keras model and its metadata
    model = load_model(model_source)
    model_meta = load_model_meta(model_source)
    model.summary()

    training_manager = TrainingManager(
        initial_lr=initial_lr, initial_batch_size=initial_batch_size, model_meta=model_meta, force_lr=force_lr)
    dataset_provider = DatasetProvider(model_meta, initial_batch_size)

    optimizer = training_manager.get_optimizer()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['categorical_crossentropy'])

    lr_scheduler_callback = LearningRateScheduler(training_manager.get_next_lr)

    while True:
        start_time = time.time()
        datasetTensor = dataset_provider.get_next_batch()
        logging.info(
            f"Loaded dataset in {time.time() - start_time:.2f} seconds")

        training_manager.print_stats()

        # Train the model on the dataset
        start_time = time.time()
        val = model.fit(datasetTensor, epochs=1,
                        callbacks=[lr_scheduler_callback]
                        ).history["loss"][0]
        logging.info(
            f"Trained model on dataset in {time.time() - start_time:.2f} seconds")

        training_manager.add_to_stats(loss=val, lr=model.optimizer.lr.numpy(), time=time.time(
        ) - start_time, sample_size=50000, batch_size=batch_size, gpu=gpu)

        # Save the model if necessary
        saveIfShould(model, val, model_dest=model_dest)
