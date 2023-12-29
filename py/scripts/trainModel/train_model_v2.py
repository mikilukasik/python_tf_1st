import tensorflow as tf
from keras.utils import to_categorical
from urllib.request import urlopen, Request
import json
import numpy as np
import gzip
import pandas as pd
import shutil
from collections import deque
from keras.models import model_from_json
import os

SHUFFLE_BUFFER_SIZE = 100


iterations_with_no_improvement = 0
lastSavedAvg = 9999


def save_model(model, avg, model_dest, is_temp=False):
    foldername = os.path.join(model_dest, str(
        avg) + '_temp' if is_temp else '')
    os.makedirs(foldername, exist_ok=True)
    model.save_weights(os.path.join(foldername, 'weights.h5'))
    with open(os.path.join(foldername, 'model.json'), 'w') as json_file:
        json_file.write(model.to_json())

    print('model saved.    * * * * * * * * * * * * * * * * * * * * * * * *')
    print('                * * * * * * * ', avg, ' * * * * * * * ')
    print('                * * * * * * * * * * * * * * * * * * * * * * * *')

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

    # if saveIfShould.counter > 0 or len(avgQ10) < 6:
    #     return

    avg10 = get_avg(avgQ10)
    avg50 = get_avg(avgQ50)
    avg250 = get_avg(avgQ250)

    avg = (avg10 + avg50 * 3 + avg250) / 5

    print('(avg, 10, 50, 250)', avg, avg10, avg50, avg250)

    if avg < lastSavedAvg:
        save_model(model, avg, model_dest=model_dest)
        iterations_with_no_improvement = 0
        lastSavedAvg = avg

    if (iterations_with_no_improvement > 50):
        save_model(model, avg, model_dest, True)
        iterations_with_no_improvement = 0


def train_model(model_source, model_dest, BATCH_SIZE=256, learning_rate=0.0003, from_json=False):
    if from_json:
        with open(model_source + '/model.json', 'r') as f:
            model_json = f.read()

            model = model_from_json(model_json)
            model.load_weights(model_source + '/weights.h5')
    else:
        model = tf.keras.models.load_model(model_source)

    datasetReaderId = json.loads(
        urlopen("http://localhost:3550/datasetReader").read())["id"]

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate), loss='categorical_crossentropy',
                  metrics=['categorical_crossentropy'])

    model.summary()

    while True:
        datasetCsv = pd.read_csv("http://localhost:3550/datasetReader/" +
                                 datasetReaderId + "/dataset?format=csv", header=None)
        dataset_features = np.array(datasetCsv.drop(columns=[896]))
        dataset_labels = to_categorical(datasetCsv[896], num_classes=1837)
        datasetTensor = tf.data.Dataset.from_tensor_slices(
            (tf.reshape(tf.constant(dataset_features), [-1, 8, 8, 14]), dataset_labels))
        datasetTensor = datasetTensor.shuffle(
            SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        val = model.fit(datasetTensor, epochs=1).history["loss"][0]
        saveIfShould(model, val, model_dest=model_dest)
