import tensorflow as tf
from keras.utils import to_categorical
from urllib.request import urlopen, Request
import json
import numpy as np
import gzip
import pandas as pd
import shutil
from collections import deque
import random
# import keyboard

with tf.device('/cpu:0'):

    BATCH_SIZE = 128
    SHUFFLE_BUFFER_SIZE = 100

    datasetReaderId = json.loads(
        urlopen("http://localhost:3550/datasetReader").read())["id"]
    print("datasetReaderId", datasetReaderId)

    model = tf.keras.models.load_model(
        './models/c3d3_S_v1/250_2.42795224571228')
    # model = tf.keras.models.load_model('./models/blanks/c3d3_S_v1')

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.000001), loss='categorical_crossentropy',
                  metrics=['categorical_crossentropy'])

    model.summary()
    lastSavedAvg10 = 9999
    avgQ10 = deque(maxlen=10)
    lastSavedAvg50 = 9999
    avgQ50 = deque(maxlen=50)
    lastSavedAvg250 = 9999
    avgQ250 = deque(maxlen=250)

    iterations_with_no_improvement = 0

    def get_avg(avgQ):
        if len(avgQ) == 0:
            return 0
        else:
            return sum(avgQ)/len(avgQ)

    def saveModel(model, avg, lastSavedAvg, qName):
        model.save('./models/c3d3_S_v1/' + qName + '_' + str(avg))
        print('model saved.    * * * * * * * * * * * * * * * * * * * * * * * *')
        print('                * * * * * * ', qName, avg, ' * * * * * * ')
        print('                * * * * * * * * * * * * * * * * * * * * * * * *')

        if lastSavedAvg < 9999:
            shutil.rmtree(r'./models/c3d3_S_v1/' +
                          qName + '_' + str(lastSavedAvg))
            print('deleted old:', qName, lastSavedAvg)

    def appendToAvg(val):
        avgQ10.append(val)
        avgQ50.append(val)
        avgQ250.append(val)

    def saveIfShould(model, val):
        global iterations_with_no_improvement
        global lastSavedAvg10
        global lastSavedAvg50
        global lastSavedAvg250

        appendToAvg(val)

        iterations_with_no_improvement += 1

        if len(avgQ10) < 6:
            return

        avg10 = get_avg(avgQ10)
        avg50 = get_avg(avgQ50)
        avg250 = get_avg(avgQ250)

        print('avg (10, 50, 250)', avg10, avg50, avg250)

        if avg10 < lastSavedAvg10:
            saveModel(model, avg10, lastSavedAvg10, '10')
            lastSavedAvg10 = avg10
            iterations_with_no_improvement = 0

        if avg50 < lastSavedAvg50:
            saveModel(model, avg50, lastSavedAvg50, '50')
            lastSavedAvg50 = avg50
            iterations_with_no_improvement = 0

        if avg250 < lastSavedAvg250:
            saveModel(model, avg250, lastSavedAvg250, '250')
            lastSavedAvg250 = avg250
            iterations_with_no_improvement = 0

        if (iterations_with_no_improvement > 50):
            model.save('./models/c3d3_S_v1/X_' + str(avg50))
            print('extra model saved.   * * * * * * ', avg50, ' * * * * * * ')
            iterations_with_no_improvement = 0

    for x in range(100000):
        datasetCsv = pd.read_csv(
            "http://localhost:3550/datasetReader/" + datasetReaderId + "/dataset?format=csv", header=None)

        dataset_features = datasetCsv.copy()
        dataset_labels = dataset_features.pop(896)

        dataset_features = np.array(dataset_features)
        dataset_labels = to_categorical(dataset_labels, num_classes=1837)

        datasetTensor = tf.data.Dataset.from_tensor_slices((tf.reshape(tf.constant(
            dataset_features), [-1, 8, 8, 14]), dataset_labels))

        datasetTensor = datasetTensor.shuffle(
            SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

        val = model.fit(datasetTensor, epochs=1).history["loss"][0]
        # avg = get_avg()
        # print('avg:', avg)

        saveIfShould(model, val)

        # if avg < lastSavedAvg10:
        # model.save('./models/c3d3_S_v1/' + str(avg))
        # print('model saved.    * * * * * * ', avg, ' * * * * * * ')

        # if lastSavedAvg10 < 9999:
        #     shutil.rmtree(r'./models/c3d3_S_v1/' + str(lastSavedAvg10))
        #     print('deleted old:', lastSavedAvg10)

        # lastSavedAvg10 = avg
        # iterations_with_no_improvement = 0
