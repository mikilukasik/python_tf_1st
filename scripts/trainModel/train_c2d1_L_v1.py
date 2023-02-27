import tensorflow as tf
from keras.utils import to_categorical
from urllib.request import urlopen, Request
import json
import numpy as np
import gzip
import pandas as pd
import shutil

with tf.device('/cpu:0'):

    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 100

    datasetReaderId = json.loads(
        urlopen("http://localhost:3500/datasetReader").read())["id"]
    print("datasetReaderId", datasetReaderId)

    model = tf.keras.models.load_model('./models/c2d1_L_v1/2.3578758239746094')
    # model = tf.keras.models.load_model('./models/blanks/c2d1_L_v1')

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.000005), loss='categorical_crossentropy',
                  metrics=['categorical_crossentropy'])

    model.summary()
    lastSavedLoss = 9999

    for x in range(100000):
        datasetCsv = pd.read_csv(
            "http://localhost:3500/datasetReader/" + datasetReaderId + "/dataset?format=csv", header=None)

        dataset_features = datasetCsv.copy()
        dataset_labels = dataset_features.pop(896)

        dataset_features = np.array(dataset_features)
        dataset_labels = to_categorical(dataset_labels, num_classes=1837)

        datasetTensor = tf.data.Dataset.from_tensor_slices((tf.reshape(tf.constant(
            dataset_features), [-1, 8, 8, 14]), dataset_labels))

        datasetTensor = datasetTensor.shuffle(
            SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

        fitResult = model.fit(datasetTensor, epochs=1).history["loss"][0]

        if fitResult < lastSavedLoss:
            model.save('./models/c2d1_L_v1/' + str(fitResult))
            print('* * * * * * model saved.', fitResult)

            if lastSavedLoss < 9999:
                shutil.rmtree(r'./models/c2d1_L_v1/' + str(lastSavedLoss))
                print('deleted old:', lastSavedLoss)

            lastSavedLoss = fitResult
