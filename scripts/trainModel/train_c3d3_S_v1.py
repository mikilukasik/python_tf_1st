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
        urlopen("http://localhost:3500/datasetReader").read())["id"]
    print("datasetReaderId", datasetReaderId)

    model = tf.keras.models.load_model('./models/c3d3_S_v1/2.4398773908615112')
    # model = tf.keras.models.load_model('./models/blanks/c3d3_S_v1')

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.000001), loss='categorical_crossentropy',
                  metrics=['categorical_crossentropy'])

    model.summary()
    lastSavedAvg = 9999
    avgQ = deque(maxlen=10)

    # def saveModel():
    #     print('It works! You typed save and triggered this function.')

    # keyboard.add_hotkey('s, a, v, e', saveModel)
    # keyboard.wait()

    def get_avg():
        if len(avgQ) == 0:
            return 0
        else:
            return sum(avgQ)/len(avgQ)

    iterations_with_no_improvement = 0

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

        avgQ.append(model.fit(datasetTensor, epochs=1).history["loss"][0])
        avg = get_avg()
        print('avg:', avg)

        if avg < lastSavedAvg:
            model.save('./models/c3d3_S_v1/' + str(avg))
            print('model saved.    * * * * * * ', avg, ' * * * * * * ')

            if lastSavedAvg < 9999:
                shutil.rmtree(r'./models/c3d3_S_v1/' + str(lastSavedAvg))
                print('deleted old:', lastSavedAvg)

            lastSavedAvg = avg
            iterations_with_no_improvement = 0

        else:
            iterations_with_no_improvement += 1

            if (iterations_with_no_improvement > 30 and random.randint(0, 20) == 20):
                model.save('./models/c3d3_S_v1/x_' + str(avg))
                print('extra model saved.    * * * * * * ', avg, ' * * * * * * ')
