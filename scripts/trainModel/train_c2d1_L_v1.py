import tensorflow as tf
from keras.utils import to_categorical
from urllib.request import urlopen, Request
import json
import numpy as np
import gzip
import pandas as pd

# with tf.device('/cpu:0'):

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 100

datasetReaderId = json.loads(
    urlopen("http://localhost:3500/datasetReader").read())["id"]
print("datasetReaderId", datasetReaderId)

model = tf.keras.models.load_model('./models/c2d1_L_v1/2.388428211212158')
# model = tf.keras.models.load_model('./models/blanks/c2d1_L_v1')

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0000005), loss='categorical_crossentropy',
              metrics=['categorical_crossentropy'])

model.summary()

for x in range(10000):
    for y in range(30):

        datasetCsv = pd.read_csv(
            "http://localhost:3500/datasetReader/" + datasetReaderId + "/dataset?format=csv", header=None)
        print("loaded dataset.", y)

        dataset_features = datasetCsv.copy()
        dataset_labels = dataset_features.pop(896)

        dataset_features = np.array(dataset_features)
        dataset_labels = to_categorical(dataset_labels, num_classes=1837)

        datasetTensor = tf.data.Dataset.from_tensor_slices((tf.reshape(tf.constant(
            dataset_features), [-1, 8, 8, 14]), dataset_labels))

        datasetTensor = datasetTensor.shuffle(
            SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

        fitResult = model.fit(datasetTensor, epochs=1)

    model.save('./models/c2d1_L_v1/'+str(fitResult.history["loss"][0]))
    print('* * * * * * model saved.', fitResult.history["loss"][0])
