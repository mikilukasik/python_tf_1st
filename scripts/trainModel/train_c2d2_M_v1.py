import tensorflow as tf
from urllib.request import urlopen
import json
import numpy as np

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

datasetReaderId = json.loads(
    urlopen("http://localhost:3500/datasetReader").read())["id"]
print("datasetReaderId", datasetReaderId)

# model = tf.keras.models.load_model('./models/c2d2_M_v1/5.450321197509766')
model = tf.keras.models.load_model('./models/blanks/c2d2_M_v1')

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.00001), loss='categorical_crossentropy',
              metrics=['categorical_crossentropy'])

model.summary()

# testDataset = json.loads(
#     urlopen("http://localhost:3500/datasetReader/" + datasetReaderId + "/testDataset").read())

# print("testDataset length", len(testDataset['xs']))

# tdst = tf.reshape(tf.constant(
#     np.asanyarray(testDataset["xs"])), [-1, 8, 8, 14])
# testDatasetTensor = tf.data.Dataset.from_tensor_slices(
#     (tdst, np.asanyarray(testDataset["ys"])))

# testDatasetTensor = testDatasetTensor.batch(BATCH_SIZE)

for x in range(10000):
    for x in range(30):
        dataset = json.loads(
            urlopen("http://localhost:3500/datasetReader/" + datasetReaderId + "/dataset").read())

        print("loaded dataset length", len(dataset['xs']))

        datasetTensor = tf.data.Dataset.from_tensor_slices((tf.reshape(tf.constant(
            np.asarray(dataset["xs"])), [-1, 8, 8, 14]), np.asarray(dataset["ys"])))

        datasetTensor = datasetTensor.shuffle(
            SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

        fitResult = model.fit(datasetTensor, epochs=1)

        print(fitResult.history["loss"][0])

    model.save('./models/c2d2_M_v1/'+str(fitResult.history["loss"][0]))
    print('model saved.')
