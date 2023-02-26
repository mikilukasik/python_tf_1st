import tensorflow as tf
from urllib.request import urlopen
import json
import numpy as np

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

datasetReaderId = json.loads(
    urlopen("http://localhost:3500/datasetReader").read())["id"]
print("datasetReaderId", datasetReaderId)

model = tf.keras.models.load_model('./models/blanks/c3d1_v1')

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='categorical_crossentropy',
              metrics=['categorical_crossentropy'])

# testDataset = json.loads(
#     urlopen("http://localhost:3500/datasetReader/" + datasetReaderId + "/testDataset").read())

# print("testDataset length", len(testDataset['xs']))

dataset = json.loads(
    urlopen("http://localhost:3500/datasetReader/" + datasetReaderId + "/dataset").read())

print("dataset length", len(dataset['xs']))


datasetTensor = tf.data.Dataset.from_tensor_slices((tf.reshape(tf.constant(
    np.asarray(dataset["xs"])), [-1, 8, 8, 14]), np.asarray(dataset["ys"])))


# tdst = tf.reshape(tf.constant(
#     np.asanyarray(testDataset["xs"])), [-1, 8, 8, 14])
# testDatasetTensor = tf.data.Dataset.from_tensor_slices(
#     (tdst, np.asanyarray(testDataset["ys"])))

datasetTensor = datasetTensor.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
# testDatasetTensor = testDatasetTensor.batch(BATCH_SIZE)


model.fit(datasetTensor, epochs=3)
