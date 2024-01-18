from .print_large import print_large
from .save_model import save_model
from .load_model import load_model, load_model_meta
from .helpers.training_manager_v3 import TrainingManager
from .helpers.dataset_provider2 import DatasetProvider
# from .helpers.dataset_provider import DatasetProvider
from utils.plot_model_meta import plot_model_meta

from collections import deque
import shutil
from urllib.request import urlopen
from keras.utils import to_categorical
from keras.layers import Dropout
from keras.models import Model
import tensorflow as tf
import os
import time
import logging
from keras.callbacks import LearningRateScheduler
import sys
from flask import Flask, request, send_file, after_this_request
from threading import Thread, Lock
import zipfile


logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

model_lock = Lock()  # Lock for thread-safe model access

model_meta = {}
iterations_with_no_improvement = 0
lastSavedAvg = 9999
currentBest = 9999
batch_size = 256
next_lr = 0.0001
training_manager = None

save_requested = False
set_lr_multiplier_to = None

latest_model_source = None


@app.route('/save', methods=['GET', 'POST'])
def save_endpoint():
    global save_requested
    save_requested = True
    print('save requested')

    lr_multiplier_in_query = request.args.get('lr')
    if lr_multiplier_in_query:
        # validate it to be between 100 and 0.0001
        if float(lr_multiplier_in_query) < 0.0001 or float(lr_multiplier_in_query) > 100:
            return "lr_multiplier should be between 100 and 0.0001"

        global set_lr_multiplier_to
        set_lr_multiplier_to = float(lr_multiplier_in_query)
        print('set_lr_multiplier_to', set_lr_multiplier_to)

    return "Save method called"


def zipdir(path, ziph):
    # Function to zip the contents of a directory
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), os.path.relpath(
                os.path.join(root, file), os.path.join(path, '..')))


@app.route('/download', methods=['GET'])
def download_model():
    global latest_model_source
    if latest_model_source is None:
        return "No model has been saved yet"

    print('Downloading model from', latest_model_source)

    # Create a temporary directory
    temp_dir = os.path.join(os.getcwd(), 'temp_model_zip')
    os.makedirs(temp_dir, exist_ok=True)

    # Zip the model files
    zipf = zipfile.ZipFile(os.path.join(temp_dir, 'model.zip'), 'w')
    zipdir(latest_model_source, zipf)
    zipf.close()

    @after_this_request
    def cleanup(response):
        # Cleanup: Delete the temporary directory after sending the file
        shutil.rmtree(temp_dir, ignore_errors=True)
        return response

    # Send the zipped model files
    return send_file(os.path.join(temp_dir, 'model.zip'), as_attachment=True)


@app.route('/stats', methods=['GET'])
def get_stats_html():

    print('get_stats_html')
    return plot_model_meta(model_meta, None, False, title='Loss and learning rate history')


def modify_dropout_rates(model, new_rate):
    with model_lock:
        for layer in model.layers:
            if isinstance(layer, Dropout):
                layer.rate = new_rate
        # Rebuild model (only necessary in some Keras versions)
        return Model(inputs=model.inputs, outputs=model.outputs)


def save_model_and_delete_last(model, val, model_dest, is_temp=False, isCurrentBest=False):
    global lastSavedAvg, currentBest, save_requested, latest_model_source

    foldername = os.path.join(model_dest, str(
        val) + ('_temp' if is_temp else '') + ('_best' if isCurrentBest else ''))
    with model_lock:
        save_model(model, foldername, model_meta)
    # training_manager.save_stats(foldername, True)
    latest_model_source = foldername
    save_requested = False

    if lastSavedAvg < 9999 and not is_temp and not isCurrentBest:
        old_foldername = os.path.join(model_dest, str(lastSavedAvg))
        shutil.rmtree(old_foldername)
        print('deleted old:', old_foldername)

    if currentBest < 9999 and not is_temp and isCurrentBest:
        old_foldername = os.path.join(model_dest, str(currentBest)+'_best')
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
    global currentBest
    global save_requested

    appendToAvg(val)
    iterations_with_no_improvement += 1

    if not hasattr(saveIfShould, "counter"):
        saveIfShould.counter = 0  # Initialize the counter
    saveIfShould.counter = (saveIfShould.counter + 1) % 3

    # if saveIfShould.counter > 0 or len(avgQ10) < 6:
    #     return

    if len(avgQ10) < 6 and not save_requested:
        return

    avg10 = get_avg(avgQ10)
    avg50 = get_avg(avgQ50)
    avg250 = get_avg(avgQ250)

    avg = (val+avg10 + avg50 + avg250) / 4

    print('(avg, 10, 50, 250)', avg, avg10, avg50, avg250)

    if val < currentBest:
        save_model_and_delete_last(model, val, model_dest, False, True)
        iterations_with_no_improvement = 0
        currentBest = val

    if avg < lastSavedAvg:
        save_model_and_delete_last(model, avg, model_dest=model_dest)
        iterations_with_no_improvement = 0
        lastSavedAvg = avg

    if (iterations_with_no_improvement > 50 or save_requested):
        save_model_and_delete_last(model, avg, model_dest, True)
        iterations_with_no_improvement = 0


def train_model(model_source, model_dest, initial_batch_size=256, initial_lr=0.0003, gpu=True, force_lr=False, lr_multiplier=None, ys_format='default', xs_format='default', fixed_lr=None, dataset_reader_version='16', filter='default', evaluateOnly=False, make_trainable=False, evaluate=True, fresh_reader=False, dropout_rate=None):
    device = '/gpu:0' if gpu else '/cpu:0'
    with tf.device(device):
        return train_model_v4(model_source, model_dest, initial_batch_size, initial_lr, gpu, force_lr, lr_multiplier, ys_format, xs_format, fixed_lr, dataset_reader_version, filter, evaluateOnly, make_trainable, evaluate, fresh_reader, dropout_rate)


def train_model_v4(model_source, model_dest, initial_batch_size=256, initial_lr=0.0003, gpu=True, force_lr=False, lr_multiplier=None, ys_format='default', xs_format='default', fixed_lr=None, dataset_reader_version='16', filter='default', evaluateOnly=False, make_trainable=False, evaluate=True, fresh_reader=False, dropout_rate=None):
    global model_meta, batch_size, next_lr, training_manager, set_lr_multiplier_to

    batch_size = initial_batch_size

    # Load the Keras model and its metadata
    with model_lock:
        model = load_model(model_source)
        model_meta = load_model_meta(model_source)

    if make_trainable:
        with model_lock:
            for layer in model.layers:
                layer.trainable = True

    if dropout_rate is not None:
        model = modify_dropout_rates(model, dropout_rate)

    model.summary()

    training_manager = TrainingManager(
        model_meta, batch_size=initial_batch_size, lr_multiplier=lr_multiplier, fixed_lr=fixed_lr)
    dataset_provider = DatasetProvider(
        model_meta, initial_batch_size, ys_format, xs_format, dataset_reader_version, filter, evaluateOnly, fresh_reader=fresh_reader)

    optimizer = training_manager.get_optimizer()

    loss = 'categorical_crossentropy'

    if ys_format == 'winner':
        loss = 'binary_crossentropy'

    if ys_format == 'chkmtOrStallEnding' or ys_format == 'is1stProgGroup' or ys_format == 'nextBalance' or ys_format == 'bal8' or ys_format.startswith('nextBal'):
        loss = tf.keras.losses.MeanSquaredError()

        # ys_format

    if ys_format == '1966':
        def custom_loss(y_true, y_pred):
            # Define the custom loss for the 1837 classes output
            loss_class = tf.keras.losses.CategoricalCrossentropy()(
                y_true[:, :1837], y_pred[:, :1837])

            # Define the custom loss for the "from_labels" output
            loss_from = tf.keras.losses.CategoricalCrossentropy()(
                y_true[:, 1837:1901], y_pred[:, 1837:1901])

            # Define the custom loss for the "to_labels" output
            loss_to = tf.keras.losses.CategoricalCrossentropy()(
                y_true[:, 1901:1965], y_pred[:, 1901:1965])

            # Define the custom loss for the "knight_promo" output
            loss_promo = tf.keras.losses.BinaryCrossentropy()(
                y_true[:, -1], y_pred[:, -1])

            # Return the sum of the four individual losses
            # (loss_class * 1.8 + loss_from + loss_to + loss_promo * 0.2)/4
            return (loss_class * 1.99 + loss_from + loss_to + loss_promo*0.01)/4

        loss = custom_loss

    # print('loss','loss')
    with model_lock:
        model.compile(optimizer=optimizer, loss=loss,
                      metrics=['accuracy'])
        #   metrics=[tf.keras.losses.MeanAbsoluteError()])

    def evaluate_model(datasetTensor):
        with model_lock:
            print('Evaluating model')
            [eval_acc, eval_loss] = model.evaluate(datasetTensor)

            print({eval_acc, eval_loss})
            return [eval_acc, eval_loss]

    if evaluateOnly:
        print('Evaluating only')
        datasetTensor = dataset_provider.get_next_batch()

        evaluate_model(datasetTensor)
        sys.exit()

    else:
        val = 9999
        lr_scheduler_callback = LearningRateScheduler(
            training_manager.get_next_lr)

        last_evaluated = 0

        while True:
            datasetTensor = dataset_provider.get_next_batch()

            if evaluate and time.time() - last_evaluated > 3600:
                [eval_acc, eval_loss] = evaluate_model(datasetTensor)
                training_manager.add_eval_result(eval_loss, eval_acc)
                last_evaluated = time.time()

            start_time = time.time()

            logging.info(
                f"Loaded dataset in {time.time() - start_time:.2f} seconds")

            training_manager.print_stats()

            # Train the model on the dataset

            start_time = time.time()
            with model_lock:
                val = model.fit(datasetTensor, epochs=1,
                                callbacks=[lr_scheduler_callback]
                                ).history["loss"][0]
            logging.info(
                f"Trained model on dataset in {time.time() - start_time:.2f} seconds")

            training_manager.add_to_stats(loss=val, lr=model.optimizer.lr.numpy(), time=time.time(
            ) - start_time, sample_size=50000, batch_size=batch_size, gpu=gpu, set_lr_multiplier_to=set_lr_multiplier_to)
            set_lr_multiplier_to = None

            # Save the model if necessary
            saveIfShould(model, val, model_dest=model_dest)


def run_app():
    app.run(port=3905, debug=True, use_reloader=False, host='0.0.0.0')


# Start Flask app in a separate thread
flask_thread = Thread(target=run_app)
flask_thread.start()
