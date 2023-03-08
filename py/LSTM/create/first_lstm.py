from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.callbacks import ModelCheckpoint
from keras.utils import pad_sequences
import keras
import os
import ebooklib
from ebooklib import epub
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import tensorflow as tf


def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def get_text_data(input_folder):
    file_extensions = ['.epub', '.pdf',  '.prc']

    # Function to recursively find all files with specified extensions in a folder
    def find_files(folder, extensions):
        for root, dirs, files in os.walk(folder):
            for file in files:
                if os.path.splitext(file)[-1] in extensions:
                    yield os.path.join(root, file)

    # Load the text from all files in the input folder with specified extensions
    book_text = ''
    for path in find_files(input_folder, file_extensions):
        print(path)
        book = None
        if path.endswith('.epub'):
            book = epub.read_epub(path)
        elif path.endswith('.pdf'):
            pdf_reader = PdfReader(path)
            for page in pdf_reader.pages:
                book_text += page.extract_text()
        if book is not None:
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                book_text += remove_html_tags(
                    item.get_body_content().decode("utf-8"))

    # print(book_text)
    return book_text


device = '/cpu:0'
with tf.device(device):

    book_text = get_text_data('../data_books')

    tokenizer = Tokenizer(
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
    tokenizer.fit_on_texts([book_text])
    sequences = tokenizer.texts_to_sequences([book_text])[0]

    seq_length = 100
    x_data = []
    y_data = []
    for i in range(seq_length, len(sequences)):
        x_data.append(sequences[i-seq_length:i])
        y_data.append(sequences[i])

    max_sequence_len = max([len(seq) for seq in x_data])
    x_data_padded = pad_sequences(
        x_data, maxlen=max_sequence_len, padding='pre')

    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index)+1,
              output_dim=512, input_length=max_sequence_len))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(tokenizer.word_index)+1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Set the number of epochs and batch size
    num_epochs = 50
    batch_size = 128

    # Define the checkpoint to save the best model weights
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # Train the model for the specified number of epochs
    for i in range(num_epochs):
        model.fit(x_data_padded, keras.utils.to_categorical(y_data, num_classes=len(tokenizer.word_index)+1),
                  epochs=1, batch_size=batch_size, callbacks=callbacks_list)
        model.save_weights(f'epoch_{i}.h5')
