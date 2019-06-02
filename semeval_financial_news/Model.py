import os
import pickle

import keras
import numpy as np
from Dataset import Dataset
from keras import Sequential
from keras.initializers import Constant
from keras.layers import Embedding, LSTM, Dense
from keras.models import load_model
from keras.optimizers import *
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))


class Model:

    def __init__(self, use_glove=False):
        self.use_glove = use_glove

        self.model = None
        self.max_len = 40
        self.embedding_size = 100
        self.vocabulary_size = 10000
        self.tokenizer = Tokenizer(num_words=self.vocabulary_size)

    def build_model(self, glove_embeddings=None):
        self.model = Sequential()

        self.model.add(Embedding(
            input_dim=self.vocabulary_size,
            output_dim=self.embedding_size,
            input_length=self.max_len,
            embeddings_initializer=Constant(glove_embeddings) if self.use_glove else glove_embeddings,
            trainable=glove_embeddings is None
        ))
        self.model.add(LSTM(64, return_sequences=True, dropout=0.5))
        self.model.add(LSTM(64, dropout=0.5, kernel_regularizer=l2(0.0001)))
        self.model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.0001)))
        self.model.add(Dense(1, activation='sigmoid'))

    def train(self, X_train, Y_train, X_test, Y_test, batch_size=32, epochs=10):
        self.tokenizer.fit_on_texts(X_train)

        # pickle word dictionary for later use
        with open('./checkpoints/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        encoded_train_titles = self.tokenizer.texts_to_sequences(X_train)
        encoded_test_titles = self.tokenizer.texts_to_sequences(X_test)

        X_train = encoded_train_titles
        X_test = encoded_test_titles

        X_train = pad_sequences(X_train, maxlen=self.max_len, value=0)
        X_test = pad_sequences(X_test, maxlen=self.max_len, value=0)

        word_index = self.tokenizer.word_index

        self.build_model(
            glove_embeddings=self.load_glove_embeddings(word_index) if self.use_glove else None
        )

        print('Y_train mean:', np.mean(Y_train))
        print('Y_test mean:', np.mean(Y_test))

        optimizer = Adam(0.01)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size, epochs=epochs)
        self.model.save('./checkpoints/model')

        scores = self.model.evaluate(X_test, Y_test, verbose=0)
        print('Test accuracy:', scores[1])

    def load_glove_embeddings(self, word_index):

        print('Loading embeddings.')
        embeddings_index = {}

        if not os.path.exists('glove.6B.100d.txt'):
            raise FileNotFoundError(
                'Download glove embeddings from http://nlp.stanford.edu/data/glove.6B.zip (822 MB file) and unzzip\n' +
                'Linux command:\n\n\t wget http://nlp.stanford.edu/data/glove.6B.zip; unzip glove.6B.zip'
            )

        f = open('glove.6B.100d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        num_words = min(len(word_index), self.vocabulary_size)
        embedding_matrix = np.zeros((num_words + 1, self.embedding_size))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        print('Embeddings loaded.')

        return embedding_matrix

    def load_model(self, path, dict_path):
        self.model = load_model(path)

        with open(dict_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def classify(self, input):
        X = self.tokenizer.texts_to_sequences(input)
        X = pad_sequences(X, maxlen=self.max_len, value=0)
        return self.model.predict(X)


if __name__ == '__main__':
    dataset = Dataset('../apple/dataset/apple_reddit.json')
    model = Model(use_glove=True)
    X_train, Y_train, X_test, Y_test = dataset.train_test_split()
    model.train(X_train, Y_train, X_test, Y_test, epochs=50)
