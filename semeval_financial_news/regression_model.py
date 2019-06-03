import keras.backend as K
from keras.optimizers import *
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Convolution1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant
from keras import metrics
from keras.models import load_model
from keras.regularizers import l2
from Dataset import Dataset

import pickle
import pathlib
import tensorflow as tf
import keras
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))


class Model():

    def __init__(self, use_glove=True):
        self.use_glove = use_glove
        self.model = None

        self.vocabulary_size = 10000
        self.embedding_size = 300
        self.max_len = 30
        self.tokenizer = Tokenizer(num_words=self.vocabulary_size)

    def build_model(self, embedding_initializer):
        self.model = Sequential()

        self.model.add(Embedding(
            input_dim=self.vocabulary_size,
            output_dim=self.embedding_size,
            input_length=self.max_len,
            embeddings_initializer=embedding_initializer,
            trainable=embedding_initializer is None
        ))
        self.model.add(LSTM(128, return_sequences=True, dropout=0.5))
        self.model.add(LSTM(128, dropout=0.5, kernel_regularizer=l2(0.001)))
        self.model.add(Dense(100, activation='relu', kernel_regularizer=l2(0.001)))
        self.model.add(Dense(1, activation='tanh'))

    @staticmethod
    def cos_distance_loss(y_true, y_pred):
        def l2_normalize(x, axis):
            norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
            return K.sign(x) * K.maximum(K.abs(x), K.epsilon()) / K.maximum(norm, K.epsilon())

        y_true = l2_normalize(y_true, axis=-1)
        y_pred = l2_normalize(y_pred, axis=-1)
        return 1 - K.mean(y_true * y_pred, axis=-1)

    def train(self, X_train, Y_train, X_test, Y_test, batch_size=16, epochs=100):

        self.tokenizer.fit_on_texts(X_train)

        # pickle word dictionary for later use
        with open('./checkpoints/reg/tokenizer_reg.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        encoded_train_titles = self.tokenizer.texts_to_sequences(X_train)
        encoded_test_titles = self.tokenizer.texts_to_sequences(X_test)

        X_train = encoded_train_titles
        X_test = encoded_test_titles

        # print('Train mean', np.mean(np.abs(Y_train)))
        # print('Test mean', np.mean(np.abs(Y_test)))

        X_train = pad_sequences(X_train, maxlen=self.max_len, value=0)
        X_test = pad_sequences(X_test, maxlen=self.max_len, value=0)

        word_index = self.tokenizer.word_index
        self.build_model(embedding_initializer=self.load_glove_embeddings(word_index) if self.use_glove else None)

        optimizer = Adam()

        self.model.compile(loss='mse', optimizer=optimizer, metrics=['mae', metrics.cosine])

        self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size, epochs=epochs)
        self.model.save('./checkpoints/reg/model')

        scores = self.model.evaluate(X_test, Y_test, verbose=0)
        print('Test MSE:', scores[1])

    def load_model(self, path, dict_path):
        self.model = load_model(path)

        with open(dict_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def load_glove_embeddings(self, word_index):

        print('Loading embeddings.')
        embeddings_index = {}

        glove_path = './data/glove.6B/glove.6B.'+str(self.embedding_size)+'d.txt'
        if not pathlib.Path(glove_path).exists():
            raise FileNotFoundError(
                'Download glove embeddings from http://nlp.stanford.edu/data/glove.6B.zip (822 MB file) and unzzip\n' +
                'Linux command:\n\n\t wget http://nlp.stanford.edu/data/glove.6B.zip; unzip glove.6B.zip'
            )

        f = open(glove_path, encoding='utf-8')
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

        return Constant(embedding_matrix)

    def classify(self, input):
        X = self.tokenizer.texts_to_sequences(input)
        X = pad_sequences(X, maxlen=self.max_len, value=0)
        return self.model.predict(X)


if __name__ == '__main__':
    dataset = Dataset('./data/headlines_train.json', regression=True)
    model = Model()
    X_train, Y_train, X_test, Y_test = dataset.train_test_split()
    model.train(X_train, Y_train, X_test, Y_test)
