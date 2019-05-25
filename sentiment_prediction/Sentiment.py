import json

import tflearn
from SentimentDataset import SentimentDataset
from tflearn.data_utils import to_categorical, pad_sequences


class Sentiment():
    def __init__(self, dictionary_size):
        rnn = tflearn.input_data([None, 50])
        rnn = tflearn.embedding(rnn, input_dim=dictionary_size, output_dim=128)

        rnn = tflearn.lstm(rnn, 512, dropout=0.8, return_seq=True)
        rnn = tflearn.lstm(rnn, 512, dropout=0.8)
        rnn = tflearn.fully_connected(rnn, 2, activation='softmax')
        rnn = tflearn.regression(rnn, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

        self.model = tflearn.DNN(rnn, tensorboard_verbose=0)

    def train(self, X_train, Y_train, X_test, Y_test, batch_size=32, epochs=10):
        X_train = pad_sequences(X_train, maxlen=50, value=0.)
        X_test = pad_sequences(X_test, maxlen=50, value=0.)

        self.model.fit(X_train, Y_train, validation_set=(X_test, Y_test), show_metric=True, batch_size=batch_size,
                       n_epoch=epochs)
        self.model.save('./mode_checkpoints/sentiment.model')

    def classify(self, input):
        X = pad_sequences(input, maxlen=50, value=0.)

        return self.model.predict(X)

    def load_model(self, path):
        self.model.load(path)


if __name__ == '__main__':
    # train,test,_=imdb.load_data(path='imdb.pkl',n_words=10000,valid_portion=0.1)

    # X_train,Y_train=train
    # X_test,Y_test=test

    with open('./preprocessed_dataset/word_dict.json', 'r') as fp:
        word_dict = json.load(fp)

    sd = SentimentDataset('preprocessed_dataset/kita.csv')
    X_train, Y_train, X_test, Y_test = sd.get_dataset(0.2)

    Y_train = to_categorical(Y_train, nb_classes=2)
    Y_test = to_categorical(Y_test, nb_classes=2)

    sentiment = Sentiment(len(word_dict))
    sentiment.train(X_train, Y_train, X_test, Y_test)
