import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import numpy as np

class Sentiment():
    def __init__(self):
        rnn = tflearn.input_data([None, 100])
        rnn = tflearn.embedding(rnn, input_dim=10000, output_dim=128)

        rnn = tflearn.lstm(rnn, 128, dropout=0.8)
        rnn = tflearn.fully_connected(rnn, 2, activation='softmax')
        rnn = tflearn.regression(rnn, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

        self.model = tflearn.DNN(rnn, tensorboard_verbose=0)

    def train(self,X_train,Y_train,X_test,Y_test,batch_size=32,epochs=10):
        X_train = pad_sequences(X_train, maxlen=100, value=0.)
        X_test = pad_sequences(X_test, maxlen=100, value=0.)

        self.model.fit(X_train, Y_train, validation_set=(X_test, Y_test), show_metric=True, batch_size=batch_size,n_epoch=epochs)
        self.model.save('sentiment.model')

    def classify(self,input):
        X=pad_sequences(input, maxlen=100, value=0.)

        return self.model.predict(X)

    def load_model(self,path):
        self.model.load(path)


if __name__=='__main__':
    train,test,_=imdb.load_data(path='imdb.pkl',n_words=10000,valid_portion=0.1)

    X_train,Y_train=train
    X_test,Y_test=test

    Y_train = to_categorical(Y_train, nb_classes=2)
    Y_test = to_categorical(Y_test, nb_classes=2)

    sentiment=Sentiment()
    sentiment.train(X_train,Y_train,X_test,Y_test)





