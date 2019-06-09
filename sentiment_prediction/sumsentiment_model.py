import tensorflow as tf
import numpy as np
import pandas as pd
from tflearn.data_utils import to_categorical, pad_sequences

model=tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(1024,batch_input_shape=(1,1, 1),stateful=True,return_sequences=False))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.compile(optimizer=tf.train.AdamOptimizer(1e-2),loss='binary_crossentropy',metrics=['accuracy'])


dataframe = pd.read_csv('djia_sumsentiment.csv')

X = dataframe.to_numpy()
Y = X[:, 0]
X = np.delete(X, 0, 1)
X = X[:,0]

#split
index=int(X.shape[0]*0.8)
X_train=X[:index]/25
X_test=X[index:]/25
Y_train=Y[:index]
Y_test=Y[index:]

print(X_train)
print(Y_train)

for epoch in range(60):
    mean_tr_acc = []
    mean_tr_loss = []
    for i in range(len(X_train)):
        y_true = Y_train[i]

        tr_loss, tr_acc = model.train_on_batch(np.expand_dims(np.expand_dims(np.expand_dims(X_train[i], axis=1), axis=1),axis=1),
                                               np.expand_dims(np.expand_dims(y_true, axis=1), axis=1))
        mean_tr_acc.append(tr_acc)
        mean_tr_loss.append(tr_loss)

    print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
    print('loss training = {}'.format(np.mean(mean_tr_loss)))
    #print('___________________________________')

    mean_te_acc = []
    mean_te_loss = []
    for i in range(len(X_test)):
        y_true = Y_test[i]
        te_loss, te_acc = model.test_on_batch(np.expand_dims(np.expand_dims(np.expand_dims(X_test[i], axis=1), axis=1),axis=1),
                                              np.expand_dims(np.expand_dims(y_true, axis=1), axis=1))
        mean_te_acc.append(te_acc)
        mean_te_loss.append(te_loss)

    model.reset_states()

    print('accuracy testing = {}'.format(np.mean(mean_te_acc)))
    #print('loss testing = {}'.format(np.mean(mean_te_loss)))
    #print('___________________________________')

for i in range(len(X_train)):
    y_pred = model.predict_on_batch(np.expand_dims(np.expand_dims(np.expand_dims(X_train[i], axis=1), axis=1),axis=1))
    print("%s %s"%(y_pred,Y_train[i]))
for i in range(len(X_test)):
    y_pred = model.predict_on_batch(np.expand_dims(np.expand_dims(np.expand_dims(X_test[i], axis=1), axis=1),axis=1))
    print("%s %s"%(y_pred,Y_test[i]))


