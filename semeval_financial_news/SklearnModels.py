from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import tflearn
import numpy as np

from Dataset import Dataset
from Model import Model

dataset = Dataset('./data/headlines_train.json')

k_fold = KFold(n_splits=5)
k = 0

for train_index, test_index in k_fold.split(dataset.dataset['title']):
    x_train, x_test = np.array(dataset.dataset['title'])[train_index], np.array(dataset.dataset['title'])[test_index]
    y_train, y_test = np.array(dataset.dataset['sentiment'])[train_index], np.array(dataset.dataset['sentiment'])[
        test_index]

    model = Model(use_glove=True)
    model.train(x_train, y_train, x_test, y_test, epochs=50)

    print(accuracy_score(y_true=y_test, y_pred=model.classify(x_test)))
