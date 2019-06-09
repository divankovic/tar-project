from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15)
    model.train(x_train, y_train, x_val, y_val, epochs=50)
    model.load_model(path=str(model.log_dir)+'/best_model.hdf5', dict_path='./checkpoints/tokenizer.pickle')

    print(accuracy_score(y_true=y_test, y_pred=model.predict_classes(x_test)))
