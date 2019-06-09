import numpy as np
import pandas
from sklearn.model_selection import KFold

apple = './sentiment_prediction/utils/news_dataset.csv'
# apple = './apple/dataset/apple_preprocessed.csv'
apple_ds = pandas.read_csv(apple)

labels = np.array(apple_ds['sentiment'].tolist())
# labels = np.array(apple_ds['label'].tolist())

np.random.shuffle(labels)
kfold = KFold(n_splits=10)

for x, y in kfold.split(np.arange(labels.shape[0])):
    rnd = np.random.randint(0, 2, size=labels[y].shape[0])
    rnd_acc = np.sum((rnd == labels[y]) * 1) / rnd.shape[0]

    most_common = 1 if np.sum(labels[y]) > (labels[y].shape[0] / 2) else 0
    most_common = np.sum((labels[y] == most_common) * 1) / labels[y].shape[0]

    print(
        "{:.3f}".format(rnd_acc),
        "{:.3f}".format(most_common),
    )
