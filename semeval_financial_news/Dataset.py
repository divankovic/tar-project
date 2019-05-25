import pandas as pd

from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


class Dataset():
    def __init__(self, dataset_path):
        self.dataset = Dataset.preprocess(pd.read_json(dataset_path))

    @staticmethod
    def preprocess(dataset):
        dataset_preproed = {'title': [], 'sentiment': []}

        lemmatizer = WordNetLemmatizer()

        for index, row in dataset.iterrows():
            tokens = word_tokenize(row['title'].lower())
            filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens
                               if (token not in stopwords.words('english') and token not in punctuation)]

            new_title = ''
            for token in filtered_tokens:
                new_title += token + ' '
            new_title.strip()

            dataset_preproed['title'].append(new_title)

            dataset_preproed['sentiment'].append(1 if row['sentiment'] >= 0 else 0)

        return dataset_preproed

    def train_test_split(self, test=0.2):
        X_train, X_test, y_train, y_test = train_test_split(self.dataset['title'], self.dataset['sentiment'],
                                                            test_size=0.2)
        return X_train, y_train, X_test, y_test
