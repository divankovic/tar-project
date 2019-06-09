from collections import Counter

import re
import nltk
import pandas as pd
from statistics import mean
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tree import Tree
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


class Dataset():
    def __init__(self, dataset_path, regression=False, ner=False):
        self.dataset = Dataset.preprocess(pd.read_json(dataset_path), regression, ner=ner)

    @staticmethod
    def preprocess(dataset, regression=False, ner=False):
        dataset_preproed = {'title': [], 'sentiment': []}

        # companies = set(dataset['company'])
        # avg_title_len = mean([len(word_tokenize(title)) for title in dataset['title']])
        # print('Total companies : ' + str(len(companies)))
        # print('Average title length : ' + str(avg_title_len))
        # cnt = Counter()

        for index, row in dataset.iterrows():
            new_title = Dataset.preprocess_title(row['title'], row['company'], ner=ner)
            # new_title = Dataset.preprocess_title(row['title'].replace(row['company'], '<company>'))
            sentiment = row['sentiment'] if regression else (1 if row['sentiment'] >= 0 else 0)
            # cnt[sentiment] += 1

            if len(word_tokenize(new_title)) < 2:
                continue

            # print(new_title)
            dataset_preproed['title'].append(new_title)
            dataset_preproed['sentiment'].append(sentiment)

        # print(cnt)

        return dataset_preproed

    @staticmethod
    def preprocess_title(title, company=None, ner=False):
        lemmatizer = WordNetLemmatizer()

        # tokens = word_tokenize(title.lower())
        tokens = word_tokenize(title)
        if ner:
            tokens = Dataset.ne_removal(tokens)

        filtered_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens
                           if (token not in stopwords.words('english') and token not in punctuation)]

        new_title = ''
        for token in filtered_tokens:
            new_title += token + ' '

        if company is not None:
            new_title = new_title.replace(company.lower(), 'company')

        new_title.strip()

        return new_title

    @staticmethod
    def ne_removal(tokens):
        ne_chunked = nltk.ne_chunk(nltk.pos_tag(tokens), binary=False)
        new_tokens = []
        for leaf in ne_chunked:
            if type(leaf) != nltk.Tree:
                new_tokens.append(leaf[0])
            # else:
            # new_tokens.append(leaf._label)

        regnumber = re.compile(r'\d+(?:,\d*)?')
        new_tokens = ['' if (regnumber.match(token) or token=='\'s') else token for token in new_tokens]
        return new_tokens

    def train_test_split(self, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(self.dataset['title'], self.dataset['sentiment'],
                                                            test_size=test_size)
        return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    title = 'BP signs $12 billion energy deal in South East Asia'
    print(Dataset.preprocess_title(title, 'BP'))
