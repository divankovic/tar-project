import json
from string import punctuation

import pandas as pd
from Sentiment import Sentiment
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def preprocess(input, word_dict):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(input)
    tokens_filtered = [
        lemmatizer.lemmatize(token) for token in tokens
        if (token not in stopwords.words('english') and token not in punctuation)
    ]

    result = []
    for token in tokens_filtered:
        if token not in word_dict:
            code = 0
        else:
            code = word_dict[token]

        result.append(code)

    return result


with open('./preprocessed_dataset/word_dict.json', 'r') as fp:
    word_dict = json.load(fp)

model = Sentiment(len(word_dict))
model.load_model('./model_checkpoints/sentiment.model')

kaggle_dataset = pd.read_csv('./utils/kaggle_dataset.csv')
kaggle_news = kaggle_dataset['title'].tolist()
kaggle_sentiment = kaggle_dataset['sentiment'].tolist()

with open('./preprocessed_dataset/kaggle_predictions.csv', 'w') as fp:
    fp.write('index,positive,negative,sentiment_gt\n')

    for i, news in enumerate(kaggle_news):
        news_preproed = preprocess(news, word_dict)
        result = model.classify([news_preproed])

        positive = result[0][0]
        negative = result[0][1]
        sentiment_gt = int(kaggle_sentiment[i])

        fp.write('{},{},{},{}\n'.format(i, positive, negative, sentiment_gt))

        pred = int(round(positive))

        print('Index: {} Positive : {:.2f}  Negative : {:.2f} Truth: {}'.format(
            i,
            positive,
            negative,
            sentiment_gt,
        ))
