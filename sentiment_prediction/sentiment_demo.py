import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentiment_prediction.Sentiment import Sentiment
from string import punctuation


def preprocess(input, word_dict):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(input)
    tokens_filtered = [lemmatizer.lemmatize(token) for token in tokens
                       if (token not in stopwords.words('english') and token not in punctuation)]

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

while True:
    news = input('Input news headline : ')
    if news.lower() == 'exit':
        print('Bye')
        break
    news_preproed = preprocess(news, word_dict)
    result = model.classify([news_preproed])
    print('Positive : %f  Negative : %f' % (result[0][0], result[0][1]))
