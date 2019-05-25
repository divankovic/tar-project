import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json

NUM_OF_KNOWN_WORDS=10000
stop_words=set(stopwords.words('english'))

new_dataset={'title':[],'sentiment':[]}
original=pd.read_csv('./news_dataset.csv')

word_count={}
word_dictionary={'UNKNOWN_TAG':0}

lemmatizer=WordNetLemmatizer()

for index,row in original.iterrows():
    tokens=word_tokenize(row['title'].lower())
    tokens_without_stopwords=[lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    new_sentence=''
    for token in tokens_without_stopwords:
        if token not in word_count:
            word_count[token]=1
        else:
            word_count[token]+=1

most_frequent_words=sorted(word_count, key=word_count.__getitem__,reverse=True)
for i in range(NUM_OF_KNOWN_WORDS-1):
    word_dictionary[most_frequent_words[i]]=i+1

for index,row in original.iterrows():
    tokens=word_tokenize(row['title'].lower())
    tokens_without_stopwords=[lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    new_sentence=''
    for token in tokens_without_stopwords:
        if token not in word_dictionary:
            code=0
        else:
            code=word_dictionary[token]
        if new_sentence=='':
            new_sentence+=str(code)
        else:
            new_sentence+=' '+str(code)
    new_dataset['title'].append(new_sentence)
    new_dataset['sentiment'].append(row['sentiment'])

topics_data = pd.DataFrame(new_dataset)
topics_data.to_csv('../preprocessed_dataset/kita.csv', index=False)

with open('../preprocessed_dataset/word_dict.json', 'w') as fp:
    json.dump(word_dictionary, fp)
