from regression_model import Model
from Dataset import Dataset

model = Model()
model.load_model('./checkpoints/model', './checkpoints/tokenizer.pickle')
while True:
    news = input('Input financial news headline : ')
    if news.lower() == 'exit':
        print('Bye')
        break
    news_preproed = Dataset.preprocess_title(news)
    result = model.classify([news_preproed])
    print(result)
