from regression_model import Model
from Dataset import Dataset

model = Model()
# model.load_model('./keras_model_checkpoint/82perc/best_model.hdf5', './keras_model_checkpoint/82perc/tokenizer.pickle')
model.load_model('./checkpoints/reg/model', './checkpoints/reg/tokenizer_reg.pickle')

while True:
    news = input('Input financial news headline : ')
    if news.lower() == 'exit':
        print('Bye')
        break
    news_preproed = Dataset.preprocess_title(news)
    result = model.classify([news_preproed])
    print(result)
