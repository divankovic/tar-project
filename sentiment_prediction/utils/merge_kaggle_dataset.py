import re
from collections import defaultdict

import pandas as pd


def load_data(path, preprocessor_fn):
    with open(path) as io:
        data = list(
            map(
                preprocessor_fn,
                map(
                    str.strip,
                    io.readlines()[1:]
                )
            )
        )
    return data


news = load_data(
    '/Users/ivanilic/Data/fax/8_semestar/apt/project/dataset/stocknews/RedditNews.csv',
    lambda line: (
        line.split(',')[0],
        "".join(line.split(',')[1:])
    )
)
market_data = load_data(
    '/Users/ivanilic/Data/fax/8_semestar/apt/project/dataset/stocknews/DJIA_table.csv',
    lambda line: (
        line.split(',')[0],
        float(line.split(',')[4]) - float(line.split(',')[1]),
        float(line.split(',')[2]) - float(line.split(',')[3]),
        float(line.split(',')[-2]),
    )
)
combined_data = defaultdict(lambda: dict(title=[], stock_move=0))

for newsline in news:
    date, title = newsline

    if title == "" or title == "\"":
        continue

    combined_data[date]["title"].append(title.strip())

for numeric_data in market_data:
    date, move, *_ = numeric_data
    combined_data[date]["stock_move"] = 1 if move >= 0 else 0

dataset = dict(date=[], title=[], stock_move=[])

for date in combined_data:
    titles = combined_data[date]["title"]
    stock_move = combined_data[date]["stock_move"]

    dataset["title"].extend(titles)
    dataset["date"].extend([date, ] * len(titles))
    dataset["stock_move"].extend([stock_move, ] * len(titles))

frame = pd.DataFrame(dataset)
frame.to_csv('kaggle_dataset.csv', index=False)
