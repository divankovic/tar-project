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
combined_data = defaultdict(lambda: dict(title=[], sentiment=0))

for newsline in news:
    date, title = newsline

    if title == "" or title == "\"":
        continue

    combined_data[date]["title"].append(title.strip())

for numeric_data in market_data:
    date, move, *_ = numeric_data
    combined_data[date]["sentiment"] = 1 if move >= 0 else 0

dataset = dict(title=[], sentiment=[])

for date in combined_data:
    titles = combined_data[date]["title"]
    sentiment = combined_data[date]["sentiment"]

    dataset["title"].extend(titles)
    dataset["sentiment"].extend([sentiment, ] * len(titles))

frame = pd.DataFrame(dataset)
frame.to_csv('kaggle_dataset.csv', index=False)
