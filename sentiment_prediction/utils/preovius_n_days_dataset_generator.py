import os
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

previous_n = 10

if not os.path.exists('kaggle_dataset_plus_sentiment.csv'):
    with open('../preprocessed_dataset/kaggle_predictions.csv', 'r') as io:
        predictions = [line.strip().split(',') for line in io.readlines()][1:]
        _, positive, negative, _ = zip(*predictions)

    kaggle_dataset = pd.read_csv('./kaggle_dataset.csv')
    kaggle_dataset['sentiment_positive'] = list(map(float, positive))
    kaggle_dataset['sentiment_negative'] = list(map(float, negative))
    kaggle_dataset.to_csv('kaggle_dataset_plus_sentiment.csv', index=False)
else:
    kaggle_dataset = pd.read_csv('kaggle_dataset_plus_sentiment.csv')

dataset = defaultdict(lambda: dict(stock_move=0, sentiment_positive=[]))

for _, row in tqdm(kaggle_dataset.iterrows()):
    dataset[row["date"]]["stock_move"] = row["stock_move"]
    dataset[row["date"]]["sentiment_positive"].append(float(row["sentiment_positive"]))

dates = list(set(kaggle_dataset['date'].tolist()))
day_dependencies = list(zip(*[dates[x::] for x in range(previous_n + 1)]))

deps = dict(stock_move=[], prevois_stock_moves=[])

for day_dep in day_dependencies:
    target_day = day_dep[0]
    previous_days = day_dep[1:]

    stock_move = dataset[target_day]["stock_move"]

    prevoius_stock_moves = []
    for prev_day in previous_days:
        prevoius_stock_moves.extend(dataset[prev_day]["sentiment_positive"])

    if len(prevoius_stock_moves) < (25 * previous_n):
        prevoius_stock_moves = prevoius_stock_moves + [0, ] * (25 * previous_n - len(prevoius_stock_moves))
    elif len(prevoius_stock_moves) > (25 * previous_n):
        prevoius_stock_moves = prevoius_stock_moves[:25 * previous_n]

    prevoius_stock_moves = ",".join(map(str, prevoius_stock_moves))

    deps["stock_move"].append(stock_move)
    deps["prevois_stock_moves"].append(prevoius_stock_moves)

dataset = pd.DataFrame(deps)
dataset.to_csv('kaggle_previous_{}.csv'.format(previous_n), index=False)
