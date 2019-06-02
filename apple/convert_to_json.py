import pathlib

import pandas

data = pandas.read_csv(pathlib.Path('dataset/apple_reddit.csv'))

dataset = {'title': [], 'sentiment': []}

for i, row in data.iterrows():
    dataset['title'].extend([
        row['title1'],
        row['title2'],
        row['title3'],
        row['title4'],
        row['title5'],
    ])
    dataset['sentiment'].extend(
        [
            -1 if int(row['stockmovement']) == 0 else 1,
        ] * 5
    )

dataset = pandas.DataFrame(dataset)
dataset.to_json('dataset/apple_reddit.json', orient='records')
