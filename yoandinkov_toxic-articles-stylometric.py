import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json
with open('/kaggle/input/mediascan-bg-articles/articles.json') as f:

    articles = json.load(f)
len(articles)
non_toxic_articles = [x for x in articles if x['label'] == 'нетоксичен']

toxic_articles = [x for x in articles if x['label'] != 'нетоксичен']
print(f'Toxic: {len(toxic_articles)}')

print(f'Non toxic: {len(non_toxic_articles)}')
def get_avg(articles, prop, separator=None):

    if separator:

        return np.average([len(x[prop].split(separator)) for x in articles])



    return np.average([len(x[prop]) for x in articles])



def print_stylometric(articles, group_name):

    print(f"For {group_name}")

    print(f"Avg title chars {get_avg(articles, 'title')}")

    print(f"Avg title words {get_avg(articles, 'title', ' ')}")

    print(f"Avg text chars {get_avg(articles, 'text')}")

    print(f"Avg text words {get_avg(articles, 'text', ' ')}")

    print(f"Avg text sentences {get_avg(articles, 'text', '.')}")
print_stylometric(articles, 'all')

print_stylometric(toxic_articles, 'toxic')

print_stylometric(non_toxic_articles, 'non-toxic')