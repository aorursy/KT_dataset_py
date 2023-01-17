import numpy as np

import pandas as pd 



import tqdm

from pathlib import Path

from collections import defaultdict, Counter
DATA_ROOT = Path('../input/2019-3rd-ml-month-with-kakr/')

train_csv = pd.read_csv('../input/2019-3rd-ml-month-with-kakr/train.csv')
def make_folds(n_folds: int) -> pd.DataFrame:

    df = pd.read_csv(DATA_ROOT / 'train.csv', engine='python')



    cls_counts = Counter([classes for classes in df['class']])

    fold_cls_counts = defaultdict()

    for class_index in cls_counts.keys():

        fold_cls_counts[class_index] = np.zeros(n_folds, dtype=np.int)



    df['fold'] = -1

    pbar = tqdm.tqdm(total=len(df))



    def get_fold(row):

        class_index = row['class']

        counts = fold_cls_counts[class_index]

        fold = np.argmin(counts)

        counts[fold] += 1

        fold_cls_counts[class_index] = counts

        row['fold']=fold

        pbar.update()

        return row

    

    df = df.apply(get_fold, axis=1)

    return df
df = make_folds(n_folds=5)

df.to_csv('folds.csv', index=None)