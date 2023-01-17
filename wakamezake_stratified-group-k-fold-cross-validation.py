!pip install git+https://github.com/wakamezake/table_pipeline.git
from tb_pipe.validation.split import StratifiedGroupKFold
import random

import numpy as np

import pandas as pd

from collections import Counter, defaultdict
train_x = pd.read_csv('../input/train/train.csv')

train_y = train_x.AdoptionSpeed.values

groups = np.array(train_x.RescuerID.values)



def get_distribution(y_vals):

    y_distr = Counter(y_vals)

    y_vals_sum = sum(y_distr.values())

    return [f'{y_distr[i] / y_vals_sum:.2%}' for i in range(np.max(y_vals) + 1)]
distrs = [get_distribution(train_y)]

index = ['training set']

n_splits = 5

sgkf = StratifiedGroupKFold(n_splits=n_splits)

for fold_ind, (dev_ind, val_ind) in enumerate(sgkf.split(train_x, train_y, groups)):

    dev_y, val_y = train_y[dev_ind], train_y[val_ind]

    dev_groups, val_groups = groups[dev_ind], groups[val_ind]

    

    assert len(set(dev_groups) & set(val_groups)) == 0

    

    distrs.append(get_distribution(dev_y))

    index.append(f'development set - fold {fold_ind}')

    distrs.append(get_distribution(val_y))

    index.append(f'validation set - fold {fold_ind}')



display('Distribution per class:')

pd.DataFrame(distrs, index=index, columns=[f'Label {l}' for l in range(np.max(train_y) + 1)])