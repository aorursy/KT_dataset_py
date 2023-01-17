import numpy as np

import pandas as pd

import os
sub_path = "../input/top-my-submission"

all_files = os.listdir(sub_path)
outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files if '.csv' in f and '_me' in f]

concat_sub = pd.concat(outs, axis=1)

cols = list(map(lambda x: "target" + str(x), range(len(concat_sub.columns))))

concat_sub.columns = cols

concat_sub.reset_index(inplace=True)

concat_sub.head()

ncol = concat_sub.shape[1]
# get the data fields ready for stacking

concat_sub['target_mean'] = concat_sub.iloc[:, 1:ncol].mean(axis=1)

concat_sub['target_median'] = concat_sub.iloc[:, 1:ncol].median(axis=1)
concat_sub.describe()
concat_sub['target'] = concat_sub['target_mean']

concat_sub[['image_name', 'target']].to_csv('submission_mean.csv', 

                                        index=False, float_format='%.6f')
concat_sub['target'] = concat_sub['target_median']

concat_sub[['image_name', 'target']].to_csv('submission_median.csv', 

                                        index=False, float_format='%.6f')