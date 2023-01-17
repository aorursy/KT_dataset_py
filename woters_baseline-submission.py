# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/Train"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/sample_submission.csv', index_col=["test_id"])

print(df.shape)

df.tail()
train_y = pd.read_csv("../input/Train/train.csv")

train_y.tail()
df = df.astype(int)

means = train_y.mean()

means = means.round().astype(int)

for row in tqdm(range(len(df))):

    # means["test_id"] = int(row)

    df.iloc[row] = means

df.tail()
df.to_csv("baseline_submission.csv")