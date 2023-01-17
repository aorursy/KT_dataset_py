# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm_notebook



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#modify this to try different values

upper_range = 3



df = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")
#generating data

for i in tqdm_notebook(range(1, 29)):

    df[f'F{i}'] = [np.random.randint(upper_range) for _ in range(len(df))]
df.to_csv("random_submission.csv", index=False)
df.head()