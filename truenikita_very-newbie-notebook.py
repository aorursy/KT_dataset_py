# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from kaggle.competitions import nflrush

env = nflrush.make_env()



train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
np.mean(train_df['Yards'])
def make_my_predictions(test_df, sample_prediction_df):

    pred_yards=0

    sample_predictions_df=[]

    for string in test_df:

        pred_yards=pred_yards+np.mean(train_df['Yards'])

    for i in range (199):

        if (i-100)<pred_yards:

            sample_predictions_df[i]=0

        else:

            sample_predictions_df[i]=1

    return sample_prediction_df
for (test_df, sample_prediction_df) in env.iter_test():

  predictions_df = make_my_predictions(test_df, sample_prediction_df)

  env.predict(predictions_df)

  

env.write_submission_file()