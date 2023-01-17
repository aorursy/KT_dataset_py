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
temp=pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')



df1 = pd.read_csv('../input/overfitting-on-fire/Aggblender.csv')

df2 = pd.read_csv('../input/overfitting-on-fire/stack_mean.csv')

df3 = pd.read_csv('../input/overfitting-on-fire/submission.csv')







temp['isFraud'] =  0.4*df1['isFraud'] + 0.2 * df1['isFraud'] +  0.4 * df3['isFraud']

temp.to_csv('submission.csv', index=False )