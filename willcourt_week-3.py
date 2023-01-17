# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_path='/kaggle/input/dwdm-week-3/Creditcardprom.csv'

data = pd.read_csv(data_path)

data.head()
data.drop([1,3]) #casewise deletion, deletes specific row

data.drop(['Magazine Promo'],axis=1) #listwise Deletion 
data.columns #prints names of columns in data set
#extracting only sex,age,income range,watch promo,life insurance

data2 = data[['Income Range','Sex','Age','Life Ins Promo','Watch Promo']]

data2
