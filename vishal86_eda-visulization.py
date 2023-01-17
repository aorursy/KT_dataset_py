# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

% matplotlib inline

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv("../input/Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv")

test=pd.read_csv('../input/Uniqlo(FastRetailing) 2017 Test - stocks2017.csv')

train.shape
train.head()
train.set_index('Date',inplace=True)
train.describe()
# volume statistics

vol = train[['Volume']]

print ("Min: %s Max: %s Average: %s" % (vol.min().values[0], vol.max().values[0], vol.mean().values[0])) 

# plot the historical closing prices and volume using matplotlib

plots = train[['Close', 'Volume']].plot(subplots=True, figsize=(10, 10))

plt.show()
# chart a basic 50 period moving average of the closing price

import pandas as pd

train['ma50'] = pd.rolling_mean(train['Close'], 50)

train['ma200'] = pd.rolling_mean(train['Close'], 200)

plots = train[['Close', 'ma50', 'ma200']].plot(subplots=False, figsize=(10, 4))

plt.show()