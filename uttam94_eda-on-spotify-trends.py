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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 
df = pd.read_csv('../input/top50spotify2019/top50.csv',encoding='latin-1')
df.head()
from wordcloud import WordCloud # wordcloud

fig, ax = plt.subplots(1, 2, figsize=(16,16))# plotting two subplots 

wordcloud = WordCloud(background_color='white',width=800, height=800).generate(' '.join(df['Genre']))

wordcloud_sub = WordCloud(background_color='white',width=800, height=800).generate(' '.join(df['Artist.Name'].dropna().astype(str)) ) # used astype(str)bbecause data is not present as str

ax[0].imshow(wordcloud)

ax[0].axis('off')

ax[0].set_title('Wordcloud for Genre')

ax[1].imshow(wordcloud_sub)

ax[1].axis('off')

ax[1].set_title('Wordcloud for Artist Name')

plt.show()
df.groupby('Genre').size().plot.bar()
df.groupby('Artist.Name').size().plot.bar()
#here target value is polpularity 

x= df.iloc[:,4:13].values

y = df.iloc[:,[13]].values 
y.shape
#creating a training and test set 

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.30)
#scaling 

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train= sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)



sc_y = StandardScaler()

y_train = sc_y.fit_transform(y_train)