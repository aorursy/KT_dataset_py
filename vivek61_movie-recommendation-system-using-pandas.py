# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
movie=pd.read_csv('/kaggle/input/movies.csv')

ratings=pd.read_csv('/kaggle/input/ratings.csv')
movie.head(3)

#ratings.head(3)
df=pd.merge(movie,ratings,on='movieId')

df.head()
df.describe()
avgrate=pd.DataFrame(df.groupby('title')['rating'].mean())

avgrate.head(3)
avgrate['no_of_ratings']=pd.DataFrame(df.groupby('title')['rating'].count())

avgrate.head(4)
plt.hist(x=avgrate['rating'],bins=50)

plt.show()

plt.hist(x=avgrate['no_of_ratings'],bins=50)

plt.show()
sns.jointplot(x='rating',y='no_of_ratings',data=avgrate)
moviematrix=df.pivot_table(index='userId',columns='title',values='rating')

AFO_user_rating = moviematrix['Air Force One (1997)']

contact_user_rating = moviematrix['Contact (1997)']

AFO_user_rating.head(10)
corofaof=moviematrix.corrwith(AFO_user_rating)

corofaof.head(50)
corr_contact = pd.DataFrame(rofaof, columns=['Correlation'])

corr_contact.dropna(inplace=True)

corr_contact.head()
corr_aof = pd.DataFrame(corofaof, columns=['Correlation'])

corr_contact.dropna(inplace=True)

corr_contact=corr_contact.join(avgrate['no_of_ratings'])

corr_aof = corr_aof.join(avgrate['no_of_ratings'])

corr_contact.head()

corr_aof[corr_aof['no_of_ratings'] > 100].sort_values(by='Correlation', ascending=False).head(10)

corr_contact[corr_contact['no_of_ratings'] > 100].sort_values(by='Correlation', ascending=False).head(10)