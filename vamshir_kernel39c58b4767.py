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
import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()



df=pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')



df.describe()



sns.jointplot(x='sqft_living',y='price',data=df)



df.columns



df['date'] = pd.to_datetime(df['date'])



X=df[['bedrooms', 'bathrooms', 'sqft_living',

       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',

       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',

       'lat', 'long', 'sqft_living15', 'sqft_lot15']]



Y=df[['price']]



x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)



lm = LinearRegression()



lm.fit(x_train,y_train)



predict = lm.predict(x_test)



fig = plt.figure()

axes = fig.add_axes([0.1,0.1,0.8,0.8])

axes.scatter(predict,y_test)