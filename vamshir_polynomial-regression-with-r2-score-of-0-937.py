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
import seaborn as sns
import matplotlib.pyplot as plt
fish_data = pd.read_csv('/kaggle/input/fish-market/Fish.csv')
fish_data.info()
plt.figure(figsize=(14,5))
sns.countplot(x='Species',data=fish_data1)
fish_data['Species'].unique()
species_data = pd.get_dummies(fish_data['Species'])
fish_data = pd.concat([fish_data,species_data],axis=1,sort=False)
fish_data.drop('Species',axis=1,inplace=True)
X=fish_data[['Length1', 'Length2', 'Length3', 'Height', 'Width', 'Bream',
       'Parkki', 'Perch', 'Pike', 'Roach', 'Smelt', 'Whitefish']]
Y=fish_data[['Weight']]
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
lm = LinearRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
poly = PolynomialFeatures(degree = 1) 
X_poly = poly.fit_transform(X_train) 
  
poly.fit(X_poly, Y_train) 
lm = LinearRegression() 
lm.fit(X_poly, Y_train) 

pr2 = lm.predict(poly.fit_transform(X_test))



r2val2 = r2_score(Y_test,pr2)
r2val2
#gives r2 score fo 0.937