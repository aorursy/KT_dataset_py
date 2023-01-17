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
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

import pandas_profiling as pp

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
data_df=pd.read_csv('/kaggle/input/nasa-airfoil-self-noise/NASA_airfoil_self_noise.csv')

data_df.head()


pp.ProfileReport(data_df)


scaler = StandardScaler()

X=data_df.drop('Sound',axis=1)

Y=data_df['Sound']



X_scaled=scaler.fit_transform(X)



X_df=pd.DataFrame(X_scaled,columns=[

'Frequency','AngleAttack','ChordLength','FreeStreamVelocity','SuctionSide'])


X_train, X_test, y_train, y_test = train_test_split(X_df, Y, test_size=0.33, random_state=42)


poly = PolynomialFeatures(4)

x_test_poly=poly.fit_transform(X_test)

x_train_poly=poly.fit_transform(X_train)

reg = LinearRegression()

reg.fit(x_train_poly,y_train)

y_pred=reg.predict(x_test_poly)



r2_score(y_test,y_pred)