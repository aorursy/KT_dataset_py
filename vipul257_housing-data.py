# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
df = pd.read_csv('../input/the-housing-data/Housing_data.csv',sep=';')
X=df.drop(['lotsize'],axis=1)
Y=df['lotsize']
from sklearn.preprocessing import StandardScaler
scl = StandardScaler()
scl.fit(X)
Xt = pd.DataFrame(scl.transform(X))
Xt
#from sklearn.decomposition import PCA
#pca = PCA(n_components=2)
#pca.fit(Xt)
#X_pca = pca.transform(Xt)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(Xt,Y)
from sklearn.linear_model import LinearRegression
lmodel = LinearRegression()
lmodel.fit(xtrain,ytrain)
ytrainp = lmodel.predict(xtrain)
ytestp = lmodel.predict(xtest)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(ytrain,ytrainp))
print(mean_absolute_error(ytest,ytestp))
from sklearn.preprocessing import PolynomialFeatures

tr_err = []
ts_err = []
for i in range(1,12):
    pol = PolynomialFeatures(degree = i)
    
    xtrain_pol = pol.fit_transform(xtrain)
    xtest_pol = pol.fit_transform(xtest)
    
    lmodelp = LinearRegression()
    
    lmodelp.fit(xtrain_pol,ytrain)
    
    ytrainp = lmodelp.predict(xtrain_pol)
    ytestp = lmodelp.predict(xtest_pol)
    
    tr_err.append(mean_absolute_error(ytrainp,ytrain))
    ts_err.append(mean_absolute_error(ytestp,ytest))
tr_err
ts_err
plt.plot(range(1,12),tr_err)
plt.plot(range(1,12),ts_err,color='red')
plt.show()
#Model Can be best defined for prediction is on degree 2
#after degree 2 model is getting overfitted with decrement in training error
#and continous increment in testing error
#On Doing Dimensionality reduction or PCA best possible outcome will be on degree 5