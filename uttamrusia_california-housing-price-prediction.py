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
demo = pd.read_csv("/kaggle/input/california-housing-prices/housing.csv")
demo.head()
demo.info()
from collections import Counter
Counter(demo.ocean_proximity)
demo.ocean_proximity.value_counts()
#Encode Categorical data
#Ocean Proximity is not an Nominal data
#Ordinal Data can be handled just like numerical columns
data = demo
dummy = pd.get_dummies(data["ocean_proximity"],prefix = "Ocean_" ,).iloc[:,:-1]
data=pd.concat([data,dummy],axis=1)
data=data.drop('ocean_proximity', axis =1)
data.shape
data.head()
#total_bedrooms are having some missing value
#167 values are missing.
#Missing Value Treatment
#Now we will try to impute these value by a ML Model
from sklearn.experimental import enable_iterative_imputer
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
it = IterativeImputer(estimator=LinearRegression())
newdata= pd.DataFrame(it.fit_transform(data))
newdata.columns = data.columns
newdata.head()
newdata.info()
data.describe()
newdata.describe()
#Now in this newdata dataframe , total_bedrooms are having negative value as min which is not good enough
#Imputation method dsnt work here. So we will try iterative technique

from sklearn.experimental import enable_iterative_imputer
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
it = IterativeImputer(estimator=RandomForestRegressor())
newdata= pd.DataFrame(it.fit_transform(data))
newdata.columns = data.columns
newdata.head()
newdata.describe()
data.describe()
from matplotlib import pyplot as plt
import seaborn as sns
sns.distplot(data['total_bedrooms'])
sns.distplot(newdata['total_bedrooms'])
# Now we can see that data is equally distributed in both dataframe
# Summary is good, graph is good
#Hypothesis testing .. To Clearly be sure about changes
#Test of means
#Test of variance
import scipy.stats as stats
stats.ttest_ind(data['total_bedrooms'],newdata['total_bedrooms'], nan_policy = 'omit')

#99 % pvalue telling that we are failed to reject null 
# Here null meaning is means(before imputation ) = means(after imputation)
#Normalise the data
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
powerdata = pd.DataFrame(pt.fit_transform(newdata))
powerdata.columns = newdata.columns
powerdata.head()
newdata.hist(figsize=(10,10), bins = 50)
plt.show
powerdata.hist(figsize=(10,10), bins = 50)
plt.show
newdata.skew()
powerdata.skew()
#Skewness of data is reduced drastically

#Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaleddata = pd.DataFrame(sc.fit_transform(powerdata))
scaleddata.columns = powerdata.columns
scaleddata.head()
scaleddata.hist(figsize=(10,10), bins = 50)
plt.show
#Scaling dsnt change the data pattern
#Incorrect approach
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X=scaleddata.drop('median_house_value', axis =1)
y=scaleddata['median_house_value']
Xtrain , Xtest , ytrain , ytest = train_test_split(X,y,test_size = .20 , random_state = 10)
lr = LinearRegression()
lr.fit(Xtrain, ytrain)
data = pd.read_csv("/kaggle/input/california-housing-prices/housing.csv")
dummy = pd.get_dummies(data["ocean_proximity"], prefix='Ocean_').iloc[:,:-1]
data = pd.concat([data,dummy], axis=1)
data = data.drop("ocean_proximity", axis=1)
data.shape
#Combined all steps in pipeline
from sklearn.pipeline import Pipeline
X=data.drop('median_house_value',axis=1)
y=data['median_house_value']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size= .20,random_state=10)
pipe = Pipeline((
("it", IterativeImputer(estimator = LinearRegression())),
("pt", PowerTransformer()),
("sc",StandardScaler()),
("lr", LinearRegression()),
))
pipe.fit(Xtrain,ytrain)
print("Training R2")
print(pipe.score(Xtrain,ytrain))
print("Testing R2")
print(pipe.score(Xtest,ytest))
#Combined all steps in pipeline
#Add Polynomial feature
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
X=data.drop('median_house_value',axis=1)
y=data['median_house_value']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size= .20,random_state=10)
pipe = Pipeline((
("it", IterativeImputer(estimator = LinearRegression())),
("pt", PowerTransformer()),
("sc",StandardScaler()),
("poly",PolynomialFeatures(degree=3)),
("lr", LinearRegression()),
))
pipe.fit(Xtrain,ytrain)
print("Training R2")
print(pipe.score(Xtrain,ytrain))
print("Testing R2")
print(pipe.score(Xtest,ytest))
#Try XGBoost
from xgboost import XGBRegressor
X=data.drop('median_house_value',axis=1)
y=data['median_house_value']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size= .20,random_state=10)
pipe = Pipeline((
("it", IterativeImputer(estimator = LinearRegression())),
("pt", PowerTransformer()),
("sc",StandardScaler()),
("lr", XGBRegressor(n_estimators = 100)),
))
pipe.fit(Xtrain,ytrain)
print("Training R2")
print(pipe.score(Xtrain,ytrain))
print("Testing R2")
print(pipe.score(Xtest,ytest))
from sklearn.model_selection import cross_val_score
scorexgb = cross_val_score(pipe, Xtrain, ytrain, cv=5 , scoring='r2')
print(scorexgb)
import numpy as np
print("Average R2 of the model")
print(np.mean(scorexgb))
print('SD of the model')
print(np.std(scorexgb))
#Estimate confidence interval of R2
#Estimate Confidence Interval of R2
import scipy.stats as stats
n=10
xbar = np.mean(scorexgb)
s = np.std(scorexgb)
se = s/np.sqrt(n)
stats.t.interval(0.95,df=n-1,loc=xbar,scale=se)