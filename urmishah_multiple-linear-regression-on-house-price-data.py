#import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn import metrics



from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df = pd.read_csv("https://raw.githubusercontent.com/tanlitung/Datasets/master/kc_house_data.csv")

df.head()
df.isnull().sum()
df.describe()
df.shape
#dropping obvious attributes that won't make any difference to our prediction

df.drop('id', axis = 1, inplace = True)

df.drop('date', axis = 1, inplace = True)
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

col_name = df.drop('price', axis = 1).columns[:]

x = df.loc[:, col_name]

y = df['price']



# Normalizing x

x = pd.DataFrame(data = min_max_scaler.fit_transform(x), columns = col_name)



# Examine the normalized data

print(df.head())

x.head()
fig, axs = plt.subplots(ncols = 3, nrows = 3, figsize = (10, 10))

sns.regplot(y = df['price'], x = x['bathrooms'], ax = axs[0, 0])

sns.regplot(y = df['price'], x = x['bedrooms'], ax = axs[0, 1])

sns.regplot(y = df['price'], x = x['sqft_living'], ax = axs[0, 2])

sns.regplot(y = df['price'], x = x['sqft_above'], ax = axs[1, 0])

sns.regplot(y = df['price'], x = x['floors'], ax = axs[1, 1])

sns.regplot(y = df['price'], x = x['yr_built'], ax = axs[1, 2])

sns.regplot(y = df['price'], x = x['waterfront'], ax = axs[2, 0])

sns.regplot(y = df['price'], x = x['grade'], ax = axs[2, 1])

sns.regplot(y = df['price'], x = x['sqft_living15'], ax = axs[2, 2])

plt.tight_layout()
plt.figure(figsize = (10, 10))

sns.heatmap(df.corr(), annot = True)

plt.show()
#scatterplot

sns.set()

cols = ['price', 'sqft_living', 'grade', 'bathrooms', 'bedrooms']

sns.pairplot(df[cols], size = 2.5)

plt.show();
#histogram and normal probability plot

sns.distplot(df['price'], fit=norm);

fig = plt.figure()

res = stats.probplot(df['price'], plot=plt)
#applying log transformation

df['price'] = np.log(df['price'])

#transformed histogram and normal probability plot

sns.distplot(df['price'], fit=norm);

fig = plt.figure()

res = stats.probplot(df['price'], plot=plt)
#histogram and normal probability plot

sns.distplot(df['sqft_living'], fit=norm);

fig = plt.figure()

res = stats.probplot(df['sqft_living'], plot=plt)
#data transformation

df['sqft_living'] = np.log(df['sqft_living'])

#transformed histogram and normal probability plot

sns.distplot(df['sqft_living'], fit=norm);

fig = plt.figure()

res = stats.probplot(df['sqft_living'], plot=plt)
#Splitting the data

features = df.drop('price', axis = 1)

target = df['price']

train_features, test_features, train_target, test_target = train_test_split(features, target, test_size = 0.2, random_state = 5)

print("Train features shape : ", train_features.shape)

print("Train target shape   : ", train_target.shape)

print("Test features shape  : ", test_features.shape)

print("Test target shape    : ", test_target.shape)
#building model

model = LinearRegression(normalize = True)

model.fit(train_features, train_target)
print("Model intercept  : ", model.intercept_, "\n")

print("Model coefficient: ", model.coef_, "\n")



for i in range(len(features.columns)):

    print(features.columns[i], ": ", model.coef_[i])
#model evaluation

train_target_pred = model.predict(train_features)

rmse = (np.sqrt(mean_squared_error(train_target, train_target_pred)))

r2 = r2_score(train_target, train_target_pred)



# Examine the first 10 predicted output from the model

output = pd.DataFrame(train_target[0:10])

output['Predicted'] = train_target_pred[0:10]

output['Difference'] = output['Predicted'] - output['price']

print(output, "\n")



print("Model training performance:")

print("---------------------------")

print('RMSE is {}'.format(rmse))

print('R2 score is {}'.format(r2))

print("\n")
accuracy = model.score(train_features, train_target)

print(accuracy)