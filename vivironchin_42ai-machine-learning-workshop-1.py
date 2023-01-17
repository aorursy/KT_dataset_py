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
data = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
data.head()
data.info()
data.describe()
data.columns
data['ocean_proximity'].value_counts()
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#load file from image into a numpy.array (read pngs preferably)
california_img = mpimg.imread('/kaggle/input/california-housing-feature-engineering/california.png')
data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=data['population']/100, label='population', figsize=(10,7), c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
#kind: scatter = nuage de points
# alpha: transparence
# figsize: width, height in inches
# edgecolors='blue'
# linewidth=2
# cmap: A Colormap instance or registered colormap name. cmap is only used if c is an array of floats. If None, defaults to rc image.cmap
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
plt.show()
_ = data.hist(bins=50, figsize=(20,15))
#bins: In a histogram, the total range of data set (i.e from minimum value to maximum value) is divided into 8 to 15 equal parts. 
#These equal parts are known as bins or class intervals.
#figsize: figure size
data[data['median_house_value'] >= 500001].count()
data_capped = data[data['median_house_value'] < 500001]
data_capped.shape
# version sans nouveau dataframe
#index = data[(data['median_house_value'] < 500001)].index
# data.drop(index, inplace=True)

data_capped[data_capped['median_house_value'] >= 500001].count()
# clean_data = data.dropna()
# missing_values = ["n/a", "na", " ", ""]
# df = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv', na_values=missing_values)
# df.isnull().sum()

data_capped.isnull().sum()
#total le .sum() donne par colomnes, le .sum().sum() donne la somme des colonnes
data_capped.isnull().sum().sum()
round(data_capped.isnull().sum().sum() / 20640, 2)
data_capped.isna()
import seaborn as sns; sns.set()
# %matplolib inlineb
data_missing = data_capped[data_capped.total_bedrooms.isnull()]
data_missing.head()


heatmap_missing_data1 = pd.pivot_table(data_missing, values='median_house_value', index=['longitude'], columns='latitude')
sns.heatmap(heatmap_missing_data1, cmap="YlGnBu")

heatmap_missing_data2 = pd.pivot_table(data_missing, values='housing_median_age', index=['longitude'], columns='latitude')
sns.heatmap(heatmap_missing_data2, cmap="YlGnBu")
data_capped.shape

clean_data = data_capped.dropna()
# on peut utiliser subset['column'] pour ne considérer que certaines catégories
# how= any ou all si on veut delete tous les rows (axis=0, par défaut)
# inplace=True pour ne pas créer une copie mais changer la variable considérée directement
#ici on sait que seul total_bedroom a des na donc inutile de préciser le subset et le how.
clean_data.shape
clean_data.columns
clean_data['ocean_proximity'].value_counts()
clean_data.ocean_proximity
dummies = pd.get_dummies(clean_data.ocean_proximity)
dummies.head()
merged = pd.concat([clean_data, dummies], axis='columns')
merged
final = merged.drop(['ocean_proximity', 'ISLAND'], axis='columns')
final
final['rooms_p_household'] = final['total_rooms'] / final['households']
final.drop(['total_rooms'], axis='columns', inplace=True)
final
final['bedrooms_p_household'] = final['total_bedrooms'] / final['households']
final.drop(['total_bedrooms'], axis='columns', inplace=True)
final
from sklearn.model_selection import train_test_split

housing_X = final.drop("median_house_value",axis=1)#reste un dataframe
housing_y = final['median_house_value']#transformation en series

X_train, X_test, y_train, y_test = train_test_split(housing_X, housing_y, test_size=0.25, random_state=42)
print(type(housing_X))
print(housing_X.columns)
print()
print(type(housing_y))
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# scaling = MinMaxScaler()
scaling = StandardScaler()
# scaled_data = scaling.fit_transform(train)
X_train = scaling.fit_transform(X_train)
X_test= scaling.fit_transform(X_test)
print(X_train)
print(X_test)
print("X_train shape {} and size {} and type {}".format(X_train.shape,X_train.size, type(X_train)))
print("X_test shape {} and size {} and type {}".format(X_test.shape,X_test.size, type(X_test)))
print("y_train shape {} and size {} and type {}".format(y_train.shape,y_train.size, type(y_train)))
print("y_test shape {} and size {} and type {}".format(y_test.shape,y_test.size, type(y_test)))
from sklearn.linear_model import LinearRegression

model = LinearRegression()
_ = model.fit(X_train, y_train)
print("Intercept is "+str(model.intercept_))
print("coefficients  is "+str(model.coef_))
y_predictions = model.predict(X_test)
print(y_test[0:5])
print("y_test shape {} and size {} and type {}".format(y_test.shape,y_test.size, type(y_test)))
print()
print(y_predictions[0:5])
print("y_predictions shape {} and size {} and type {}".format(y_predictions.shape,y_predictions.size, type(y_predictions)))
comparaison = pd.DataFrame({'Predicted':y_predictions,'Actual':y_test})

#reset l'index qui était hérité du pandas.serise
comparaison.reset_index(inplace=True)
d
#on enleve l'index pour pouvoir mieux comparer
comparaison = comparaison.drop(['index'], axis=1)

comparaison.head()
plt.figure(figsize=(15,20))
plt.plot(comparaison[:30])
plt.legend(['Predicted', 'Actual'])
sns.jointplot(x='Predicted',y='Actual',data=comparaison[:500], kind='reg')
sns.distplot(comparaison['Predicted'], color="r")
sns.distplot(comparaison['Actual'], color="b")
from sklearn.metrics import mean_squared_error

print("MSE= ", mean_squared_error(y_test, y_predictions))
print("RMSE= ", np.sqrt(mean_squared_error(y_test, y_predictions)))
from sklearn.metrics import r2_score

r2_score(y_test, y_predictions)
model.score(X_test, y_test)