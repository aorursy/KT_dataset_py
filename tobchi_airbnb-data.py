import pandas as pd

import numpy as np

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
dataset = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
dataset.head()
dataset.info()
dataset.isnull().sum()
dataset.duplicated().sum()
dataset.drop(['name', 'host_name', 'last_review', 'id'], axis=1, inplace=True)
dataset.fillna({'reviews_per_month' : 0}, inplace=True)
# Checking for nulls

dataset.isnull().sum()
dataset.describe()
corr_matrix = dataset.corr()



# Visualising all correlations



plt.figure(figsize=(10, 8))

ax4 = sns.heatmap(

    corr_matrix, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True,

    cbar_kws={'label': 'Correlation'}) 



ax4.set_title('Correlation Matrix of Attributes')
# Finding correlation between the price and independent variables

corr_matrix["price"].sort_values(ascending=False)
# Geographical plot of prices below $500



price_less_500 =dataset[dataset.price < 500]



plt.figure(figsize=(12,10))

nyc_img=plt.imread('../input/new-york-city-airbnb-open-data/New_York_City_.png', 0)

plt.imshow(nyc_img,zorder=0,extent=[-74.26, -73.687, 40.49,40.92])

ax=plt.gca()

price_less_500.plot(kind='scatter', x='longitude', y='latitude', label='Listing', c='price', ax=ax, 

           cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.3)

plt.xlabel('Longitude', size=15)

plt.ylabel('Latitude', size=15)

plt.title('New York Airbnb 2019 Price Heat-map (< $500)', size=15)

plt.show()
# Geographical plot of neighbourhood groups



plt.figure(figsize=(12,10))

sns.set_style("whitegrid")

plt.imshow(nyc_img,zorder=0,extent=[-74.26, -73.687, 40.49,40.92])

ax=plt.gca()

ax1 = sns.scatterplot(x='longitude', y='latitude', data=dataset, hue='neighbourhood_group', alpha=0.7)

ax1.set(xlabel='Longitude', ylabel='Latitude', title='Listings in Each Neighbourhood Group')

plt.show()

# Bar count of listings in each neighbourhood group



plt.figure(figsize=(12, 10))

sns.set_style("whitegrid")

ax2 = sns.catplot(x='neighbourhood_group', kind='count', data=dataset, palette="ch:.025")

ax2.set(xlabel='Neighbourhood Group', ylabel='Number of Listings', title='Total listings in each neighbourhood group')
dataset['neighbourhood_group'].value_counts()
# Bar count of listings of room types



plt.figure(figsize=(12, 10))

sns.set_style("whitegrid")

ax3 = sns.catplot(x='room_type', kind='count', data=dataset, palette='magma')

ax3.set(xlabel='Room Type', ylabel='Number of Listings', title='Total listings of Each Room Type')
dataset['room_type'].value_counts()
# Bar count of listings of room types in each neighbourhood group



plt.figure(figsize=(12, 10))

ax4 = sns.countplot(dataset['room_type'],hue=dataset['neighbourhood_group'], palette='mako')

ax4.set_xlabel('Room Type')

ax4.set_ylabel('Number of Listings')

ax4.set_title('Room Types for Each Neighbourhood Group')
# Distribution of prices in neighbourhood groups



ax5 = sns.catplot(x="neighbourhood_group", y="price", kind="violin", data=dataset)

ax5.set(xlabel='Neighbourhood Group', ylabel='Price', title='Distribtuion of Prices in Neighbourhood Groups (< $500)')

plt.ylim(0, 500)
dataset.drop(['host_id'], axis=1, inplace=True)

dataset.head()
# Splitting the data into independent and dependent variables



X = dataset.iloc[:, dataset.columns != 'price'].values

y = dataset.iloc[:, 5].values

y = y.reshape(-1, 1)
# Label encoding categorical variables



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encode = LabelEncoder()



X[:, 0] = label_encode.fit_transform(X[:, 0])

X[:, 1] = label_encode.fit_transform(X[:, 1])

X[:, 4] = label_encode.fit_transform(X[:, 4])



X_labelenc = pd.DataFrame(X)
X_labelenc.head()
# One hot encoding labeled variables



hotencode = OneHotEncoder(categorical_features=[0, 1, 4])



dataset_encoded = hotencode.fit_transform(X_labelenc).toarray()
dataset_encoded.shape
# Converting array back into a dataframe



df = pd.DataFrame(dataset_encoded)

df.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.metrics import r2_score, mean_absolute_error

regressor=LinearRegression()

regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)



r2_score(y_test,y_pred)



print("""

        Mean Squared Error: {}

        R-squared Score: {}

        Mean Absolute Error: {}

     """.format(

        np.sqrt(metrics.mean_squared_error(y_test, y_pred)),

        r2_score(y_test,y_pred) * 100,

        mean_absolute_error(y_test,y_pred)

        ))
# Comparing actual values with predictions



error_airbnb = pd.DataFrame({

        'Actual Values': np.array(y_test).flatten(),

        'Predicted Values': y_pred.flatten()}).head(20)



error_airbnb.head(5)
from sklearn.tree import DecisionTreeRegressor

DTree=DecisionTreeRegressor(min_samples_leaf=.0001)

DTree.fit(X_train,y_train)

y_pred2=DTree.predict(X_test)

r2_score(y_test,y_pred2)



print("""

        Mean Squared Error: {}

        R-squared Score: {}

        Mean Absolute Error: {}

     """.format(

        np.sqrt(metrics.mean_squared_error(y_test, y_pred2)),

        r2_score(y_test,y_pred2) * 100,

        mean_absolute_error(y_test,y_pred2)

        ))
# Comparing actual values with predictions



error_airbnb2 = pd.DataFrame({

        'Actual Values': np.array(y_test).flatten(),

        'Predicted Values': y_pred2.flatten()}).head(20)



error_airbnb2.head(5)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(DTree, X_train, y_train, scoring="neg_mean_squared_error", cv=10)

Dtree_rmse_scores = np.sqrt(-scores)
print("""

        Scores: {}

        Mean: {}

        Standard deviation: {}

        """.format(

        Dtree_rmse_scores,

        Dtree_rmse_scores.mean(),

        Dtree_rmse_scores.std())

     )
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()

forest_reg.fit(X_train, y_train.ravel())

y_pred3 = forest_reg.predict(X_test)

r2_score(y_test,y_pred3)



print("""

        Mean Squared Error: {}

        R-squared Score: {}

        Mean Absolute Error: {}

     """.format(

        np.sqrt(metrics.mean_squared_error(y_test, y_pred3)),

        r2_score(y_test,y_pred3) * 100,

        mean_absolute_error(y_test,y_pred3)

        ))
scores = cross_val_score(forest_reg, X_train, y_train.ravel(), scoring="neg_mean_squared_error", cv=10)

Rforest_rmse_scores = np.sqrt(-scores)
print("""

        Scores: {}

        Mean: {}

        Standard deviation: {}

        """.format(

        Rforest_rmse_scores,

        Rforest_rmse_scores.mean(),

        Rforest_rmse_scores.std())

     )
from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators': [3, 10, 30, 50], 'max_features': [2, 4, 6, 8]},

    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}

]



forest_reg = RandomForestRegressor()



grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')



grid_search.fit(X_train, y_train.ravel())
print("""

        Best parameters: {}

        """.format(

        grid_search.best_params_,)

     )
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):

    print(np.sqrt(-mean_score), params)
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint



param_dist = {

        'n_estimators': randint(low=1, high=200),

        'max_features': randint(low=1, high=8),

    }



random_search = RandomizedSearchCV(forest_reg, param_distributions=param_dist ,n_iter=10, cv=5, scoring='neg_mean_squared_error')

random_search.fit(X_train, y_train.ravel())
print("""

        Best parameters: {}

        """.format(

        random_search.best_params_,)

     )
cvres2 = random_search.cv_results_

for mean_score, params in zip(cvres['mean_test_score'], cvres2['params']):

    print(np.sqrt(-mean_score), params)