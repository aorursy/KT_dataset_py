# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline



import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



import pandas as pd

import random

import math

import sklearn.linear_model



import numpy as np

np.random.seed(1)

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD

from keras import regularizers



import os

#import tarfile

import urllib 



# checks Python version (version greater than 3.5 required)

import sys

assert sys.version_info >= (3, 5)



# Checks Scikit version - (Scikit-Learn gretaer than 0.20 required)

import sklearn

assert sklearn.__version__ >= "0.20"
# Directory to save the figures

PROJECT_ROOT_DIR = "."

CHAPTER_ID = "Housing_project_Example"

IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

os.makedirs(IMAGES_PATH, exist_ok=True)



#Save figures

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):

    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)

    print("Saving figure", fig_id)

    if tight_layout:

        plt.tight_layout()

    plt.savefig(path, format=fig_extension, dpi=resolution)



# Ignore useless warnings (see SciPy issue #5998)

import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")
HOUSING_PATH = os.path.join("../input/", "california-housing-prices/")



def load_housing_data(housing_path=HOUSING_PATH):

    csv_path = os.path.join(housing_path, "housing.csv")

    return pd.read_csv(csv_path)
housing = load_housing_data()

# info functions helps us to understand the data type of all the columns

housing.info()
# Total Bedrooms have few NAN values (replaced by median values, later in the code)
#ocean proximity info

print(housing["ocean_proximity"].value_counts())



# describe function gives a summary like mean, median, std, count, etc for the numeric columns

housing.describe()
# lets check if there are missing values in the data

housing.isnull().sum()



## We can observe ``total_bedrooms`` column has 207 NAN values. It necessray to fix this before training the model 

#### There are 2 ways to treat NAN



#### 1. Delete records which are missing 

#### 2. Fill the missing values using the mean or median - which in this case is a pretty much easier.



#### To decide from all the options, we need to check the outliers.



#Box plot

import seaborn as sns

plt.figure(figsize=(15,5))

sns.boxplot(y="total_bedrooms",data=housing, orient="h", palette="plasma")

plt.plot



# There are a lot of outliers. They are filles using median, 

# (Mean could vary a lot because of outliers and can later impact the accuracy of our model)

#Fixed in later step -
#Analyzing the dependent variable - median house price .

# Firstly Histogram



plt.figure(figsize=(20,5))

sns.set_color_codes(palette="bright")

sns.distplot(housing['median_house_value'],color='g')



# We can see there is sudden increase in the median house value greater than 5,00,000, 

# this also could be outliers or also may be at some areas the prices could go really high. Need to fix this
#The bins parameter is used to customize the number of bins shown on the plots.s

housing.hist(bins=50, figsize=(20,15))

save_fig("attribute_histogram_plots")

plt.show()
##### Visulaization of data



housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,

    s=housing["population"]/100, label="population", figsize=(10,7),

    c="median_house_value", cmap=plt.get_cmap("cool"), colorbar=True,

    sharex=False)

plt.legend()

save_fig("housing_prices_scatterplot")



# There are some high density areas in california, so we can say the price of house is a bit realted to

# location as well. 



# Earlier with the data, I thought longitude & latitude would not be weak predictors

# but after plotting the graph, we can conclude even they are useful features.
# Finding Correlations

corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)
# Median income highly correlated with hose value
from pandas.plotting import scatter_matrix



attributes = ["median_house_value", "median_income", "total_rooms",

              "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12, 8))

save_fig("scatter_matrix_plot")
# Before we split the data, It can be observed that the feature - total_rooms has no significance, as this talks 

# about the rooms in the entire district. 

# Instead, we should find out, how many rooms are there in individual household, that would be more informative

# for our analysis. and also numbers like population per household
#Attribute combintaions

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]

housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]

housing["population_per_household"]=housing["population"]/housing["households"]
# Remove this feature

housing.drop("total_rooms", axis=1, inplace=True)
corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)
# Rooms per household also is correlated to median price value
#Split the Data in to test and train

#Dividing in to 5 bins (for stratification sampling

housing["income_cat"] = pd.cut(housing["median_income"],

                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],

                               labels=[1, 2, 3, 4, 5])

print(housing["income_cat"].value_counts())

plt.figure()

housing["income_cat"].hist()

# so number of incomes between 0 and 1.5 is 814, 1.5 and 3 is 6552 etc.

# Now the challenge is to divide the data in same propotion in both train and test set
# split the test data from overal set using stratification sampling

#To split train and test data with consistency among the sets. 

#For instances, the income distribution must be same in both the sets. Done using sklearn.model_selection

from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]

        
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
housing["income_cat"].value_counts() / len(housing)
# Both sets have similar (not exact) but similar counts across them.
#removing "_cat" to bring data to original state

for set_ in (strat_train_set, strat_test_set):

    set_.drop("income_cat", axis=1, inplace=True)
# Prepare the data for Machine Learning algorithms

housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set

housing_labels = strat_train_set["median_house_value"].copy()

#print(housing.info())

#print(housing_labels.describe())
# Data cleaning (incomplete rows) at command - housing.isnull().sum() (Earlier)

sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()

sample_incomplete_rows
#Intially dropping categorical columns (they are brought back later). Beasue median on cateorgical columns might impact them. 

#Imputer function to replace the NAN value present

# columns with Numerical values

housing_num = housing.drop("ocean_proximity", axis=1)



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer



num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

         ('std_scaler', StandardScaler()),

    ])



housing_num_tr = num_pipeline.fit_transform(housing_num)

print(housing_num_tr.shape)

#print(housing.shape)

# but this (housing_num_tr) does not include hotencoder 
# Bringing back categorical columns with hotencoder function for data transformation.

# Converting Categorical attribute to numeric

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder



num_attribs = list(housing_num)

#print(num_attribs)

cat_attribs = ["ocean_proximity"]



full_pipeline = ColumnTransformer([

        ("num", num_pipeline, num_attribs),

        ("cat", OneHotEncoder(), cat_attribs),

    ])



housing_prepared = full_pipeline.fit_transform(housing)

print(housing_prepared)

# Trian the model

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



#RandomizedSearchCV method

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint



param_distribs = {

        'n_estimators': randint(low=1, high=200),

        'max_features': randint(low=1, high=8),

    }



forest_reg = RandomForestRegressor(random_state=42)

rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,

                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)

rnd_search.fit(housing_prepared, housing_labels)
# mean scores for few of the combinations

cvres = rnd_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
feature_importances = rnd_search.best_estimator_.feature_importances_

feature_importances
# scores of each of the attributes

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

#cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution

cat_encoder = full_pipeline.named_transformers_["cat"]

cat_one_hot_attribs = list(cat_encoder.categories_[0])

attributes = num_attribs + extra_attribs + cat_one_hot_attribs

sorted(zip(feature_importances, attributes), reverse=True)
#Evaluation on test set

final_model = rnd_search.best_estimator_



X_test = strat_test_set.drop("median_house_value", axis=1)

y_test = strat_test_set["median_house_value"].copy()



X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)



final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)



final_rmse
# Plot Actual vs. Predicted



test = pd.DataFrame({'Predicted':final_predictions,'Actual':y_test})

fig= plt.figure(figsize=(16,8))

test = test.reset_index()

test = test.drop(['index'],axis=1)

plt.plot(test[1::50])

plt.legend(['Actual','Predicted'])

sns.jointplot(x='Actual',y='Predicted',data=test,kind='reg',color="grey")
## Thank you all for reading

 

#### If you have any doubts please right back to me