# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd

# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
%matplotlib inline
# import color maps
from matplotlib.colors import ListedColormap

# Seaborn for easier visualization
import seaborn as sns

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

from math import sqrt

# Function for splitting training and test set
from sklearn.model_selection import train_test_split

# Function to perform data standardization 
from sklearn.preprocessing import StandardScaler

# Libraries to perform hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Import classes for ML Models
from sklearn.linear_model import Ridge  ## Linear Regression + L2 regularization
from sklearn.svm import SVR ## Support Vector Regressor
from sklearn.ensemble import RandomForestRegressor ## Random Forest Regressor
from sklearn.neighbors import KNeighborsRegressor ## KNN regressor
from sklearn.tree import DecisionTreeRegressor ## Decision Tree Regressor
from sklearn import linear_model ## Lasso Regressor

# Evaluation Metrics
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae

# To save the final model on disk
from sklearn.externals import joblib
df = pd.read_csv('../input/BlackFriday.csv')
df.shape
df.head()
df.dtypes[df.dtypes=='object']
# Plot histogram grid
df.hist(figsize=(15,15), xrot=-45) ## Display the labels rotated by 45 degress

# Clear the text "residue"
plt.show()
df.describe()
df.describe(include=['object'])
plt.figure(figsize=(10,8))
sns.countplot(y='Age', data=df)
plt.figure(figsize=(10,8))
sns.countplot(y='Gender', data=df)
plt.figure(figsize=(10,8))
sns.countplot(y='City_Category', data=df)
plt.figure(figsize=(10,8))
sns.countplot(y='Stay_In_Current_City_Years', data=df)
df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(df.corr())
mask=np.zeros_like(df.corr())
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(15,10))
with sns.axes_style("white"):
    ax = sns.heatmap(df.corr()*100, mask=mask, fmt='.0f', annot=True, lw=1, cmap=ListedColormap(['green', 'yellow', 'red','blue']))
df = df.drop_duplicates()
print( df.shape )
df.Product_Category_2.unique()
df.Product_Category_2.fillna(0, inplace=True)
df.Product_Category_2.unique()
df.Product_Category_3.unique()
df.Product_Category_3.fillna(0, inplace=True)
# Display number of missing values by numeric feature
df.select_dtypes(exclude=['object']).isnull().sum()
# female: 0 and male: 1
def gender(x):
    if x=='M':
        return 1
    return 0

df['Gender']=df['Gender'].map(gender)
# Defining different age groups
def agegroup(x):
    if x=='0-17':
        return 1
    elif x=='18-25':
        return 2
    elif x ==  "26-35" :
        return 3
    elif x ==  "36-45" :
        return 4
    elif x ==  "46-50" :
        return 5
    elif x ==  "51-55" :
        return 6
    elif x ==  "55+" :
        return 7
    else:
        return 0
    
df['AgeGroup']=df['Age'].map(agegroup)
df.drop(['Age'],axis=1,inplace=True)
df['Bachelor']=((df.AgeGroup == 2) & (df.Marital_Status == 0) & (df.Gender == 1)).astype(int)
# Display percent of rows where Bachelor == 1
df[df['Bachelor']==1].shape[0]/df.shape[0]
from sklearn.preprocessing import LabelEncoder
P = LabelEncoder()
df['Product_ID'] = P.fit_transform(df['Product_ID'])
U = LabelEncoder()
df['User_ID'] = P.fit_transform(df['User_ID'])
# Create a new dataframe with dummy variables for for our categorical features.
df = pd.get_dummies(df, columns=['City_Category', 'Stay_In_Current_City_Years'])
df.head()
df.shape
df.shape
sample_df = df.sample(n=50000,random_state=100)
# Create separate object for target variable
y = sample_df.Purchase
# Create separate object for input features
X = sample_df.drop('Purchase', axis=1)
# Split X and y into train and test sets: 80-20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
train_mean = X_train.mean()
train_std = X_train.std()
## Standardize the train data set
X_train = (X_train - train_mean) / train_std
## Check for mean and std dev.
X_train.describe()
## Note: We use train_mean and train_std_dev to standardize test data set
X_test = (X_test - train_mean) / train_std
## Check for mean and std dev. - not exactly 0 and 1
X_test.describe()
## Predict Train results
y_train_pred = np.ones(y_train.shape[0])*y_train.mean()
## Predict Test results
y_pred = np.ones(y_test.shape[0])*y_train.mean()
print("Train Results for Baseline Model:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Mean Absolute Error: ", mae(y_train.values, y_train_pred))
print("Results for Baseline Model:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_test, y_pred)))
print("R-squared: ", r2_score(y_test, y_pred))
print("Mean Absolute Error: ", mae(y_test, y_pred))
tuned_params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
model = GridSearchCV(Ridge(), tuned_params, scoring = 'neg_mean_absolute_error', cv=10, n_jobs=-1)
model.fit(X_train, y_train)
model.best_estimator_
## Predict Train results
y_train_pred = model.predict(X_train)
## Predict Test results
y_pred = model.predict(X_test)
print("Train Results for Ridge Regression:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Mean Absolute Error: ", mae(y_train.values, y_train_pred))
print("Test Results for Ridge Regression:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_test, y_pred)))
print("R-squared: ", r2_score(y_test, y_pred))
print("Mean Absolute Error: ", mae(y_test, y_pred))
## Building the model again with the best hyperparameters
model = Ridge(alpha=0.0001)
model.fit(X_train, y_train)
indices = np.argsort(-abs(model.coef_))
print("The features in order of importance are:")
print(50*'-')
for feature in X.columns[indices]:
    print(feature)
tuned_params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
model = GridSearchCV(linear_model.Lasso(), tuned_params, scoring = 'neg_mean_absolute_error', cv=10, n_jobs=-1)
model.fit(X_train, y_train)
model.best_estimator_
## Predict Train results
y_train_pred = model.predict(X_train)
## Predict Test results
y_pred=model.predict(X_test)
print("Train Results for Lasso Regression:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Mean Absolute Error: ", mae(y_train.values, y_train_pred))
print("Test Results for Lasso Regression:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_test, y_pred)))
print("R-squared: ", r2_score(y_test, y_pred))
print("Mean Absolute Error: ", mae(y_test, y_pred))
## Building the model again with the best hyperparameters
model = linear_model.Lasso(alpha=0.0001)
model.fit(X_train, y_train)
indices = np.argsort(-abs(model.coef_))
print("The features in order of importance are:")
print(50*'-')
for feature in X.columns[indices]:
    print(feature)
tuned_params = {'n_estimators': [100, 200, 300, 400, 500], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
model = RandomizedSearchCV(RandomForestRegressor(), tuned_params, n_iter=20, scoring = 'neg_mean_absolute_error', cv=5, n_jobs=-1)
model.fit(X_train, y_train)
model.best_estimator_
## Predict Train results
y_train_pred = model.predict(X_train)
## Predict Test results
y_pred = model.predict(X_test)
print("Train Results for Random Forest Regression:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Mean Absolute Error: ", mae(y_train.values, y_train_pred))
print("Test Results for Random Forest Regression:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_test, y_pred)))
print("R-squared: ", r2_score(y_test, y_pred))
print("Mean Absolute Error: ", mae(y_test, y_pred))
## Building the model again with the best hyperparameters
model = RandomForestRegressor(n_estimators=200, min_samples_split=10, min_samples_leaf=4)
model.fit(X_train, y_train)
indices = np.argsort(-model.feature_importances_)
print("The features in order of importance are:")
print(50*'-')
for feature in X.columns[indices]:
    print(feature)
joblib.dump(model, 'rfr_BlackFriday.pkl') 