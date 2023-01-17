import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
print(os.listdir("../input"))
avocado = pd.read_csv("../input/avocado.csv", index_col = "year") # DataFrame holds 'index' (ex. sort_index)
print("shape =", avocado.shape)
avocado.describe()
avocado.columns
avocado["region"]
avocado.sample(10)
avocado.region.value_counts() # Produce Series which holds 'region' as key, its number of record as value
y = avocado.AveragePrice # Setting AveragePrice as the prediction target by dot-notation
X = avocado[["Date", "Total Volume", "region"]] # Setting Date, type and region as the features
X.describe() # Hmm...
avocado[avocado.year == 2015].AveragePrice.tail(10)
avocado.region.unique()
avocado.region.value_counts()
g1 = avocado.groupby("region") # Create tables for each region
g2 = avocado.groupby("type")
g1.describe().head(10)
g2.describe().head(10)
avocado[avocado.region.isin(["SanDiego", "Chicago"])].head(10)
avocado.groupby('region').sum()
avocado.region.replace('NewYork', 'newyork')

avocado.isnull()
# This code doesn't work
cols_with_missing = [col for col in avocado.columns 
                                 if avocado[col].isnull().any()]
train = train.drop(cols_with_missing, axis=1)
test = test.drop(cols_with_missing, axis=1)
# Apply lambda to each value
avocado.apply(lambda n: n / 2 if n.dtype == 'float' else n, axis='columns')
columns = avocado.columns
names = {'AveragePrice':'price', 'Total Volume':'volume', 'Total Bags':'bags'}
avocado = avocado.rename(columns = names)
# Create custom column
avocado.assign(rate=(avocado.price / avocado.volume))
avocado.AveragePrice.head(10).plot.bar()
avocado.sample(10).plot.scatter(x = 'Total Bags', y = 'AveragePrice') # This case has less overwrapping
avocado.sample(1000).plot.scatter(x = 'Total Bags', y = 'AveragePrice') # This case has a large overwrapping
avocado.sample(1000).plot.hexbin(x = 'Total Bags', y = 'AveragePrice', gridsize = 20)
sns.countplot(avocado.sample(1000).AveragePrice)
sns.kdeplot(avocado.sample(1000).AveragePrice)
def fit(X, y):
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)
    model = DecisionTreeRegressor(random_state=1)
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    mae = mean_absolute_error(test_y, pred_y)
    return model, mae
def formatting(d, nf, cf, tr=[]):
    """
    nf = numerical features
    cf = categolical features
    One stop function for
    - Drop NaN
    - One-hot encoding
    """
    
    # Drop NaN
    d = d[nf + cf + tr]
    d = d.dropna(axis=0)
    
    # One-hot encoding
    num_df = d[nf]
    cat_df = pd.get_dummies(d[cf])
    X = pd.concat([num_df, cat_df], axis=1)
    if len(tr) != 0:
        y = d[tr]
    else:
        y = None
    return X, y
train = avocado

nf = ['bags'] # numerical features
cf = ['region'] # categolical features
tr = ['price'] # target

X, y = formatting(train, nf, cf, tr)
model, mae = fit(X, y)
print(mae)
