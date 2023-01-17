
import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
%matplotlib inline
import glob
import missingno as mssno


from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import gc
import lightgbm as lb
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix, roc_auc_score ,roc_curve,auc
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
train=pd.read_csv("../input/train.csv",sep=',')
test=pd.read_csv("../input/test.csv",sep=',')

train.shape
train.head()
test.head()
train.describe()
train.info()
#counting the null values
train.isnull().sum()
#null values visualization
mssno.bar(train,color='g',figsize=(16,5),fontsize=12)
mssno.bar(test,color='r',figsize=(16,5),fontsize=12)
#no. of unique values 
a=train.columns
a1=[len(train[col].unique()) for col in a]
sns.set(font_scale=1.2)
ax = sns.barplot(a, a1, palette='rainbow', log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique per feature')
for p, uniq in zip(ax.patches, a1):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()

#grouping by deal probability
train.groupby('deal_probability').nunique()

#Replacing NULL values with 0.
train = train.replace(np.NaN,0)
test = test.replace(np.NaN,0)
#converting to datetime format 
train.activation_date = pd.to_datetime(train.activation_date)

train['day_of_month'] = train.activation_date.apply(lambda x: x.day)
train['day_of_week'] = train.activation_date.apply(lambda x: x.weekday())
train.head()
test.activation_date = pd.to_datetime(test.activation_date)
test['day_of_month'] = test.activation_date.apply(lambda x: x.day)
test['day_of_week'] = test.activation_date.apply(lambda x: x.weekday())
test.head()
train['char_len_title'] = train.title.apply(lambda x: len(str(x)))
train['char_len_desc'] = train.description.apply(lambda x: len(str(x)))
test['char_len_title'] = test.title.apply(lambda x: len(str(x)))
test['char_len_desc'] = test.description.apply(lambda x: len(str(x)))
train.head()
cols = ['parent_category_name', 'category_name', 'price', 'user_type', 'item_seq_number', 'image_top_1','day_of_month','day_of_week','char_len_title','char_len_desc']
dummy_cols = ['parent_category_name', 'category_name','user_type']
y = train['deal_probability'].copy()
x_train = train[cols].copy()
x_test  = test[cols].copy()
del train, test; gc.collect()
n = len(x_train)
x = pd.concat([x_train, x_test])
x = pd.get_dummies(x, columns=dummy_cols)
x.head()
x_train = x.iloc[:n, :]
x_test = x.iloc[n:, :]
del x; gc.collect()


x, x_val, y, y_val = train_test_split(x_train, y, test_size=0.2, random_state=40)

# Create the LightGBM data containers
train_data = lb.Dataset(x, label=y)
val_data = lb.Dataset(x_val, label=y_val)

parameters = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 50
}

model = lb.train(parameters,
                  train_data,
                  valid_sets=val_data,
                  num_boost_round=2000,
                  early_stopping_rounds=120,
                  verbose_eval=50)


#will update as i improve the result.


