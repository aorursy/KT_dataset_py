import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder,StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score,auc

from lightgbm import LGBMClassifier

train_data = pd.read_csv('../input/janatahack-crosssell-prediction-dataset/train.csv',index_col = 'id')

test_data = pd.read_csv('../input/janatahack-crosssell-prediction-dataset/test.csv',index_col = 'id')

combined = [train_data,test_data]
train_data.head()
train_data.isnull().sum()
test_data.info()
train_data.describe()
cat_features = train_data.select_dtypes(['object']).columns

cat_features
for col in cat_features:

    print(train_data[col].nunique())
le = LabelEncoder()

for col in cat_features:

    train_data[col] = le.fit_transform(train_data[col])

    test_data[col] = le.fit_transform(test_data[col])
train_data.info()
train_data.describe()
plt.figure(figsize = (9,6))

sns.set_style('ticks')

sns.countplot('Gender', hue = 'Response',data = train_data)

plt.show()
train_data['Age'].nunique()
plt.figure(figsize = (9,6))

sns.set_style('ticks')

sns.violinplot('Response', 'Age',data = train_data)

plt.show()
sc = StandardScaler()

col = 'Age'

train_data[col] = sc.fit_transform(train_data[col].values.reshape(-1,1))

test_data[col] = sc.transform(test_data[col].values.reshape(-1,1))
train_data['Driving_License'].nunique()
train_data[train_data['Driving_License'] == 1]['Response'].count()
plt.figure(figsize = (9,6))

sns.set_style('ticks')

sns.countplot('Driving_License', hue = 'Response',data = train_data)

plt.show()
train_data['Region_Code'].nunique()
plt.figure(figsize = (9,6))

sns.set_style('ticks')

sns.violinplot('Response', 'Region_Code',data = train_data)

plt.show()
train_data['Previously_Insured'].nunique()
plt.figure(figsize = (9,6))

sns.set_style('ticks')

sns.countplot('Previously_Insured', hue = 'Response',data = train_data)

plt.show()
plt.figure(figsize = (9,6))

sns.set_style('ticks')

sns.countplot('Vehicle_Age', hue = 'Response',data = train_data)

plt.show()
train_data['Vehicle_Damage'].nunique()
plt.figure(figsize = (9,6))

sns.set_style('ticks')

sns.countplot('Vehicle_Damage', hue = 'Response',data = train_data)

plt.show()
train_data['Annual_Premium'].nunique()
plt.figure(figsize = (9,6))

sns.set_style('ticks')

sns.violinplot('Response', 'Annual_Premium',data = train_data)

plt.show()
# dataset = train_data

# index = dataset[dataset['Annual_Premium'] >= 115000]['Annual_Premium'].index.values

# dataset.drop(index,inplace = True)
sc = StandardScaler()

col = 'Annual_Premium'

train_data[col] = sc.fit_transform(train_data[col].values.reshape(-1,1))

test_data[col] = sc.transform(test_data[col].values.reshape(-1,1))
plt.figure(figsize = (9,6))

sns.set_style('ticks')

sns.violinplot('Response', 'Annual_Premium',data = train_data)

plt.show()
train_data['Policy_Sales_Channel'].nunique()
mms = MinMaxScaler()

col = 'Policy_Sales_Channel'

train_data[col] = mms.fit_transform(train_data[col].values.reshape(-1,1))

test_data[col] = mms.transform(test_data[col].values.reshape(-1,1))
sc = StandardScaler()

col = 'Policy_Sales_Channel'

train_data[col] = sc.fit_transform(train_data[col].values.reshape(-1,1))

test_data[col] = sc.transform(test_data[col].values.reshape(-1,1))
plt.figure(figsize = (9,6))

sns.set_style('ticks')

sns.violinplot('Response', 'Policy_Sales_Channel',data = train_data)

plt.show()
train_data['Vintage'].nunique()
plt.figure(figsize = (9,6))

sns.set_style('ticks')

sns.violinplot('Response', 'Vintage',data = train_data)

plt.show()
train_data.head()
plt.figure(figsize = (9,6))

sns.heatmap(train_data.corr(),annot=True,cbar = True)

plt.show()
y = train_data.loc[:,'Response']

X = train_data.drop('Response',axis = 1)

X


x_train,x_val,y_train,y_val = train_test_split(X,y,test_size = 0.2,random_state = 42)

model = LGBMClassifier(boosting_type='gbdt',n_estimators=500,depth=10,learning_rate=0.04,objective='binary',metric='auc',is_unbalance=True,

                 colsample_bytree=0.5,reg_lambda=2,reg_alpha=2,random_state=294,n_jobs=-1)



clf = model.fit(x_train,y_train)
roc_auc_score(y_train,clf.predict(x_train))
roc_auc_score(y_val,clf.predict_proba(x_val)[:,1])
y_pred = clf.predict_proba(test_data)[:,1]
sub = pd.DataFrame(y_pred,columns = ['Response'],index = [i + 381110 for i in range(test_data.shape[0])])



sub.index.name = 'id'

sub.tail()
sub.to_csv('sub.csv')