import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample,shuffle

test = pd.read_csv("../input/health-insurance-cross-sell-prediction/test.csv")
train = pd.read_csv("../input/health-insurance-cross-sell-prediction/train.csv")
test_id = test['id']

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train.head()
def info(df):
    print("_______Info_______")
    print(df.info())
    print("_______Describe_______")
    print(df.describe())
    print("_______Columns_______")
    print(df.columns)
    print("_______Data Types_______")
    print(df.dtypes)
    print("_______Missing Values_______")
    print(df.isnull().sum())
    print("_______NULL values_______")
    print(df.isna().sum())

info(train) 
import matplotlib.pyplot as plt
import seaborn as sns

def plot(df,feat,palette='rainbow'):
    plt.style.use('seaborn')
    sns.set_style('whitegrid')

    labels=df[feat].value_counts().index
    values=df[feat].value_counts().values
    
    plt.figure(figsize=(15,5))

    ax = plt.subplot2grid((1,2),(0,0))
    sns.barplot(x=labels, y=values,palette=palette, alpha=0.75)
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2., height + 0.1, values[i],ha="center")
    plt.title('Response of Customer', fontsize=15, weight='bold')    
    plt.show()
plot(train, 'Response', 'Blues')
plt.show()
train["Vehicle_Age"].value_counts()
plot(train, 'Vehicle_Age', 'Blues')
plt.show()
train["Driving_License"].value_counts()
df_majority = train[train['Response']==0]
df_minority = train[train['Response']==1]

df_minority_upsampled = resample(df_minority,replace=True,n_samples=334399,random_state = 123)

train = pd.concat([df_minority_upsampled,df_majority])

train = shuffle(train)

train
train.Response.value_counts()
def preprocessing(df):
    
    #dictionary for encoding vehicle_age    
    age_dict = {'> 2 Years': 2,
            '1-2 Year': 1,
            '< 1 Year': 0}
    
    df["Vehicle_Age"] = df["Vehicle_Age"].map(age_dict)
    
    #encoding gender male/female
    df["Gender"] = np.where(df['Gender'] == "Female", 0, 1)
    
    #encoding vehicle damage yes/no
    df["Vehicle_Damage"] = np.where(df['Vehicle_Damage'] == "No", 0, 1)
    
    df.drop(["id"], axis=1, inplace=True)
        
    #df.drop(["Gender"], axis=1, inplace=True)
    
    df.drop(["Driving_License"], axis=1, inplace=True)
    
    # normalization
    cols_to_norm = ['Age','Annual_Premium', 'Region_Code', 'Policy_Sales_Channel', 'Vintage']
    
    df[cols_to_norm] = StandardScaler().fit_transform(df[cols_to_norm])  
    
    
    return df


test = preprocessing(test)
train = preprocessing(train)
train
X = train.loc[:, train.columns != 'Response']
y = train["Response"]

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=1)

X_train
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
LR  = LogisticRegression()
#set parameters
C = {
             'C': [0.01, 0.1, 1, 10, 100],
            }

grid_lr = GridSearchCV(estimator=LR,
                       param_grid=C,
                       scoring='roc_auc',
                       cv=5,
                       n_jobs=-1)
#grid fit
grid_lr.fit(X_train, y_train)

print(grid_lr.best_params_)
LR  = LogisticRegression(C=10)

LR.fit(X_train,y_train)

predictions = LR.predict(X_val)

print(f"ROC AUC Score Logistic Regression: {roc_auc_score(y_val, predictions)}")
import tensorflow 
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
classifier = Sequential()

classifier.add(Dense(32,activation='tanh',kernel_initializer='uniform',input_dim=9))
classifier.add(Dense(32,activation='relu',kernel_initializer='uniform'))


classifier.add(Dense(1,activation="sigmoid", kernel_initializer="uniform"))
classifier.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics = ['AUC'])

classifier.fit(X_train, y_train, batch_size=5, epochs=5)    
y_pred_test = classifier.predict(X_val)

roc_auc = roc_auc_score(y_val, y_pred_test)

print(roc_auc)
classifier.fit(X, y, batch_size=5, epochs=5)
y_pred = classifier.predict(test)
y_pred = pd.DataFrame(y_pred)
submission = pd.concat([test_id,
                        y_pred],
                       axis=1)

submission.columns = ['id','Response']

submission.to_csv('./submission.csv', index=False)
