import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv("../input/training-dataset-from-kaggle/train.csv")
df.head()
df.info()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap="YlGnBu")
df['Survived'].value_counts()
sns.set_style("whitegrid")
sns.countplot(x='Survived',hue="Sex",data=df)
plt.figure(figsize=(12,8))
sns.boxplot(x="Pclass",y=df["Age"].dropna(),data=df)
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap="YlGnBu")
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.drop("Cabin",axis=1,inplace=True)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap="YlGnBu")
df_obj = df.select_dtypes(include=["object"]).copy()
df_obj.columns
df["Embarked"].value_counts()
col_obj = ['Sex', 'Embarked']
def get_dummies(cols):
    get_dumm = pd.DataFrame()
    for c in cols:
        df1 = pd.get_dummies(df[c],drop_first=True)
        if get_dumm.empty:
            get_dumm = df1
        else:
            df1 = pd.concat([get_dumm,df1],axis=1)
    return df1
        
obj_df = get_dummies(col_obj)
obj_df.head()
final_df = pd.concat([df,obj_df],axis=1)
final_df.head()
final_df.drop(['PassengerId','Name','Sex','Ticket','Embarked'],inplace=True,axis=1)
final_df.head()
corr_matrix=final_df.corr()
corr_matrix["Survived"].sort_values(ascending=False)
final_df.shape
X_train = final_df.drop("Survived",axis=1)
y_train = final_df["Survived"]
X_train.shape
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
test_df = pd.read_csv("../input/test-dataset/test_data_comp.csv")
test_df.shape
clf = xgb.XGBClassifier()
clf.fit(X_train,y_train)
pred = clf.predict(test_df)
pred
clf2 = LogisticRegression()
clf2.fit(X_train,y_train)
pred2 = clf2.predict(test_df)
pred2
clf3 = RandomForestClassifier()
clf3.fit(X_train,y_train)
pred3 = clf3.predict(test_df)
pred3
y_train.shape
pred2.shape
pred2.columns = ["Survived"]
pred2.head()
test_df.head()
temp_test_df = pd.concat([test_df,pred2],axis=1)
temp_test_df.head()
X_train_test = temp_test_df.drop('Survived',axis=1)
y_train_test = temp_test_df['Survived']
final_train_x = pd.concat([X_train,X_train_test],axis=0)
final_train_y = pd.concat([y_train,y_train_test],axis=0)
final_train_x.shape,final_train_y.shape
clf2.fit(final_train_x,final_train_y)
pred4 = clf2.predict(test_df)
pred4
from sklearn.model_selection import RandomizedSearchCV
param_grid_LR = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],'class_weight':['balanced', None],'max_iter':[100,200,300,400,1000,1500] }
LR_model = LogisticRegression()
random_search_LR=RandomizedSearchCV(LR_model,param_distributions=param_grid_LR
                                    ,n_iter=30,scoring='roc_auc',n_jobs=4,cv=5,verbose=3)
random_search_LR.fit(X_train,y_train)
random_search_LR.best_estimator_
new_LR_clf = LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
new_LR_clf.fit(X_train,y_train)
pred_new_LRclf = new_LR_clf.predict(test_df)
pred_new_LRclf
X_train.shape,y_train.shape
from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

model = Sequential()
model.add(Dense(output_dim=50,init = 'he_uniform',activation='sigmoid',input_dim = 8))
model.add(Dropout(0.2))
model.add(Dense(output_dim=25,init = 'he_uniform',activation='sigmoid'))
#model.add(Dropout(0.2))
model.add(Dense(output_dim=50,init = 'he_uniform',activation='sigmoid'))
model.add(Dense(output_dim = 1, init = 'he_uniform',activation='linear'))
model.compile(loss= root_mean_squared_error,optimizer='Adamax')

history = model.fit(X_train, y_train,validation_split=0.12, batch_size = 10, nb_epoch = 10)
pred_ann=model.predict(test_df)
pred_ann
pred_ann = pd.DataFrame(pred_ann)
sub_df=pd.read_csv("gender_submission.csv")
datasets = pd.concat([sub_df['PassengerId'],pred_ann],axis=1)
datasets.columns = ['PassengerId','Survived']
datasets.to_csv("sample_submission_ann_simple.csv",index=False)