# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Ignore warnings

#import warnings

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keras import layers

from keras.layers import Input, Dense, Activation,Dropout

from keras.models import Model

from keras.callbacks import ReduceLROnPlateau

from keras.optimizers import Adam

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.tree import export_graphviz, DecisionTreeClassifier,DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV,StratifiedKFold

from xgboost import XGBClassifier

from sklearn import metrics

import os

print(os.listdir("../input"))

#warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
def cleanupFrame(data_frame):

    

    #total Nulls

    #print(test.isnull().sum(axis = 0))

    #print(data_frame[data_frame['PassengerId']==1306].Name)

    #changing Cabin column

    

    cabin={'A' : 0, 'B' : 1, 'C':2, 'D':3, 'E':4 , 'F':5 , 'T':6 , 'G':7}

    data_frame['Cabin']=data_frame['Cabin'].str.extract('([A-Z][0-9])', expand=False).str[:1].map(cabin)

    data_frame['Cabin']=data_frame['Cabin'].fillna(8)

    data_frame = pd.get_dummies(data_frame, columns = ["Cabin"],prefix="Cabin")

    

    #changing Name column

    name={'Mrs.':0,'Mr.':1,'Master.':2,'Miss.':3,'Major.':4,'Rev.':4,'Dr.':4,'Ms.':3,'Mlle.':3,'Col.':4,'Capt.':4,'Mme.':0,'Countess.':4,'Don.':4,'Jonkheer.':4,'Sir.':4,'Lady.':4,'Dona.':4 }

    data_frame['Name']=data_frame['Name'].str.extract('(Mrs\.|Mr\.|Master\.|Miss\.|Major\.|Rev\.|Dr\.|Ms\.|Mlle\.|Col\.|Capt\.|Mme\.|Countess\.|Don\.|Jonkheer\.|Sir\.|Lady\.|Dona.)', expand=False).str[:].map(name)

    data_frame['Name']=data_frame['Name'].fillna(lambda x: 0 if data_frame['Sex'] == 'female' else 1)

    #g = sns.heatmap(data_frame[["Age","Name","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)

    data_frame = pd.get_dummies(data_frame, columns = ["Name"],prefix="Name")

    #print(data_frame[['Name','Survived']].groupby(['Name']).mean())

    #print(data_frame[data_frame['Name'].isnull()])

    #print(data_frame.groupby(['Name'])['Name'].count())

    

    #change Sex

    sex = {'male' : 0, 'female' : 1}

    data_frame['Sex']=data_frame['Sex'].map(sex)

    data_frame['Sex']=data_frame['Sex'].fillna(0)

    

    #change fare

    data_frame['Fare']=data_frame['Fare'].fillna(data_frame['Fare'].median())

    data_frame['Fare_bin'] = pd.qcut(data_frame['Fare'],5,labels=[1,2,3,4,5]).astype(int)

    data_frame = pd.get_dummies(data_frame, columns = ["Fare_bin"],prefix="Fare_bin")

    #print(data_frame['Fare_bin'].value_counts())

    

    #change Age

    #print(data_frame['Age'].min(),data_frame['Age'].max(),data_frame['Age'].mean())

    #data_frame['Age']=data_frame['Age'].fillna(data_frame['Age'].mean())

    #data_frame['Age']=data_frame['Age'].astype(int)

    # Index of NaN age rows

       

    # Edited to use Random Forest

    index_NaN_age = list(data_frame["Age"][data_frame["Age"].isnull()].index)



    for i in index_NaN_age :

        age_med = data_frame["Age"].median()

        age_pred = data_frame["Age"][((data_frame['SibSp'] == data_frame.iloc[i]["SibSp"]) & (data_frame['Parch'] == data_frame.iloc[i]["Parch"]) 

                                  & (data_frame['Pclass'] == data_frame.iloc[i]["Pclass"]))].median()

        if not np.isnan(age_pred) :

            data_frame['Age'].iloc[i] = age_pred

        else :

            data_frame['Age'].iloc[i] = age_med 

    """ 

    df_sub = data_frame[['Age','Name_0','Name_1','Name_2','Name_3','Name_4','Fare_bin_1','Fare_bin_2','Fare_bin_3','Fare_bin_4','SibSp']]

    X_age_train  = df_sub.dropna().drop('Age', axis=1).astype(int)

    Y_age_train  = data_frame['Age'].dropna()

    X_age_test = df_sub.loc[np.isnan(data_frame.Age)].drop('Age', axis=1)

     

    regressor =RandomForestRegressor(n_estimators = 300)

    regressor.fit(X_age_train.values, Y_age_train.values)

    y_pred = np.round(regressor.predict(X_age_test),1)

    data_frame.Age.loc[df.Age.isnull()] = y_pred

    """

    bins = [ 0, 4, 12, 18, 30, 50, 65, 100] # This is somewhat arbitrary...

    age_index = (1,2,3,4,5,6,7)

    #('baby','child','teenager','young','mid-age','over-50','senior')

    data_frame['Age_bin'] = pd.cut(data_frame['Age'], bins, labels=age_index).astype(int)

    data_frame = pd.get_dummies(data_frame, columns = ["Age_bin"],prefix="Age_bin")

    

    #Changing Embarked -- map default to S as majority is embarked from there 

    embarked={'S' : 0, 'C' : 1, 'Q':2}

    data_frame['Embarked']=data_frame['Embarked'].map(embarked)

    data_frame['Embarked']=data_frame['Embarked'].fillna(0)

    data_frame = pd.get_dummies(data_frame, columns = ["Embarked"],prefix="Embarked")

    



    #changing Ticket column keeping first letter ['A','W','F','L','5','6','7','8','9'] with 20 & C:10,P:11 S:12

    

    ticket={'1':1,'2':2,'3':3,'4':4,'C':10,'P':11,'S':12,'A':20,'W':20,'F':20,'L':20,'5':20,'6':20,'7':20,'8':20,'9':20}

    data_frame['Ticket'] = data_frame['Ticket'].map(lambda x: x[0])

    data_frame['Ticket'] = data_frame['Ticket'].map(ticket)

    data_frame = pd.get_dummies(data_frame, columns = ["Ticket"],prefix="Ticket")

    

    #introducce a new column 'Family Size' Combining familt size >4 into others

    data_frame['Familly_size'] = data_frame['SibSp'] + data_frame['Parch'] + 1

    data_frame['Familly_size'] = data_frame['Familly_size'].map(lambda x: 0 if x > 4 else x)

    data_frame = pd.get_dummies(data_frame, columns = ["Familly_size"],prefix="Familly_size")

    

    #Putting PCass in Dummies 

    data_frame = pd.get_dummies(data_frame, columns = ["Pclass"],prefix="Pclass")

    

    #Drop Columns

    data_frame=data_frame.drop(['PassengerId','SibSp','Parch','Fare','Age'], axis=1) 

    #print(data_frame.describe(),data_frame.shape,data_frame.iloc[0] )

    

    return data_frame    
def read_csv(df):

     

    test=cleanupFrame(df)

    

    #dropping rows with Nulls

    #test=test.dropna()

    #print(test.isnull().sum(axis = 0))

    #Normalize Classifiers

    #K=(test-test.mean())/test.std()

    Y_train=test[0:891]['Survived'].values

    X_train=test[0:891].drop(['Survived'], axis=1).values

    X_test=test[891:].drop(['Survived'], axis=1).values

    print(test.head())

    return X_train, Y_train,X_test
def write_csv(filename,predictions):

# Writing a CSV file submission. PassengerId,Prediction 

    my_submission = pd.DataFrame({'PassengerId': range(892,892+predictions.shape[0]), 'Survived': 0})

    my_submission['Survived']=predictions

    #print(my_submission['PassengerId'], my_submission['Survived'])

    # you could use any filename. We choose submission here

    my_submission.to_csv('submission.csv', index=False)
df_train=pd.read_csv("../input/train.csv")

df_test=pd.read_csv("../input/test.csv")

df=df_train.append(df_test , ignore_index = True)

X_train, Y_train,X_test=read_csv(df)

print(X_train.shape, Y_train.shape,X_test.shape)
"""

def TitanicModel(input_shape):

    

       X_train size[None,classifiers]

       Y_train size[None,]

    

    # Define the inputpadding = 'same', placeholder as a tensor with shape input_shape. Think of this as your input image!

    X_input = Input(input_shape)

    X=Dense(224,  activation = "relu")(X_input)

    X=Dropout(p=0.10)(X)

    X=Dense(120,  activation = "relu")(X)

    X=Dropout(p=0.20)(X)

    X=Dense(56, activation = "relu")(X)

    X=Dropout(p=0.30)(X)

    X=Dense(1,  activation = "sigmoid")(X)

    # Create model

    model = Model(inputs = X_input, outputs = X, name='TitanicModel')    

    return model

"""
"""

#Neural Network

titanicModel = TitanicModel(X_train[0].shape)

titanicModel.summary()

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)

titanicModel.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

titanicModel.fit(x=X_train, y=Y_train, epochs=200, batch_size=50)





test_predictions=titanicModel.predict(X_test)

test_predictions=(test_predictions>0.5)*1

test_predictions=np.reshape(test_predictions,-1)

write_csv('submission.csv',test_predictions)

"""
"""

#Gives V22 - 79.25% accuracy%

#Decision Tree

dtModel = DecisionTreeClassifier(min_samples_split=15, min_samples_leaf=20, random_state=42)

dtModel.fit(X_train, Y_train)

# Predict for train data sample

dtModel_prediction = dtModel.predict(X_train)

# Compute error between predicted data and true response and display it in confusion matrix

score = metrics.accuracy_score(Y_train, dtModel_prediction)

print("Training Score:",score)



#Test Prediction

test_predictions=dtModel.predict(X_test)

test_predictions=(test_predictions>0.5).astype(int)

test_predictions=np.reshape(test_predictions,-1)

write_csv('submission.csv',test_predictions)

"""


#Gives 92 on training but on test set only 76% on test data by default Settings

#Gives 81.1 with V30 -- RandomForestClassifier(n_estimators=200,min_samples_split=5, min_samples_leaf=4, random_state=42). Any move change any of parameters results in degrading performance. 

# Create and train model on train data sample Random Forest

rfModel = RandomForestClassifier(n_estimators=150,min_samples_split=5, min_samples_leaf=5,oob_score=True,n_jobs=-1, random_state=42)

rfModel.fit(X_train, Y_train)

print("%.4f" % rfModel.oob_score_)

#Predict for train data sample

rfModel_prediction = rfModel.predict(X_train)

#test=list(rfModel_prediction)

#print((test==Y_train).sum(axis = 0)/891)

score = metrics.accuracy_score(Y_train, rfModel_prediction)

print("Training Score:",score,)



#Test Prediction

test_predictions=rfModel.predict(X_test)

test_predictions=(test_predictions>0.5)*1

test_predictions=np.reshape(test_predictions,-1)

write_csv('submission.csv',test_predictions) 
"""

#AdaBoost

# Create adaboost classifer object

#dtModel = DecisionTreeClassifier(min_samples_split=15, min_samples_leaf=20, random_state=42)

#adaBoostModel =AdaBoostClassifier()

DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

kfold = StratifiedKFold(n_splits=10)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[10,20,30,40,50],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1,1.5]}



adaBoostModel = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



adaBoostModel.fit(X_train,Y_train)

#adaBoostModel_prediction = list(adaBoostModel.predict(X_train))

#score=(adaBoostModel_prediction==Y_train).sum(axis=0)/891

print("Training Score:",adaBoostModel.best_estimator_,adaBoostModel.best_score_ )





#Test Prediction

test_predictions=adaBoostModel.predict(X_test)

test_predictions=(test_predictions>0.5)*1

test_predictions=np.reshape(test_predictions,-1)

write_csv('submission.csv',test_predictions) 

"""
"""

#XBoost Gradient Boost

kfold = StratifiedKFold(n_splits=10)

xgb_param_grid = {

                  'reg_alpha':[0.1]

                  }

xgb = XGBClassifier(learning_rate =0.001,

                         n_estimators=800,

                         max_depth=6,

                         min_child_weight=5,

                         gamma=0,

                         subsample=0.8,

                         colsample_bytree=0.8,

                         objective= 'binary:logistic',

                         nthread=4,

                         scale_pos_weight=1,

                         reg_alpha=0.1,

                         seed=27)

xgbModel = GridSearchCV(xgb,param_grid = xgb_param_grid, cv=kfold, scoring='roc_auc', n_jobs= 4, verbose = 1)

xgbModel.fit(X_train, Y_train)

print("Best: %f using %s" % (xgbModel.best_score_, xgbModel.best_params_))

means = xgbModel.cv_results_['mean_test_score']

stds = xgbModel.cv_results_['std_test_score']

params = xgbModel.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))

# Predict for train data sample

xgbModel_prediction = xgbModel.predict(X_train)

# Compute error between predicted data and true response and display it in confusion matrix

score = metrics.accuracy_score(Y_train, xgbModel_prediction)

print("Training Score:",score)

#Test Prediction

test_predictions=xgbModel.predict(X_test)

test_predictions=(test_predictions>0.5)*1

test_predictions=np.reshape(test_predictions,-1)

write_csv('submission.csv',test_predictions) """
"""

# Gradient boosting tunning

kfold = StratifiedKFold(n_splits=10)

GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

                  'n_estimators' : [100,200,300],

                  'learning_rate': [0.1, 0.05, 0.01],

                  'max_depth': [4, 8],

                  'min_samples_leaf': [100,150],

                  'max_features': [0.3, 0.1] 

                  }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsGBC.fit(X_train,Y_train)



GBC_best = gsGBC.best_estimator_



# Best score

print(gsGBC.best_score_)

#Test Prediction

#test_predictions=gsGBC.predict(X_test)

#test_predictions=(test_predictions>0.5)*1

#test_predictions=np.reshape(test_predictions,-1)

#write_csv('submission.csv',test_predictions)

"""