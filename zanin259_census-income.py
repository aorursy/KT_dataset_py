import warnings

warnings.filterwarnings('ignore')
!pip install feature-engine
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier

from feature_engine.discretisers import EqualWidthDiscretiser

from feature_engine.categorical_encoders import RareLabelCategoricalEncoder

from feature_engine.categorical_encoders import OneHotCategoricalEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn.model_selection import GridSearchCV
train=pd.read_csv('/kaggle/input/adult-census-income/adult.csv')
train.head()
train.info()
train.describe()
# We plot to see the capital gains of people with income greater 50k and also less than 50k



plt.figure(figsize=(12,4))

sns.barplot(x='education',hue='income',y='capital.gain',data=train)

plt.xticks(rotation=45)
# We similarly plot to see the capital losses of people with income greater than 50k and also less than 50k



plt.figure(figsize=(12,4))

sns.barplot(x='education',hue='income',y='capital.loss',data=train)

plt.xticks(rotation=45)
train.head()
# We plot to see if race has effect on the income of the person.

plt.figure(figsize=(12,4))

sns.countplot(x='race',hue='income',data=train)
# We plot to check if the sex has an effect on the income of a person

plt.figure(figsize=(12,4))

sns.countplot(x='sex',hue='income',data=train)
# We plot to find if education has an effect on the income of a person



plt.figure(figsize=(12,4))

sns.countplot(x='education',hue='income',data=train)

plt.xticks(rotation=45)
# We plot to see if being in a relationship has an effect on income

plt.figure(figsize=(12,4))

sns.countplot(x='relationship',hue='income',data=train)

plt.xticks(rotation=45)
# We plot to the workclass to see if it has any effect on income

plt.figure(figsize=(12,4))

sns.countplot(x='workclass',hue='income',data=train)

plt.xticks(rotation=45)
plt.figure(figsize=(12,4))

sns.countplot(x='occupation',hue='income',data=train)

plt.xticks(rotation=45)
X=train.drop(columns='income')



y=train['income']
# We change the <=50K to 1 and >50K to 0 for ease of classification

y=y.map({'<=50K':0,'>50K':1})
# We split the dataset into train and test dataset



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train['workclass'].value_counts()
# We also the ? with the type as missing

X_train['workclass']=X_train['workclass'].replace({'Without-pay':'Never-worked','?':'Missing'}) # We replace the without pay and to never worked as they both imply the same thing.



X_test['workclass']=X_test['workclass'].replace({'Without-pay':'Never-worked','?':'Missing'}) 
# We change the ? to indicate it as missing

X_train['occupation']=X_train['occupation'].replace({'?':'Missing'})



X_test['occupation']=X_test['occupation'].replace({'?':'Missing'})
# We bin the age and hours per week columns



ewd=EqualWidthDiscretiser(bins=10,variables=['age','hours.per.week'])



ewd.fit(X_train)



X_train=ewd.transform(X_train)



X_test=ewd.transform(X_test)
X_train=X_train.drop(columns=['fnlwgt','education'])



X_test=X_test.drop(columns=['fnlwgt','education'])
# We create a list of categorical variables

qualitative=list(X_train.select_dtypes(include='object'))



qualitative
# Inorder to check the sparsity of the categorical variables we plot them on a graph.

count=1

plt.figure(figsize=(20,10))

for col in qualitative:

    temp_df = pd.Series(X_train[col].value_counts() / len(X_train) )



    # make plot with the above percentages

    plt.subplot(3,3, count)

    fig = temp_df.sort_values(ascending=False).plot.bar()

    fig.set_xlabel(col)

    

    # add a line at 4 % to flag the threshold for rare categories

    fig.axhline(y=0.04, color='red')

    fig.set_ylabel('Percentage Count')

    count +=1
# We impute rare labels to the native country column

rce=RareLabelCategoricalEncoder(tol=0.04,n_categories=2,variables='native.country')



rce.fit(X_train)



X_train=rce.transform(X_train)



X_test=rce.transform(X_test)
# We use the heatmap to find the most important features



data_com=pd.concat([X_train,y_train],axis=1)



data_com=data_com.corr()



plt.figure(figsize=(8,8))



sns.heatmap(data_com,annot=True)
# We use one hot categorical encoder on the dataset



oce=OneHotCategoricalEncoder(drop_last=True)



oce.fit(X_train)



X_train=oce.transform(X_train)



X_test=oce.transform(X_test)
# We standard scale on the dataset



train_cols=X_train.columns



sc=StandardScaler()



sc.fit(X_train)



X_train=pd.DataFrame(sc.transform(X_train),columns=train_cols)



X_test=pd.DataFrame(sc.transform(X_test),columns=train_cols)
# We perform the random forest classifier on the dataset



classifier_rf=RandomForestClassifier(random_state=0)



classifier_rf.fit(X_train,y_train)



y_pred_rf=classifier_rf.predict(X_test)
# We find the classification report for the model



class_report=classification_report(y_pred_rf,y_test)



print('The classification report is \n{}'.format(class_report))
# We perform the gradient boost classifier on the dataset



classifier_gb=GradientBoostingClassifier(random_state=0)



classifier_gb.fit(X_train,y_train)



y_pred_gb=classifier_gb.predict(X_test)
# We find the classification report for the model



class_report=classification_report(y_pred_gb,y_test)



print('The classification report is \n{}'.format(class_report))
# We perform the Extra Tree classifier on the dataset



classifier_ec=ExtraTreesClassifier(random_state=0)



classifier_ec.fit(X_train,y_train)



y_pred_ec=classifier_ec.predict(X_test)
# We find the classification report for the model



class_report=classification_report(y_pred_ec,y_test)



print('The classification report is \n{}'.format(class_report))
# We use grid search to find the best parameters for the gradient boosting model



params=[{'learning_rate':[0.1,0.001,0.001],'n_estimators':[100,300,500,700,1000]}]



gsc=GridSearchCV(estimator=classifier_gb,param_grid=params,scoring='accuracy',n_jobs=-1,cv=5)
gsc.fit(X_train,y_train)
gsc.best_params_
# We perform the gradient boost classifier on the dataset after finding the best parameters



classifier_gb1=GradientBoostingClassifier(n_estimators=500,random_state=0)



classifier_gb1.fit(X_train,y_train)



y_pred_gb1=classifier_gb1.predict(X_test)
# We find the classification report for the model



class_report=classification_report(y_pred_gb1,y_test)



print('The classification report is \n{}'.format(class_report))