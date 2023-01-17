import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv("/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv")
df.head(),df.shape
df.isnull().sum()
df.set_index('PassengerId')
from sklearn.preprocessing import LabelEncoder



label_encoder = LabelEncoder()



df['Sex'] = label_encoder.fit_transform(df['Sex']) #Male is 1

df['Category'] = label_encoder.fit_transform(df['Category']) #P is 1
def age_group(Age):

  a=''

  if (Age<=1):

    a='toddler'



  elif(Age<=4):

    a='infant'



  elif(Age<=12):

    a='child'



  elif(Age<=19):

    a='teenager'

    

  elif(Age<=55):

    a='adult'



  else:

    a='senior'

    

  return a



df['age_group']=df.Age.map(age_group)
import matplotlib as plt 

import seaborn as sns
sns.heatmap(data = df.corr(), annot = True)
df['Sex'].value_counts()
sns.barplot(x='Sex',y='Survived', data =df)
df['Category'].value_counts()
sns.barplot(x='Category',y='Survived', data =df)
sns.kdeplot(df.loc[(df['Survived']==0),'Age'],shade = True ,Label='Not Survived')

sns.kdeplot(df.loc[(df['Survived']==1),'Age'],shade = True ,Label='Survived')


X =df.loc[:,["Sex","Age","Category"]]

y =df.loc[:,["Survived"]]
from sklearn.model_selection import train_test_split

train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.2, random_state=32, stratify=y)
from sklearn.model_selection import RandomizedSearchCV



n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]



max_features = ['auto', 'sqrt']



max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)



min_samples_split = [2, 5, 10]



min_samples_leaf = [1, 2, 4]



bootstrap = [True, False]



random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()





rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(train_X, train_y)
rf_random.best_params_
rf = RandomForestClassifier()



rf = RandomForestClassifier(n_estimators= 800,

 min_samples_split= 10,

 min_samples_leaf = 4,

 max_features= "sqrt",

 max_depth = 50,

 bootstrap = True)

                            

rf.fit(train_X, train_y)



preds2  = rf.predict(test_X)
from sklearn.metrics import accuracy_score, roc_auc_score

print("Accuracy: ",round((accuracy_score(test_y,preds2)*100),4),"%")
from sklearn.metrics import confusion_matrix, classification_report

conf_mat = confusion_matrix(test_y, preds2)
sns.heatmap(conf_mat, annot=True, fmt='g') #plotting confusion matrix
df_copy = df.copy()
df_copy['age_group'] = label_encoder.fit_transform(df['age_group']) 
X2 =df_copy.loc[:,["Sex","age_group","Category"]]

y2 =df_copy.loc[:,["Survived"]]
from sklearn.model_selection import train_test_split

train_X2,test_X2,train_y2,test_y2 = train_test_split(X2,y2, test_size=0.2, random_state=32, stratify=y)
rf = RandomForestClassifier(random_state = 2)

rf.fit(train_X2, train_y2)



preds3 = rf.predict(test_X2)
from sklearn.metrics import accuracy_score

print("Accuracy: ",round((accuracy_score(test_y2,preds3)*100),4),"%")

conf_mat2 = confusion_matrix(test_y2, preds3)
sns.heatmap(conf_mat2, annot=True, fmt='g') 