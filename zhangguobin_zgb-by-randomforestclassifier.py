import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
train = pd.read_csv('train1.csv')
test = pd.read_csv('test1.csv')
X = train.iloc[:,0:9]
y = train.iloc[:,9]
rand = RandomForestClassifier(min_samples_split=2,n_estimators=30,max_depth=100,random_state=820)
rand.fit(X,y)
dat = rand.predict(test).reshape(43,1)
rang = np.array(range(1,44)).reshape(43,1)
target = np.concatenate([rang,dat],axis = 1)
pd_data = pd.DataFrame(target,columns=['Id','label'],index=None)
pd_data.to_csv('data_rand3.csv',index=False)