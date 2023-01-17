import dask.dataframe as ddf
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

train=ddf.read_csv("../input/train.csv")
test=ddf.read_csv("../input/test.csv")
sample=ddf.read_csv('../input/sample_submission.csv')
sample=sample.compute()
train=train.compute()
test=test.compute()
train1,validate1=train_test_split(train,test_size=0.3,random_state=100)
train_x=train1.drop('label',axis=1)
train_y=train1.label
validate_x=validate1.drop('label',axis=1)
validate_y=validate1.label
acc_df=pd.DataFrame(columns=['DecisionTree','RandomForest','AdaBoost'],index=['Accuracy'])
'''Decision Tree'''
model=DecisionTreeClassifier(random_state=100,max_depth=2)
model.fit(train_x,train_y)
pred=model.predict(validate_x)
acc_df.DecisionTree.Accuracy=accuracy_score(validate_y,pred)
'''Random Forest'''
model_rf=RandomForestClassifier(random_state=100,n_estimators=300)
model_rf.fit(train_x,train_y)
pred=model_rf.predict(validate_x)
acc_df.RandomForest.Accuracy=accuracy_score(validate_y,pred)
'''AdaBoost'''
model_ab=AdaBoostClassifier(random_state=100)
model_ab.fit(train_x,train_y)
pred_ab=model_ab.predict(validate_x)
acc_df.AdaBoost.Accuracy=accuracy_score(validate_y,pred_ab)
print(acc_df)
'''Random Forest has the highest accuracy with respect to the predictions so we stick on to
Random  Forest and lets start predicting for the test dataset'''

pred=model_rf.predict(test)
final_pred=pd.DataFrame({'ImageId':sample.ImageId,'Label':pred})
'''Exporting to a csv'''
final_pred.index = np.arange(1, len(final_pred)+1)
final_pred.to_csv('final_pred.csv',index=False)