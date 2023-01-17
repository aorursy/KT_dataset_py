import numpy as np

import pandas as pd

from sklearn import linear_model



dataset=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

train_Y=dataset.iloc[:,0]

train_X=dataset.iloc[:,1:]



test_X=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")





model=linear_model.LogisticRegression()

model.fit(train_X,train_Y)



rs=model.predict(test_X)

sm=pd.DataFrame({'ImageId':range(1,len(rs)+1),"Label":rs})

sm.to_csv('mine_submission.csv',index=False)