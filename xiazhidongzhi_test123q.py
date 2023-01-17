import pandas as pd 

input_df = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")



import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
TResp = input_df['Response']
test = input_df.drop('Response',axis=1)
test = test.drop('Id',axis=1)
test=test.fillna(value=-1)
TRespV = TResp.values
test=pd.get_dummies(test,prefix=['Product_Info_2'])
testV = test.values
lst=[input_df,TResp,test]

dtrain = xgb.DMatrix( testV, label=TRespV)
param = { 'objective':'multi:softmax' ,'num_class':9}
num_round = 10
bst = xgb.train( param , dtrain, num_round )


input_df = pd.read_csv("../input/test.csv")
Aid=input_df['Id']
test = input_df.drop('Id',axis=1)
test=test.fillna(value=-1)
test=pd.get_dummies(test,prefix=['Product_Info_2'])
testV = test.values
testV=xgb.DMatrix(testV)
pred=bst.predict(testV)



 
import pandas as pd 

input_df = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")



import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
TResp = input_df['Response']
test = input_df.drop('Response',axis=1)
test = test.drop('Id',axis=1)
test=test.fillna(value=-1)
TRespV = TResp.values
test=pd.get_dummies(test,prefix=['Product_Info_2'])
testV = test.values
lst=[input_df,TResp,test]

dtrain = xgb.DMatrix( testV, label=TRespV)
param = { 'objective':'multi:softmax' ,'num_class':9}
num_round = 10
bst = xgb.train( param , dtrain, num_round )


input_df = pd.read_csv("../input/test.csv")
Aid=input_df['Id']
test = input_df.drop('Id',axis=1)
test=test.fillna(value=-1)
test=pd.get_dummies(test,prefix=['Product_Info_2'])
testV = test.values
testV=xgb.DMatrix(testV)
pred=bst.predict(testV)




 
 
submission = pd.DataFrame({ 'Id': Aid,'Response': pred })
submission.to_csv("submission.csv", index=False)

 
