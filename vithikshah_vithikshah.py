import os
print((os.listdir('../input/')))
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
df_train = pd.read_csv('../input/web-club-recruitment-2018/train.csv')
df_test = pd.read_csv('../input/web-club-recruitment-2018/test.csv')
feature_cols=['X1','X2','X3','X5','X6','X7','X8','X9','X10','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23']
train_X = df_train[feature_cols]
train_y = df_train.loc[:, 'Y']
df_test = df_test[feature_cols]



rf = RandomForestClassifier(n_estimators=200,max_features='auto',max_depth=23)
rf.fit(train_X, train_y)

pred = rf.predict_proba(df_test)
result = pd.DataFrame(pred[:,1])
result.index.name = 'id'
result.columns = ['predicted_val']
result.to_csv('output.csv', index=True)
