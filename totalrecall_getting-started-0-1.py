# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau
sns.set_style('whitegrid')
%matplotlib inline

from sklearn import linear_model


## import the data 
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")

##print(train.shape)
##print(train.head(n=5))
##print(test.shape)
##print(test.head(n=5))
##print(sample_submission.head())
#### looking at the country distribution through a couple of variables 
#fig, (axis1) = plt.subplots(1,1,figsize=(15,5))
#sns.countplot(x='Product_Info_2', data = train, hue = "Response", palette="husl", ax=axis1)

##train.head()
##train['Family_Hist_5'].round(1)
#test = train.iloc[:, [33, 34, 35, 36, 37, 38, 127]]
#test['Family_Hist_2_rounded'] = test['Family_Hist_2'].round(1)
#test['Family_Hist_3_rounded'] = test['Family_Hist_3'].round(1)
#test['Family_Hist_4_rounded'] = test['Family_Hist_4'].round(1)
#test['Family_Hist_5_rounded'] = test['Family_Hist_5'].round(1)

#### looking at the country distribution through a couple of variables 
#fig, (axis1, axis2, axis3, axis4, axis5) = plt.subplots(5, 1,figsize=(15,15))
#sns.countplot(x='Family_Hist_1', data = test, hue = "Response", palette="husl", ax=axis1)
#sns.countplot(x='Family_Hist_2_rounded', data = test, hue = "Response", palette="husl", ax=axis2)
#sns.countplot(x='Family_Hist_3_rounded', data = test, hue = "Response", palette="husl", ax=axis3)
#sns.countplot(x='Family_Hist_4_rounded', data = test, hue = "Response", palette="husl", ax=axis4)
#sns.countplot(x='Family_Hist_5_rounded', data = test, hue = "Response", palette="husl", ax=axis5)

train2 = train
train2['med_keyword_bucket_1'] = train2['Medical_Keyword_38'] + train2['Medical_Keyword_43'] + train2['Medical_Keyword_45']
train2['med_keyword_bucket_2'] = train2['Medical_Keyword_3']
train2['med_keyword_bucket_3'] = train2['Medical_Keyword_20']
train2['med_keyword_bucket_4'] = 0
train2['med_keyword_bucket_5'] = 0
train2['med_keyword_bucket_6'] = 0
train2['med_keyword_bucket_7'] = 0
train2['med_keyword_bucket_8'] = train2['Medical_Keyword_39'] + train2['Medical_Keyword_6'] + train2['Medical_Keyword_8'] + train2['Medical_Keyword_11'] + train2['Medical_Keyword_17'] + train2['Medical_Keyword_30']

test['med_keyword_bucket_1'] = test['Medical_Keyword_38'] + test['Medical_Keyword_43'] + test['Medical_Keyword_45']
test['med_keyword_bucket_2'] = test['Medical_Keyword_3']
test['med_keyword_bucket_3'] = test['Medical_Keyword_20']
test['med_keyword_bucket_4'] = 0
test['med_keyword_bucket_5'] = 0
test['med_keyword_bucket_6'] = 0
test['med_keyword_bucket_7'] = 0
test['med_keyword_bucket_8'] = test['Medical_Keyword_39'] + test['Medical_Keyword_6'] + test['Medical_Keyword_8'] + test['Medical_Keyword_11'] + test['Medical_Keyword_17'] + test['Medical_Keyword_30']

fig, (axis1, axis2) = plt.subplots(2, 1,figsize=(15,55))
sns.countplot(x='Insurance_History_1', data = train2[train2['Insurance_History_1'] > 0], hue = "Response", palette="husl", ax=axis1)
sns.countplot(x='Insurance_History_2', data = train2[train2['Insurance_History_2'] > 0], hue = "Response", palette="husl", ax=axis2)


###### look at scatter plots for some variables
#fig, (axis1, axis2, axis3, axis4, axis5) = plt.subplots(5, 1,figsize=(15,25))
#sns.regplot(x="Family_Hist_1", y="Response", x_jitter=.1, y_jitter = .1, data=test, ax= axis1)
#sns.regplot(x="Family_Hist_2_rounded", y="Response", x_jitter=.1, y_jitter = .1, data=test, ax= axis2)
#sns.regplot(x="Family_Hist_3_rounded", y="Response", x_jitter=.1, y_jitter = .1, data=test, ax= axis3)
#sns.regplot(x="Family_Hist_4_rounded", y="Response", x_jitter=.1, y_jitter = .1, data=test, ax= axis4)
#sns.regplot(x="Family_Hist_5_rounded", y="Response", x_jitter=.1, y_jitter = .1, data=test, ax= axis5)




#### before we do our next modeling stuff, we need to convert some variables

#helper = train2.iloc[:, [127, 128, 129, 130, 131, 132, 133, 134, 135, 8, 9, 10, 11]]
#train2.head()
#helper.head()

### this is a useful (and interesting) function..
def do_treatment(df):
    for col in df:
        if df[col].dtype == np.dtype('O'):
            df[col] = df[col].apply(lambda x : hash(str(x)))
            
    df.fillna(-1, inplace = True)
    
    
do_treatment(train2)
do_treatment(test)


####### guess #3 --- 
X_train = train2.drop([['Response', 'Id']], axis=1)
y_train = train2['Response']
X_test = test ##test[['med_keyword_bucket_1', 'med_keyword_bucket_2', 'med_keyword_bucket_3', 'med_keyword_bucket_4', 'med_keyword_bucket_5', 'med_keyword_bucket_6' ,'med_keyword_bucket_7', 'med_keyword_bucket_8', 'Ins_Age', 'Ht', 'Wt', 'BMI']]

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=200, max_features = 'sqrt',
                             max_depth = None, verbose = 1, n_jobs = -1)
clf.fit(X_train, y_train)
clf_probs = clf.predict_proba(X_test) ## prob of being in a certain class
test_prediction = pd.DataFrame(clf.predict(X_test)) ## predicted class
##### get the output ready for a csv submission
output = pd.DataFrame(test.Id).join(pd.DataFrame(test_prediction))
output.columns = ['Id', 'Response']

output.to_csv('sub3.csv', index = False, header = ['Id', 'Response'])


##### checking the variable importance 
importances = clf.feature_importances_
#importances

for i in range(0, len(list(X_train.columns.values))):
    print(list(X_train.columns.values)[i]), print(importances[i])


#i in list(X_train.columns.values):
#    print(i)

