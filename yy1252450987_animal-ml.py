import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
test_ID = test_df['ID']
train_OutcomeType = train_df['OutcomeType']
train_df = train_df.drop(['AnimalID','OutcomeSubtype','OutcomeType'] , axis=1)
test_df = test_df.drop('ID', axis=1)

combine_df = pd.concat([train_df, test_df], axis=0)
train_df.info()
test_df.info()
combine_df.info()
# Name
combine_df['HasName']=combine_df['Name'].notnull().astype(int)
# DateTime
interval = pd.to_datetime(combine_df['DateTime'])-pd.to_datetime('2013-01-01')
combine_df['DateTime_day'] = interval /np.timedelta64(1, 'D')
interval = ((interval/np.timedelta64(1, 'D'))%1*24)
interval[(interval>=0) & (interval)<6] = 0 #'midnight'
interval[(interval>=6) & (interval<12)] = 1 #'morning'
interval[(interval>=12) & (interval<18)] = 2 #'afternoon'
interval[(interval>=18) & (interval<22)] = 3 #'evening'
interval[(interval>=22) & (interval<24)] =0 # 'midnight'
combine_df['DateTime_mmae'] = interval
# AnimalType
combine_df['AnimalType'] = combine_df['AnimalType'].map({'Dog':0, 'Cat':1})
# SexuponOutcome
combine_df['SexuponOutcome'] = combine_df['SexuponOutcome'].fillna(combine_df['SexuponOutcome'].mode()[0])
combine_df['Sex'] = combine_df['SexuponOutcome'].map({'Intact Female':0,'Spayed Female':0,'Intact Male':1,'Neutered Male':1,'Unknown':2})
combine_df['IsIntact'] = combine_df['SexuponOutcome'].map({'Intact Female':0,'Intact Male':0, 'Neutered Male':1,'Spayed Female':1,'Unknown':2})
combine_df[['SexuponOutcome','Sex','IsIntact']].head()
# AgeuponOutcome
combine_df['AgeuponOutcome'] = combine_df['AgeuponOutcome'].fillna(combine_df['AgeuponOutcome'].mode()[0])
def convert(x):
    a = str(x).split(' ')[0]
    b = str(x).split(' ')[-1]
    if 'year' in str(b):
        return int(a)*12*52*7
    elif 'month' in str(b):
        return int(a)*4*7
    elif 'week' in str(b):
        return int(a)*7
    else:
        return int(a)

combine_df['AgeuponOutcome'] = combine_df['AgeuponOutcome'].map(convert)
#Breed
combine_df['IsMix'] = combine_df['Breed'].map(lambda x: 'Mix' in x).astype(int)
combine_df['Breed'] = combine_df['Breed'].map(lambda x:x.strip(' Mix').split('/')[0].split(' ')[-1])
# Color 
combine_df['IsPure'] = combine_df['Color'].map(lambda x: '/' not in x).astype(int)
combine_df[['Color','IsPure']].head()
combine_df['Color'] = combine_df['Color'].map(lambda x: x.split('/')[0].split(' ')[0])
combine_df_DateTime_mmae = pd.get_dummies(combine_df['DateTime_mmae'], prefix='DateTime_mmae')
combine_df_breed = pd.get_dummies(combine_df['Breed'], prefix='Breed')
combine_df_color = pd.get_dummies(combine_df['Color'], prefix='Color')
combine_df['DateTime_day'] = (combine_df['DateTime_day'] - combine_df['DateTime_day'].min())/(combine_df['DateTime_day'].max() - combine_df['DateTime_day'].min())
combine_df['AgeuponOutcome'] = (combine_df['AgeuponOutcome'] - combine_df['AgeuponOutcome'].min())/(combine_df['AgeuponOutcome'].max() - combine_df['AgeuponOutcome'].min())
X = combine_df[['HasName','DateTime_day', 'AnimalType', 'AgeuponOutcome','Sex','IsIntact','IsMix','IsPure']]
X = pd.concat([X, combine_df_breed, combine_df_color, combine_df_DateTime_mmae], axis=1)
train_x = X.iloc[0:train_df.shape[0],]
test_x = X.iloc[train_df.shape[0]:,]
train_y = train_OutcomeType.map({'Adoption':0,'Died':1,'Euthanasia':2,'Return_to_owner':3,'Transfer':4})
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
print(train_x.shape, test_x.shape)
from sklearn.cross_validation import train_test_split
tr_x, va_x, tr_y, va_y = train_test_split(train_x, train_y,test_size=0.3,random_state=0)
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf1 = KNeighborsClassifier(3)
clf1.fit(tr_x, tr_y)
name = clf1.__class__.__name__
print(name),
va_pred = clf1.predict(va_x)
acc = accuracy_score(va_y, va_pred)
print("Accuracy: {:.4%}".format(acc)),
    
va_pred = clf1.predict_proba(va_x)
ll = log_loss(va_y, va_pred)
print("Log Loss: {:.4f}".format(ll))
clf2 = SVC(kernel="rbf", C=0.025, probability=True)
clf2.fit(tr_x, tr_y)
name = clf2.__class__.__name__
print(name),
va_pred = clf2.predict(va_x)
acc = accuracy_score(va_y, va_pred)
print("Accuracy: {:.4%}".format(acc)),
va_pred = clf2.predict_proba(va_x)
ll = log_loss(va_y, va_pred)
print("Log Loss: {:.4f}".format(ll))
clf3 = DecisionTreeClassifier()
clf3.fit(tr_x, tr_y)
name = clf3.__class__.__name__
print(name),
va_pred = clf3.predict(va_x)
acc = accuracy_score(va_y, va_pred)
print("Accuracy: {:.4%}".format(acc)),
    
va_pred = clf3.predict_proba(va_x)
ll = log_loss(va_y, va_pred)
print("Log Loss: {:.4f}".format(ll))
clf4 = RandomForestClassifier()
clf4.fit(tr_x, tr_y)
name = clf4.__class__.__name__
print(name),
va_pred = clf4.predict(va_x)
acc = accuracy_score(va_y, va_pred)
print("Accuracy: {:.4%}".format(acc)),
    
va_pred = clf4.predict_proba(va_x)
ll = log_loss(va_y, va_pred)
print("Log Loss: {:.4f}".format(ll))
clf5 = AdaBoostClassifier()
clf5.fit(tr_x, tr_y)
name = clf5.__class__.__name__
print(name),
va_pred = clf5.predict(va_x)
acc = accuracy_score(va_y, va_pred)
print("Accuracy: {:.4%}".format(acc)),
    
va_pred = clf5.predict_proba(va_x)
ll = log_loss(va_y, va_pred)
print("Log Loss: {:.4f}".format(ll))
clf6 = GradientBoostingClassifier()
clf6.fit(tr_x, tr_y)
name = clf6.__class__.__name__
print(name),
va_pred = clf6.predict(va_x)
acc = accuracy_score(va_y, va_pred)
print("Accuracy: {:.4%}".format(acc)),
    
va_pred = clf6.predict_proba(va_x)
ll = log_loss(va_y, va_pred)
print("Log Loss: {:.4f}".format(ll))
clf7 = GaussianNB()
clf7.fit(tr_x, tr_y)
name = clf7.__class__.__name__
print(name),
va_pred = clf7.predict(va_x)
acc = accuracy_score(va_y, va_pred)
print("Accuracy: {:.4%}".format(acc)),
    
va_pred = clf7.predict_proba(va_x)
ll = log_loss(va_y, va_pred)
print("Log Loss: {:.4f}".format(ll))
clf8 = MultinomialNB()
clf8.fit(tr_x, tr_y)
name = clf8.__class__.__name__
print(name),
va_pred = clf8.predict(va_x)
acc = accuracy_score(va_y, va_pred)
print("Accuracy: {:.4%}".format(acc)),
    
va_pred = clf8.predict_proba(va_x)
ll = log_loss(va_y, va_pred)
print("Log Loss: {:.4f}".format(ll))
clf9 = BernoulliNB()
clf9.fit(tr_x, tr_y)
name = clf9.__class__.__name__
print(name),
va_pred = clf9.predict(va_x)
acc = accuracy_score(va_y, va_pred)
print("Accuracy: {:.4%}".format(acc)),
    
va_pred = clf9.predict_proba(va_x)
ll = log_loss(va_y, va_pred)
print("Log Loss: {:.4f}".format(ll))
clf10 = LinearDiscriminantAnalysis()
clf10.fit(tr_x, tr_y)
name = clf10.__class__.__name__
print(name),
va_pred = clf10.predict(va_x)
acc = accuracy_score(va_y, va_pred)
print("Accuracy: {:.4%}".format(acc)),
    
va_pred = clf10.predict_proba(va_x)
ll = log_loss(va_y, va_pred)
print("Log Loss: {:.4f}".format(ll))
# prediction 
model = GradientBoostingClassifier()
model.fit(train_x, train_y)
test_prediction = model.predict_proba(test_x)
LABELS = sorted(train_OutcomeType.unique())
PredictResult = pd.DataFrame(test_prediction,columns=LABELS)
PredictResult.columns.names = ['ID']
PredictResult.index.names = ['ID']
PredictResult.index += 1
PredictResult.to_csv('GBC_pred_1.csv', index_label='ID')