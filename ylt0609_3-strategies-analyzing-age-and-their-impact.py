# loading package
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
%matplotlib inline
sns.set()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
# loading data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_data = df_train.append(df_test)
df_data['Has_Age'] = df_data['Age'].isnull().map(lambda x : 0 if x == True else 1)
fig = plt.figure(figsize=(11,6))
ax = sns.countplot(df_data['Pclass'],hue=df_data['Has_Age'])
ax.set_title('Passenger has age',fontsize = 20)
# extracted title using name
df_data['Title'] = df_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df_data['Title'] = df_data['Title'].replace(['Capt', 'Col', 'Countess', 'Don',
                                               'Dr', 'Dona', 'Jonkheer', 
                                                'Major','Rev','Sir'],'Rare') 
df_data['Title'] = df_data['Title'].replace(['Mlle', 'Ms','Mme'],'Miss')
df_data['Title'] = df_data['Title'].replace(['Lady'],'Mrs')
g = sns.factorplot(y='Age',x='Title',kind='box',hue='Pclass', data=df_data, 
               size=6,aspect=1.5)
plt.subplots_adjust(top=0.93)
g.fig.suptitle('Age vs Pclass cross Title', fontsize = 20)
missing_mask = (df_data['Has_Age'] == 0)
pd.crosstab(df_data[missing_mask]['Pclass'],df_data[missing_mask]['Title'])
# filling Missing age with Pclass & Title
df_data['Title'] = df_data['Title'].map({"Mr":0, "Rare" : 1, "Master" : 2, "Miss" : 3, "Mrs" : 4 })
Pclass_title_pred = df_data.pivot_table(values='Age', index=['Pclass'], columns=['Title'],aggfunc=np.median).values
df_data['P_Ti_Age'] = df_data['Age']
for i in range(0,5):
    # 0,1,2,3,4
    for j in range(1,4):
        # 1,2,3
            df_data.loc[(df_data.Age.isnull()) & (df_data.Pclass == j) & (df_data.Title == i),'P_Ti_Age'] = Pclass_title_pred[j-1, i]
df_data['P_Ti_Age'] = df_data['P_Ti_Age'].astype('int')

# filling Missing age with Title only

Ti_pred = df_data.groupby('Title')['Age'].median().values
df_data['Ti_Age'] = df_data['Age']
for i in range(0,5):
 # 0 1 2 3 4
    df_data.loc[(df_data.Age.isnull()) & (df_data.Title == i),'Ti_Age'] = Ti_pred[i]
df_data['Ti_Age'] = df_data['Ti_Age'].astype('int')


df_data['P_Ti_AgeBin'] = pd.qcut(df_data['P_Ti_Age'], 4)
df_data['Ti_AgeBin'] = pd.qcut(df_data['Ti_Age'], 4)
label = LabelEncoder()
df_data['P_Ti_Code'] = label.fit_transform(df_data['P_Ti_AgeBin'])
df_data['Ti_Code'] = label.fit_transform(df_data['Ti_AgeBin'])
# convet sex on to 0 1
df_data['Sex'] = df_data['Sex'].map( {'female' : 1, 'male' : 0}).astype('int')
# separate feature form whole dataset
df_train[['Sex','P_Ti_Code','Ti_Code']] = df_data[['Sex','P_Ti_Code','Ti_Code']][:len(df_train)]
df_test[['Sex','P_Ti_Code','Ti_Code']] = df_data[['Sex','P_Ti_Code','Ti_Code']][len(df_train):]
X = df_train.drop(labels=['Survived','PassengerId'],axis=1)
Y = df_train['Survived']
P_Ti_vs_Ti = ['Sex', 'Pclass', 'P_Ti_Code', 'Ti_Code']
X[P_Ti_vs_Ti].head()
selector = RFECV(RandomForestClassifier(n_estimators=250,min_samples_split=20),cv=10,n_jobs=-1)
selector.fit(X[P_Ti_vs_Ti], Y)
print(selector.support_)
print(selector.ranking_)
print(selector.grid_scores_*100)
seeds = 10
P_Ti_vs_Ti_diff = np.zeros(seeds)
for i in range(seeds):
    diff_cv = StratifiedKFold(n_splits=10,shuffle=True,random_state=i)
    selector = RFECV(RandomForestClassifier(random_state=i,n_estimators=250,min_samples_split=20),cv=diff_cv,n_jobs=-1)
    selector.fit(X[P_Ti_vs_Ti], Y)
    P_Ti_vs_Ti_diff[i] = ( selector.grid_scores_[2] - selector.grid_scores_[3] )*100

fig = plt.figure(figsize=(18,8))
ax  = plt.gca()
difference_plot = ax.bar(range(seeds),P_Ti_vs_Ti_diff,color='g')
plt.xlabel(" Seed # ",fontsize = "14")
ax.set_title("P_Ti vs Ti",fontsize=20)
plt.ylabel("score difference  % ", fontsize="14")
mean = P_Ti_vs_Ti_diff.mean()
ax.axhline(mean)
Minor_mask = (df_train.Age <= 14) 
Not_Minor_mask = (df_train.Age > 14) 
display(df_train[Minor_mask][['Pclass','Survived']].groupby(['Pclass']).mean())
display(df_train[Not_Minor_mask][['Pclass','Survived']].groupby(['Pclass']).mean())
df_data['Age_copy'] = df_data['Age'].fillna(-1)
df_data['Minor'] = (df_data['Age_copy'] < 14.0) & (df_data['Age_copy']>= 0)
df_data['Minor'] = df_data['Minor'] * 1
# We could capture more 8 Master in Pclass = 3 by filling missing age 
df_data['P_Ti_Minor'] = ((df_data['P_Ti_Age']) < 14.0) * 1
(df_data['P_Ti_Minor'] - df_data['Minor']).sum()
# separate feature form whole dataset
df_train[['P_Ti_Minor', 'Minor']] = df_data[['P_Ti_Minor', 'Minor']][:len(df_train)]
df_test[['P_Ti_Minor' , 'Minor']] = df_data[['P_Ti_Minor', 'Minor']][len(df_train):]
X = df_train.drop(labels=['Survived','PassengerId'],axis=1)
Y = df_train['Survived']
P_Ti = ['Sex', 'Pclass', 'P_Ti_Code', 'P_Ti_Minor']
Discard = ['Sex','Pclass','Minor']
display(X[P_Ti].head())
display(X[Discard].head())
selector_P_Ti = RFECV(RandomForestClassifier(n_estimators=250,min_samples_split=20),cv=10,n_jobs=-1)
selector_P_Ti.fit(X[P_Ti], Y)
selector_Discard = RFECV(RandomForestClassifier(n_estimators=250,min_samples_split=20),cv=10,n_jobs=-1)
selector_Discard.fit(X[Discard], Y)
print(selector_P_Ti.support_)
print(selector_P_Ti.ranking_)
print(selector_P_Ti.grid_scores_*100)
print(selector_Discard.support_)
print(selector_Discard.ranking_)
print(selector_Discard.grid_scores_*100)
seeds = 10
P_Ti_vs_Discard = np.zeros(seeds)
for i in range(seeds):
    diff_cv = StratifiedKFold(n_splits=10,shuffle=True,random_state=i)
    selector_P_Ti = RFECV(RandomForestClassifier(random_state=i,n_estimators=250,min_samples_split=20),cv=diff_cv,n_jobs=-1)
    selector_P_Ti.fit(X[P_Ti], Y)
    selector__Discard = RFECV(RandomForestClassifier(random_state=i,n_estimators=250,min_samples_split=20),cv=diff_cv,n_jobs=-1)
    selector__Discard.fit(X[Discard], Y)
    # print(selector_P_Ti.grid_scores_[3],selector__Discard.grid_scores_[3])
    P_Ti_vs_Discard[i] = ( selector_P_Ti.grid_scores_[3] - selector__Discard.grid_scores_[2] )*100
    # print(P_Ti_vs_Discard)

fig = plt.figure(figsize=(18,8))
ax  = plt.gca()
difference_plot = ax.bar(range(seeds),P_Ti_vs_Discard,color='g')
plt.xlabel(" Seed # ",fontsize = "14")
ax.set_title("P_Ti vs Discard",fontsize=20)
plt.ylabel("score difference  % ", fontsize="14")
mean = P_Ti_vs_Discard.mean()
ax.axhline(mean)
Base = ['Sex','Pclass']
Base_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
Base_Model.fit(X[Base], Y)
print('Base oob score :%.5f' %(Base_Model.oob_score_),'   LB_Public : 0.76555')
Ti = ['Sex', 'Pclass', 'Ti_Code']
# Note that even we change the split = 40 , 50 , 70 seems to underfit, the same LB_Public 0.74162 they did.
P_Ti_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
P_Ti_Model.fit(X[P_Ti], Y)
# spilit = 20 , 40 LB = 0.77511
Discard_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
Discard_Model.fit(X[Discard], Y)
# split = 10, 20, 40 --> LB = 0.74162, spilits = 50 , underfit , LB = 0.732
Ti_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=10,oob_score=True)
Ti_Model.fit(X[Ti], Y)
print('P_Ti oob score :%.5f' %(P_Ti_Model.oob_score_),'   LB_Public : 0.74162')
print('Discard oob score :%.5f '%(Discard_Model.oob_score_),' LB_Public : 0.77511')
print('Ti oob score : %.5f' %(Ti_Model.oob_score_), '  LB_Public : 0.74162')