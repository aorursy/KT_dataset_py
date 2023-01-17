import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



import seaborn as sns

sns.set()



import sklearn

import lightgbm as lgb

from sklearn.model_selection import cross_val_score
df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')
df_train.head()
df_test.head()
df_train.info()

print 

df_test.info()
df_train.describe()
for col in df_train.columns:

    print(col + ' has ' + str(df_train[col].nunique()) + ' unique values.')
df_test.describe()
df_train.describe(include='object')
df_test.describe(include='object')
df_train.describe().columns
numerical = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']
corr_matrix = df_train[numerical].corr()

fig,ax= plt.subplots(figsize=(10,10))

sns.heatmap(corr_matrix, ax=ax, annot=True)
%config InlineBackend.figure_format = 'png'

sns.pairplot(data=df_train[numerical].dropna())
df = pd.concat([df_train, df_test], sort=True)

df.head()
df['Survived'].unique()
df['Group'] = df['Survived'].map({1:'train', 0:'train', np.nan:'test'})
a = list(df['Ticket'].unique())

# sorted(a)
df['Ticket_1'], df['Ticket_2'] = df['Ticket'].str.split(' ', 1).str



df['Ticket_P2'] = np.where(df['Ticket_2'].isnull() == True, df['Ticket_1'], df['Ticket_2'])

df['Ticket_P1'] = np.where(df['Ticket_2'].isnull() == True, 'no P1', df['Ticket_1'])



df['Ticket_P2'] = pd.to_numeric(df['Ticket_P2'], errors='coerce')

df['Ticket_P2'] = df['Ticket_P2'].fillna(0)



del df['Ticket_1']

del df['Ticket_2']

del df['Ticket']
a = list(df['Cabin'].unique())

# sorted(a)
df['Cabin_P1'] = df['Cabin'].str.extract('([A-Z]+)')

df['Cabin_P2'] = df['Cabin'].str.replace('([A-Z]+)', '')



df['Cabin_P1'] = df['Cabin_P1'].fillna('U')

df['Cabin_P2'] = df['Cabin_P2'].fillna('U')



df['Multi_Cabin'] = np.where(df['Cabin_P2'].apply(len)>3, 1, 0)
del df['Cabin']
df['Age'] = df['Age'].fillna(df['Age'].mode())

df['Embarked'] = df['Embarked'].fillna('Unknown')
df['Sex'] = df['Sex'].map({'male':1, 'female':0})
df.describe(include='object')
df.set_index('PassengerId', inplace=True)
one_hot = pd.get_dummies(df['Pclass'])

df = df.join(one_hot, lsuffix='Pclass')
one_hot = pd.get_dummies(df['SibSp'])

df = df.join(one_hot, lsuffix='SibSp')
one_hot = pd.get_dummies(df['Parch'])

df = df.join(one_hot, lsuffix='Parch')
one_hot = pd.get_dummies(df['Ticket_P1'])

df = df.join(one_hot, lsuffix='Ticket_P1')
one_hot = pd.get_dummies(df['Cabin_P1'])

df = df.join(one_hot, lsuffix='Cabin_P1')
one_hot = pd.get_dummies(df['Embarked'])

df = df.join(one_hot, lsuffix='Embarked')
df.head().T
df_use = df[df['Group'] == 'train']
numeric = df_use.describe().columns
df_use = df_use[numeric]
df_use.describe()
df_score = pd.DataFrame({'numleaves':[], 'minchildsamples':[], 'maxdepth':[], 'learningrate':[], 'regalpha':[], 'reglambda':[], 'mean_cvs':[]})

for numleaves in [16]:

    for minchildsamples in range(5,13,2):

        for maxdepth in range(-1,10,3):

            for learningrate in [0.1]:

                for regalpha in [0]:

                    for reglambda in (0,0.1):

                        clf = lgb.LGBMClassifier(num_leaves=numleaves, min_child_samples=minchildsamples, max_depth=maxdepth, learning_rate=learningrate, reg_alpha=regalpha, reg_lambda=reglambda)

                        cvs = cross_val_score(clf, df_use.drop(columns='Survived'), df_use['Survived'], cv=5)

                        df_score = df_score.append({'numleaves':numleaves, 'minchildsamples':minchildsamples, 'maxdepth':maxdepth, 'learningrate':learningrate, 'regalpha':regalpha, 'reglambda':reglambda, 'mean_cvs':np.mean(cvs)}, ignore_index=True)               
df_score = df_score[['numleaves','minchildsamples','maxdepth','learningrate','regalpha','reglambda','mean_cvs']].sort_values('mean_cvs', ascending=False)

df_score.head(50)
clf_final = lgb.LGBMClassifier(num_leaves=16, min_child_samples=7, max_depth=8)
clf_final.fit(df_use.drop(columns='Survived'), df_use['Survived'])
df_test_final = df[df['Group'] == 'test']

numeric = df_test_final.describe().columns

df_test_final = df_test_final[numeric].drop(columns='Survived')
df_pre = clf_final.predict(df_test_final)

df_pre
df_result = pd.Series(df_pre, name='Survived', index=df_test_final.index).to_frame()

df_result.reset_index(inplace=True) 
df_result['Survived'] = df_result['Survived'].astype(int)
df_result.to_csv('result.csv', index=False)