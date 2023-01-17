import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.api.types import CategoricalDtype

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from matplotlib import style
style.use('seaborn')

import os
print(os.listdir("../input"))

%matplotlib inline
pd.options.display.float_format = '{:,.2f}'.format
PATH = '../input/'
df = pd.read_csv(PATH + 'train.csv', low_memory=False, index_col=0)
test = pd.read_csv(PATH + 'test.csv', low_memory=False, index_col=0)
df.head()
df.shape, test.shape
df.describe(include='all')
test.describe(include='all')
df[['Survived']].groupby(df['Pclass']).mean().plot.bar()
plt.show()
df[['Survived']].groupby(df['Sex']).mean().plot.bar()
plt.show()
bins=[b for b in range(0, 91, 5)]
fig = plt.figure()
# ax0 = plt.subplot2grid((1, 2), (0, 0), colspan=2)
# plt.title("Age Histogram")

ax1 = plt.subplot2grid((1, 2), (0, 0))
df['Age'][df['Survived'] == 1].hist(bins=bins, color='g')
plt.title("Surviving passengers")
ax2 = plt.subplot2grid((1, 2), (0, 1), sharey=ax1)
df['Age'][df['Survived'] == 0].hist(bins=bins, color='r')
plt.title("Deceased passengers")
plt.tight_layout()
plt.show()
threshold = 7
df[['Survived']].groupby(df['Age'].apply(lambda x: f'below {threshold}' if x < threshold else f'above {threshold}')).mean().plot.bar()
plt.show()
df[['Survived']].groupby(df['Name'].apply(lambda x: x.split(sep = ',')[1].split(sep = ".")[0].strip())).mean().sort_values(by="Survived").plot.bar()
plt.show()
df[['Survived']].groupby([df['SibSp']]).mean().plot.bar()
plt.show()
df[['Survived']].groupby([df['Parch']]).mean().plot.bar()
plt.show()
df[['Survived']].groupby([df['Parch'] + df['SibSp']]).mean().plot.bar()
plt.show()
df[['Survived']].groupby([df['Parch'] + df['SibSp']]).count()/df.shape[0]
df[['Survived']].groupby([df['Embarked']]).mean().plot.bar()
plt.show()
df[['Survived']].groupby([df['Embarked']]).count()
df.isna().sum()[df.isna().sum() != 0]/df.shape[0] * 100
test.isna().sum()[test.isna().sum() != 0]/test.shape[0] * 100
df['Cabin'].value_counts()
df['Deck'] = df['Cabin'].fillna(value='NA').apply(lambda x: ''.join(filter(str.isalpha, x))[0] if x != 'NA' else x)
test['Deck'] = test['Cabin'].fillna(value='NA').apply(lambda x: ''.join(filter(str.isalpha, x))[0] if x != 'NA' else x)
df[['Survived']].groupby(df['Deck']).mean().plot.bar()
plt.title("Survival rate by deck")
plt.show()
df[['Name', 'Sex', 'Age']].sort_values('Age')
titles = ['Mr.', 'Master.', 'Rev.', 'Dr.', 'Sir.', 'Don.', 'Capt.', 'Lady.', 'Miss.', 'Ms.', 'Mrs.']

for title in titles:
    print(f"Title: {title}, Median of {df[['Name', 'Age']][df['Name'].str.contains(title)].median()}")
df[['Name', 'Age']][~df['Name'].str.contains(', M')]
def db_mod(db, title): db.loc[db[db['Name'].str.contains(title) & (db['Age'].isnull())].index, 'Age'] = df["Age"][df['Name'].str.contains(title)].median()

for title in titles: 
    db_mod(df, title=title)
    db_mod(test, title=title)
df['Embarked'].fillna(value='NA', inplace=True)
df[['Pclass', 'Fare']].groupby('Pclass').mean()
test[test['Fare'].isnull()]
test.loc[test[(test['Pclass'] == 3) & (test['Fare'].isnull())].index, 'Fare'] = df['Fare'][df['Pclass'] == 3].mean()
df['FamSize'] = df['SibSp'] + df['Parch']
test['FamSize'] = test['SibSp'] + test['Parch']
df['Title'] = df['Name'].apply(lambda x: x.split(sep = ',')[1].split(sep = ".")[0].strip())
test['Title'] = test['Name'].apply(lambda x: x.split(sep = ',')[1].split(sep = ".")[0].strip())
df.columns
cat_list = ['Sex', 'FamSize', 'Pclass']
dummy_list = ['Embarked', 'Deck', 'Title']
cont_list = ['Age', 'Fare']
dep = 'Survived'
train_set = df[cat_list + cont_list + dummy_list + [dep]].copy()
test_set = test[cat_list + cont_list + dummy_list].copy()
def process_df(df, is_train=True, train=train_set):
    for c in cat_list + cont_list + dummy_list:
        if is_train:
            if c in cat_list: df[c] = df[c].astype("category").cat.as_ordered()
        else:
            if c in cat_list: df[c] = df[c].astype(CategoricalDtype(train[c].cat.categories))
        if c in cont_list:
            df[c] = df[c].astype("float32")
        if c in dummy_list:
            cols = pd.get_dummies(df[c], prefix=c + "_")
            for col in cols.columns: df[col] = cols[col]
            df.drop(columns=c, inplace=True)
process_df(train_set)
process_df(test_set, is_train=False)
train_num = train_set.copy()

for c in train_num.columns: train_num[c] = train_num[c].cat.codes if c in cat_list else train_num[c]
plt.figure(figsize=(20, 20))
plt.title("Correlation table")
# sns.heatmap(train_num[[dep] + cat_list + cont_list].corr(), annot=True, cmap="seismic")
sns.heatmap(train_num.corr(), annot=True, cmap="seismic")
plt.tight_layout()
plt.show()
test_num = test_set.copy()

for c in test_num.columns: test_num[c] = test_num[c].cat.codes if c in cat_list else test_num[c]
def acc(targ, pred): return (targ == pred).mean()
r = 0.3
idxs = np.random.choice(np.arange(len(train_num)), size = int(len(train_num) * 0.3), replace=False)
idxs_mask = train_num.index.isin(idxs)
X_train = train_num.drop(columns='Survived')[~idxs_mask]
X_val = train_num.drop(columns='Survived')[idxs_mask]
y_train = train_num['Survived'][~idxs_mask]
y_val = train_num['Survived'][idxs_mask]
X_val.describe(include='all')
test_num.describe(include='all')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
accurs = []
min = 1
max = 15
step = 1
for p in range(min, max, step):
    m = RandomForestClassifier(n_estimators=45, max_features=0.88, min_samples_leaf=p,
                              n_jobs=-1, oob_score=True)
    m.fit(X_train, y_train)
    preds = m.predict(X_val)
    print("-"*30, f'''
    n-estimators: {p}
    Training score: {m.score(X_train, y_train)*100:.2f}%
    Validation score: {m.score(X_val, y_val)*100:.2f}%
    Out-of-Bag score: {m.oob_score_*100:.2f}%
    Accuracy: {acc(y_val, preds)*100:.2f}%
    ''')
    accurs.append([p, acc(y_val, preds)])
accurs = np.array(accurs)
accurs[np.unravel_index(accurs[:, 1].argmax(), accurs[:, 1].shape)[0], :]
m = RandomForestClassifier(n_estimators=70, max_features=0.5, min_samples_leaf=1,
                          n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
preds = m.predict(X_val)
print("-"*30, f'''
Training score: {m.score(X_train, y_train)*100:.2f}%
Validation score: {m.score(X_val, y_val)*100:.2f}%
Out-of-Bag score: {m.oob_score_*100:.2f}%
Accuracy: {acc(y_val, preds)*100:.2f}%
''')
def cross_val(X, y, cv=10):
    accuracies = cross_val_score(estimator = m,
                                 X = X,
                                 y = y,
                                 cv = cv) # k: number of folds - typically 10
    print("Average accuracy:", round(accuracies.mean()*100,1),"%")
    print("Standard deviation:", round(accuracies.std()*100,1),"%")
cross_val(X_train, y_train)
accs = []
targ = m.score(X_train, y_train)
num_features = 15

for c in X_train.columns:
    X = X_train.copy()
    X[c] = X[[c]].sample(frac=1).set_index(X.index)[c]  # random shuffle of one column
    accs.append(targ - m.score(X, y_train))
    

FI = sorted([[c, float(a)] for c, a in zip(X.columns, accs)], key=lambda x: x[1], reverse=True)[:num_features]
pd.DataFrame({'Score loss': [FI[i][1] for i in range(len(FI))], 'Features': [FI[i][0] for i in range(len(FI))]}).set_index('Features').sort_values(by='Score loss', ascending=True).plot.barh()
plt.show()
top = 8
selected = [FI[i][0] for i in range(len(FI))][:top]
Xt = X_train[selected].copy()
Xv = X_val[selected].copy()
t = test_num[selected].copy()
m = RandomForestClassifier(n_estimators=70, max_features=0.5, min_samples_leaf=5,
                          n_jobs=-1, oob_score=True)
m.fit(Xt, y_train)
preds = m.predict(Xv)
print("-"*30, f'''
Training score: {m.score(Xt, y_train)*100:.2f}%
Validation score: {m.score(Xv, y_val)*100:.2f}%
Out-of-Bag score: {m.oob_score_*100:.2f}%
Accuracy: {acc(y_val, preds)*100:.2f}%
''')
cross_val(Xt, y_train)
accs = []
targ = m.score(Xt, y_train)

for c in Xt.columns:
    X = Xt.copy()
    X[c] = X[[c]].sample(frac=1).set_index(X.index)[c]  # random shuffle of one column
    accs.append(targ - m.score(X, y_train))
    
pd.DataFrame({'Score loss': accs}, index=X.columns).sort_values(by='Score loss', ascending=True).plot.barh()
plt.title('Feature Importance')
plt.show() 
m.predict(t).sum()/t.shape[0]
my_submission = pd.DataFrame({'Survived': m.predict(t)}, index=t.index)
my_submission.to_csv('submission.csv')
