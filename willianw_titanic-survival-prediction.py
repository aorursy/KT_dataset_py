%matplotlib inline

from sklearn.model_selection import cross_val_score, RandomizedSearchCV, learning_curve
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import roc_curve, f1_score, make_scorer, r2_score, accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pandas as pd
import numpy as np
import glob
import re
import warnings

warnings.filterwarnings(action='ignore')
pd.options.mode.chained_assignment = None  # default='warn'
ts = pd.read_csv('../input/train.csv')
vd = pd.read_csv('../input/test.csv')
vd['Survived'] = np.nan
df = pd.concat([ts, vd], sort=False)
print(ts.shape, vd.shape)
print('Numerical Features')
df.describe(include=['int64', 'float64']).T
print('Categorical Features')
df.describe(include=['object', 'category']).T
df['Train'] = df['Survived'].notna() * 1
print("Size of Trainning set: %.1f%%"%(df['Train'].mean()*100))
df['Cabin'] = df['Cabin'].fillna('').str.replace(r'[^A-Z]', '').apply(lambda x: str(x)[0] if len(str(x)) > 0 else '').astype('category')
df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip()).astype('category')
df['Family_size'] = df['Parch'] + df['SibSp'] + 1
df['IsAlone'] = (df['Family_size'] == 1) * 1
df['Pclass'] = df['Pclass'].astype('category')
'; '.join(df.sample(10)['Name'])
df['nSurnames'] = df['Name'].apply(lambda x: len(re.sub(r'\(.*\)', '', x).strip().split(' ')))
fig, ax = plt.subplots(figsize=(20, 3))
sns.countplot('Train', hue='Sex', data=df)
ax.set_xticklabels(['Test Set', 'Train Set'])
plt.show()
_, bins = np.histogram(df[df['Age'].notna()]["Age"])
g = sns.FacetGrid(df, hue="Sex", aspect=4, height=4)
g = g.map(sns.distplot, "Age", bins=bins).add_legend()
df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'male' else 1).astype('int')
fig, axs = plt.subplots(1, 3, figsize=(20,5))
bins = np.linspace(df['Age'].min(), df['Age'].max(), 10)
axs[0].hist(df[(df['Age'].notna())&(df['Survived']==1)]['Age'], bins=bins, alpha=0.5, label='Survived')
axs[0].hist(df[(df['Age'].notna())&(df['Survived']==0)]['Age'], bins=bins, alpha=0.5, label='Died')
axs[0].legend()
axs[1].hist(df[df['Age'].notna()]['Age'], cumulative=True, bins=50)
sns.violinplot('Sex', 'Age', hue='Survived', data=df, inner='quartile', ax=axs[2], split=True)
plt.show();
def age_treatment(age):
    if age == age:
        if age < 5:
            return 0
        elif age < 18:
            return 1
        elif age < 45:
            return 2
        else:
            return 3
    return np.nan
old_age = df['Age'].copy()
df['Age'] = df['Age'].apply(age_treatment)
df['Age_isna'] = (df['Age'].isna()) * 1
title_df = df[['Title']]
columns_for_title = ['Sex', 'SibSp', 'Parch', 'Fare', 'Age']
title_ds = StandardScaler().fit_transform(pd.get_dummies(df[columns_for_title].fillna(df[columns_for_title].median())))
title_df['x'], title_df['y'] = TSNE(2, perplexity=100, early_exaggeration=800, verbose=True).fit_transform(title_ds).T
sns.lmplot(
    'x', 'y', hue='Title', data=title_df, fit_reg=False, size=8, aspect=2,
    palette=sns.color_palette("Set1", n_colors=18, desat=1),
    hue_order = title_df['Title'].value_counts().index.tolist(),
    legend_out=False
  )
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex='row', figsize=(25,16))
sns.boxplot('Title', 'Age', data=df, ax=ax1)
sns.boxplot('Title', 'Fare', data=df, ax=ax2)
sns.boxplot('Title', 'SibSp', data=df, ax=ax3)
sns.boxplot('Title', 'Parch', data=df, ax=ax4)
ax1.set_title("Grouping Titles");
cluster = KMeans(5, random_state=15)
title_df['cluster'] = cluster.fit_predict(title_ds)

plt.figure(figsize=(20,5))
sns.countplot(title_df['Title'], hue=title_df['cluster'])
plt.yscale('log')
print("Let's apply the titles transformation!")
titles = {
    'Mr': 'Mr',
    'Miss': 'Miss',
    'Mrs': 'Mrs',
    'Master': 'Master',
    'Dr': 'Master',
    'Rev': 'Mr',
    'Col': 'Mr',
    'Ms': 'Miss',
    'Mlle': 'Mrs',
    'Major': 'Mr',
    'Don': 'Mr',
    'Dona': 'Miss',
    'the Countess': 'Miss',
    'Jonkheer': 'Mrs',
    'Lady': 'Mrs',
    'Sir': 'Mr',
    'Mme': 'Miss',
    'Capt': 'Mr'
}
fig, axs = plt.subplots(1, 2, figsize=(25,5))
sns.countplot('Title', data=df, hue='Train', ax = axs[0])
axs[0].set_title("Then")
df['Title'] = df['Title'].apply(lambda x: titles[x] if x in titles.keys() else x).astype('category')
sns.countplot('Title', data=df, hue='Train', ax = axs[1])
axs[1].set_title("Now")
plt.show();
df['Embarked'] = df['Embarked'].fillna('S').astype('category')
fig, axs = plt.subplots(1, 4, figsize=(25, 4)) 
sns.barplot('Embarked', 'Survived', data=df, ax=axs[0])
axs[0].set_title('Survival Rate by Embarked Spot')
sns.violinplot('Embarked', 'Age', data=df, ax=axs[1])
axs[1].set_title('Age by Embarked Spot')
sns.countplot('Title', hue='Embarked', data=df, ax=axs[2])
sns.barplot('Family_size', 'Fare', hue='Embarked', data=df, ax=axs[3])
plt.show()
plt.figure(figsize=(20,5))
sns.barplot(x=df['Cabin'], y=df['Survived'], hue=df['Cabin'], capsize=0.2)
plt.title("Survivance Rate vs. Cabin")
plt.show();
plt.figure(figsize=(20,5))
sns.boxplot(x=df['Cabin'], y=np.log(df['Fare']), hue=df['Cabin'])
plt.legend(loc="upper right")
plt.title("Fare vs Cabin")
plt.show();
df['has_fare'] = df['Fare'].notna().astype(int)
from scipy.stats import boxcox
plt.figure(figsize=(20,4))
sns.distplot(boxcox(df['Fare'].apply(lambda x: 100 if x > 70 else x).fillna(df['Fare'].median())+0.1)[0])
df['Fare'] = boxcox(df['Fare'].apply(lambda x: 100 if x > 100 else x).fillna(df['Fare'].median())+0.1)[0]
fig, axs = plt.subplots(1, 2, figsize=(25,5))

axs[0].hist(
    [df[(df['Survived']==1)&(df['Train']==1)]['Fare'], df[(df['Survived']==0)&(df['Train']==1)]['Fare']],
    label=["Survived", "Died"], alpha=1, stacked=True)
axs[0].set_title("Train")
axs[0].legend()

axs[1].hist(
    [df[(df['Train']==0)]['Fare']],
    label=["Test"], alpha=1, stacked=True)
axs[1].set_title("Test")
axs[1].legend()
axs[1].set_ylim(top=axs[0].get_ylim()[1])
plt.show();
df['Name'].sort_values().unique()
import lightgbm as lgb

df[df.select_dtypes('O').columns] = df[df.select_dtypes('O').columns].astype('category')
age_train = np.in1d(np.arange(len(df)), np.where(df['Age'].notna())[0][:697])
age_test  = np.in1d(np.arange(len(df)), np.where(df['Age'].notna())[0][697:])

ageX = df.drop(['Age', 'Name', 'Ticket'], axis=1)
ageX, agey = ageX.fillna(ageX.median()), df[['Age']]

train_data = lgb.Dataset(ageX[age_train],label=agey[age_train],feature_name=ageX.columns.tolist(),
    categorical_feature=[x for x, column in enumerate(ageX.columns) if x in ageX.select_dtypes(['category']).columns])
test_data = lgb.Dataset(ageX[age_test], label=agey[age_test])

age_parameters = {'task': 'train','objective': 'multiclass','num_class': len(agey[age_train]['Age'].unique()),
    'verbose': 0,'metrics': 'multi_logloss','early_stopping_round': 1000}

age_model = lgb.train(age_parameters,train_data,valid_sets=test_data)

print("Model: %.2f"%   accuracy_score(agey[age_test], age_model.predict(ageX[age_test]).argmax(axis=1)))
print("Baseline: %.2f"%accuracy_score(agey[age_test], np.full(agey[age_test].shape, agey[age_train].median())))
age_score = pd.DataFrame(
    data={
        'age': np.squeeze(agey[age_test]),
        'pred': np.argmax(age_model.predict(ageX[age_test]), axis=1)
    }).sort_values(by='pred', ascending=True).reset_index(drop=True)

from sklearn.metrics import confusion_matrix
cfm = confusion_matrix(age_score['age'], age_score['pred'])
print("Age Model Accucacy: %.1f%%"%(cfm.diagonal().sum()/cfm.sum()*100))
train_data = lgb.Dataset(
    ageX[age_train|age_test],label=agey[age_train|age_test],feature_name=ageX.columns.tolist(),
    categorical_feature=[x for x, column in enumerate(ageX.columns) if x in ageX.select_dtypes(['category']).columns])

age_parameters = {'task': 'train','objective': 'multiclass','num_class': len(agey[age_train]['Age'].unique()),'verbose': 0,
                  'metrics': 'multi_logloss'}
age_model = lgb.train(age_parameters,train_data)
df['Age_isna'] = df['Age'].isna().astype(int)
df.loc[df['Age'].isna(), 'Age'] = np.argmax(age_model.predict(ageX[~(age_train|age_test)]), axis=1)
ages = {0: 'baby', 1:'teen', 2:'adult', 3:'senior'}
df['Age'] = df['Age'].map(ages).astype('category')
df2 = pd.get_dummies(df.drop(['PassengerId', 'Survived'], axis=1))
df2.fillna(df2.median(), inplace=True)

t = TSNE(n_components=2, early_exaggeration=20, perplexity=100)
x, y = t.fit_transform(df2).T

plt.figure(figsize=(24, 5))
plt.scatter(x, y, c=df['Survived'], cmap=plt.get_cmap('coolwarm', 3))
plt.title("Similarity of survival")
plt.colorbar(ticks=[.0, .5, 1.])
plt.axis('off')
plt.show();
name = df['Name']
df.drop(['Train', 'Ticket', 'Name', 'Cabin'], axis=1, inplace=True)
df.info(verbose=True)
df = pd.get_dummies(df)
corr = pd.DataFrame(df.corr())
plt.figure(figsize=(25, 8))
sns.heatmap(corr, cmap='bwr', vmin=-1, vmax=1);
df.loc[df['Age_baby']==1, 'Survived'].value_counts()
target = df['Survived'].isna()
ts = df[~target]
vd = df[target]
target.astype(float).mean()
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
X, y = shuffle(ts.drop('Survived', axis=1), ts['Survived'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
X_train.shape, X_test.shape
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
def model_name(model):
    return str(model).split('(')[0]
model_search = {}
models = [xgb.XGBClassifier(), GaussianNB(), BernoulliNB(), LogisticRegression(), Lasso(), GaussianProcessClassifier(),
         KNeighborsClassifier(), RandomForestClassifier(), MLPClassifier(), KNN(4)]
for model in models:
    results = cross_val_score(model, X_train, y_train, cv=3)
    model_search[model_name(model)] = results
result = pd.DataFrame.from_records(model_search).T
result['mean'] = result.mean(axis=1); result['std'] = result.std(axis=1)
result.sort_values(by='mean', ascending=False)
search_xgb = False
if search_xgb:
    clf = xgb.XGBClassifier()
    rscv = RandomizedSearchCV(clf, params, n_jobs=-1, cv=3, n_iter=100, scoring='accuracy', verbose=True, return_train_score=False)
    rscv.fit(X, y);

    results = pd.DataFrame(rscv.cv_results_).sort_values(by='mean_test_score', ascending=False)
    results = results[['mean_test_score','std_test_score']]
    model = rscv.best_estimator_
    results.head()
else:
    model = LogisticRegression()
    model.fit(X_train, y_train)
if search_xgb:
    n, score_ts, score_vd = learning_curve(model, X_train, y_train, cv=3, scoring='accuracy');
    plt.figure(figsize=(20, 5))
    plt.plot(n, score_ts.mean(axis=1), label='Training')
    plt.plot(n, score_vd.mean(axis=1), label='Validation')
    plt.legend()
    plt.show();
cross_val_score(model, X_test, y_test, cv=5)
model = VotingClassifier(estimators=[
    ('xgb', xgb.XGBClassifier()),
    ('rf',  RandomForestClassifier()),
    ('lr',  LogisticRegression()),
    ('gnb', GaussianNB()),
    ('mlp', MLPClassifier())
], voting='soft', n_jobs=-1)
cross_val_score(model, X_test, y_test, cv=3)
model.fit(X_train, y_train);
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, model.predict(X_test))
print("False Positives and False Negatives")
ŷ = model.predict(X_test)
X_test[ŷ != y_test].head()
X_test_compare = X_test.copy()
X_test_compare2 = X_test_compare.copy()
X_test_compare['Correct'] = (ŷ == y_test).astype(float)
X_test_compare2['Correct'] = 0.5
comparative = X_test_compare.append(X_test_compare2).groupby('Correct').agg('mean').drop(['PassengerId'], axis=1)

print(comparative)
plt.figure(figsize=(20,4))
sns.heatmap((comparative-comparative.mean())/comparative.std())
true = ŷ == y_test
positive = ŷ == 1
scaled = StandardScaler().fit_transform(X_test)
tsne = TSNE(2)
px, py = tsne.fit_transform(scaled).T
plt.figure(figsize=(20, 5))
sns.scatterplot(px, py, hue=true, style=positive, markers=['o', 'P'], palette="seismic_r")
plt.show();
ŷ = model.predict_proba(X_test).T[1]
y_pred = model.predict(X_test)

plt.figure(figsize=(20, 2))
plt.scatter(ŷ, np.random.uniform(0, 1, size=ŷ.shape[0]), c=y_pred)
plt.xlim(0, 1)
plt.show();
ŷ = model.predict_proba(X_test).T[1]
bins = np.linspace(0, 1, 20)

plt.figure(figsize=(20, 5))
sns.distplot(ŷ[y_test==1], bins=bins, label='Survived')
sns.distplot(ŷ[y_test==0], bins=bins, label='Died')
plt.legend()
plt.xlim(0, 1)
plt.show();
ŷ = model.predict_proba(X_test).T[1]
fp, tp, thr = roc_curve(y_test, ŷ)
prec = tp / (tp + fp)
f1 = 2 * (np.multiply(prec,tp))/(prec+tp)
bst = thr[np.argmax(f1[1:])+1]
plt.figure(figsize=(20, 3))
plt.plot(thr, prec)
plt.plot(thr, tp)
plt.plot(thr, f1, c='k')
plt.axvline(bst, c='g')
plt.annotate('{:.2f}'.format(bst), (bst+0.005, 0), color='g')
plt.xlim(0, 1)
plt.show();
model.fit(X, y)
y_pred = model.predict_proba(vd.drop(['Survived'], axis=1)).T[1]
vd['Survived'] = (y_pred > bst).astype(int)
vd[['PassengerId', 'Survived']].to_csv('submission.csv', index=None)