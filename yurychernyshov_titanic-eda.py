import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os

from pandas.plotting import scatter_matrix

from sklearn import tree
from sklearn import model_selection, feature_selection
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, recall_score

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    print(big_string)
    return np.nan
df = pd.read_csv('/kaggle/input/titanic/train.csv')
dft = pd.read_csv('/kaggle/input/titanic/test.csv')
datas=[df,dft]
df.head(2)
print("В исходных данных Train {} строк и {} столбцов".format(df.shape[0],df.shape[1]))
print("В исходных данных Test {} строк и {} столбцов".format(dft.shape[0],dft.shape[1]))
df.columns
print(df.info())
print(df.isnull().sum())
dft.columns
print(dft.info())
print(dft.isnull().sum())
print(df.dtypes.value_counts())
print(df.select_dtypes(include = ['float64', 'int64']).head(2))
df.loc[:,['Age','SibSp','Parch','Fare']].describe()
print(dft.dtypes.value_counts())
print(dft.select_dtypes(include = ['float64', 'int64']).head(2))
dft.loc[:,['Age','SibSp','Parch','Fare']].describe()
df.info()
dft.info()
print("Последовательность Train содержит {} уникальных элементов, минимальный: {}, максимальный: {}".format(
    len(df.PassengerId.unique()),df['PassengerId'].min(), df['PassengerId'].max()))
print("Последовательность Test содержит {} уникальных элементов, минимальный: {}, максимальный: {}".format(
    len(dft.PassengerId.unique()),dft['PassengerId'].min(), dft['PassengerId'].max()))
df['Survived'].unique()
print(df.Survived.value_counts())
print(df.Survived.value_counts(normalize=True))
print("{}% пассажиров спаслось".format(round(df[df['Survived']==1]['PassengerId'].count()/df['Survived'].count(),3)*100))
df['Pclass'].unique()
print(df['Pclass'].value_counts())
df['Pclass'].value_counts(normalize=True)
df.groupby(['Pclass','Survived'])['PassengerId'].count()
pd.crosstab(df['Pclass'], df['Survived'], margins=True)
pd.crosstab(df['Pclass'], df['Survived'], margins=True, normalize=True)
for i in [1,2,3]:
    a = df[(df['Pclass']==i) & (df['Survived']==0)]['Survived'].count()
    b = df[df['Pclass']==i]['Survived'].count()
    print("Процент погибших пассажиров кают {}-го класса относительно только своего класса: {} ({} из {})".format(i,round(a/b,2),a,b))           
a = df[(df['Pclass']==2) & (df['Survived']==0)]['Survived'].count() + df[(df['Pclass']==3) & (df['Survived']==0)]['Survived'].count()
b = len(df)
print("Процент погибших пассажиров кают 2 и 3-го класса относительно всех пассажиров: {} ({} из {})".format(round(a/b,2),a,b))           
print(df['Name'].head(10))
print(df[df['Name'].isna()])
titles = ['Mr.','Mrs.', 'Miss.', 'Ms.', 'Rev.', 'Dr.', 'Master.', 
          'Don.','Major.','Mme.','Mlle.','Col.','Capt.', 'Jonkheer.', 
          'Countess.', 'Sir.', 'Lady.', 'Dona.']
vSum = 0
print("Статистика титулов в выборке train")
for title in titles:
    c = df[df['Name'].str.find(title)!=-1]['Name'].count()
    print("{}:{}".format(title, c))
    vSum += c
print("Sum: {}".format(vSum))
vSum = 0
print("Статистика титулов в выборке test")
for title in titles:
    c = dft[dft['Name'].str.find(title)!=-1]['Name'].count()
    print("{}:{}".format(title, c))
    vSum += c
print("Sum: {}".format(vSum))
title = '|'.join(titles)
print(df[df['Name'].str.contains(title)!=True]['Name'])
print("Количество строк с титулами, неучтенными в списке titles: {}".format(
    df[df['Name'].str.contains(title)!=True]['Name'].count()))
title = '|'.join(titles)
print(dft[dft['Name'].str.contains(title)!=True]['Name'])
print("Количество строк с титулами, неучтенными в списке titles: {}".format(
    dft[dft['Name'].str.contains(title)!=True]['Name'].count()))
vSumC, vSumS = 0, 0
dftitles = pd.DataFrame(columns=['Number','Survived','Percentage'])
for title in titles:
    c = df[df['Name'].str.find(title)!=-1]['Survived'].count()
    s = df[df['Name'].str.find(title)!=-1]['Survived'].sum()
    if c!=0:
        dftitles.loc[title] = {'Number':c, 'Survived':s, 'Percentage':round(s/c,2)}
        print("Титул: {:<10}, количество людей с титулом: {:>4},  из них выживших: {:>4} ({}%)".format(
            title, c, s, round(s/c*100,2)))
    vSumC += c
    vSumS += s
print("Всего: {}, выживших:{}".format(vSumC, vSumS))
dftitles.sort_values('Number',ascending=False)
dftitles.sort_values('Survived',ascending=False)
dftitles.sort_values('Percentage',ascending=False)
print("Всего в выборках train и test {} различных титулов".format(len(titles)))
df['Sex'].unique()
if not all(df['Sex'].isna()):
    print("Нет ячеек без значения")
else:
    print("{} ячеек без значения".format(df[df['Sex'].isna()].shape[0]))
df.groupby(['Sex','Survived'])['PassengerId'].count()
vNumMale = df[df['Sex']=='male']['PassengerId'].count()
vNumSurvivedMale = df[(df['Sex']=='male')&(df['Survived']==1)]['PassengerId'].count()
vNumFemale = df[df['Sex']=='female']['PassengerId'].count()
vNumSurvivedFemale = df[(df['Sex']=='female')&(df['Survived']==1)]['PassengerId'].count()
print("На корабле находилось {} мужчин (выжило {}, {}%) и {} женщин (выжило {}, {}%)".format(
    vNumMale, vNumSurvivedMale, round(vNumSurvivedMale/vNumMale,2)*100,
    vNumFemale, vNumSurvivedFemale, round(vNumSurvivedFemale/vNumFemale,2)*100))
pd.crosstab(df['Survived'],df['Sex'], margins=True)
pd.crosstab(df['Survived'],df['Sex'], normalize=True, margins=True)
print("Средний возраст погибших мужчин: {:.2f}".format(df[(df['Sex']=='female') & (df['Survived']==False)]['Age'].mean()))
print("Средний возраст погибших женщин: {:.2f}".format(df[(df['Sex']=='male') & (df['Survived']==False)]['Age'].mean()))
print(pd.crosstab(df['Pclass'],df['Survived'], normalize=True))
print(pd.crosstab(df['Sex'],df['Survived'], normalize=True))
df['Age'].isna().sum()
plt.boxplot(df['Age'].fillna(df['Age'].median()))
plt.show()
print(df[df.Sex=='female']['Age'].mean())
print(df[df.Sex=='male']['Age'].mean())
print("Средний возраст погибших мужчин: {:.2f}".format(df[(df['Sex']=='female') & (df['Survived']==False)]['Age'].mean()))
print("Средний возраст погибших женщин: {:.2f}".format(df[(df['Sex']=='male') & (df['Survived']==False)]['Age'].mean()))
print("Средний возраст выживших мужчин: {:.2f}".format(df[(df['Sex']=='female') & (df['Survived']==True)]['Age'].mean()))
print("Средний возраст выживших женщин: {:.2f}".format(df[(df['Sex']=='male') & (df['Survived']==True)]['Age'].mean()))
print("В исходных данных Age содержится {} ячеек без значений из {} ({}% от всех данных)".format(
    df[df['Age'].isna()].shape[0], 
    df.shape[0],
    round(df[df['Age'].isna()].shape[0]/df.shape[0],2)*100))
df.loc[:,['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr()
df['SibSp'].unique()
if not any(df['SibSp'].isna()):
    print("Нет ячеек без значения")
else:
    print("{} ячеек без значения".format(df[df['SibSp'].isna()].shape[0]))
print(df.groupby(['SibSp'])['PassengerId'].count())
print(df.groupby(['Survived','SibSp'])['PassengerId'].count())
df['SibSp'].unique()
df['SibSp'].isna().sum()
print(df.groupby(['Survived','Parch'])['PassengerId'].count())
df['Ticket']
df[df.Ticket.str.match(r'PP')==True][['Ticket','Embarked','Fare','Name','Age']]
from collections import Counter
c = Counter(df['Ticket'])
s = set([i for i in c.keys() if c[i]>1])
print("Приобретено {} коллективных билетов".format(len(s)))
df['Fare']
df['Fare'].isna().sum()
plt.boxplot(df['Fare'])
plt.show()
df.Cabin.isna().sum()
print(df.groupby(['Survived','Cabin'])['PassengerId'].count())
df['Embarked'].unique()
display(df.groupby(['Embarked'])['PassengerId'].count())
display(df.groupby(['Survived','Embarked'])['PassengerId'].count())
df_proc = df.copy()
dft_proc = dft.copy()
datas_proc = [df_proc, dft_proc]
# complete or delete missing values in train and test/validation dataset
for dataset in datas_proc:
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
    #complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)
for dataset in datas_proc:
    dataset["Embarked"] = dataset["Embarked"].map({"S":2,"C":1,"Q":0}).astype(int)
    dataset["Sex"]  = dataset["Sex"].map({"female":1,"male":0}).astype(int)
for dataframe in datas_proc:    
    dataframe['FamilySize'] = dataframe['SibSp'] + dataframe['Parch'] + 1
dict_titles = {'Mr.': 1,
               'Mrs.': 2, 
               'Miss.': 3, 
               'Ms.': 4, 
               'Rev.': 5, 
               'Dr.': 6, 
               'Master.': 7, 
               'Don.': 8,
               'Major.': 9,
               'Mme.': 10,
               'Mlle.': 11,
               'Col.': 12,
               'Capt.': 13, 
               'Jonkheer.': 14, 
               'Countess.': 15, 
               'Sir.': 16, 
               'Lady.': 17,
               'Dona.': 18
              }

for dataframe in datas_proc:
    dataframe['Title'] = dataframe['Name'].map(lambda x: substrings_in_string(x, titles))
    dataframe['TitleIndex'] = dataframe['Name'].map(lambda x: dict_titles[substrings_in_string(x, titles)])
AGE_BABY = 1
AGE_TEEN = 2
AGE_MIDDLE = 3
AGE_OLD = 4

for dataframe in datas_proc:
    dataframe['AgeRange'] = pd.cut(dataframe['Age'], [0,5,18,50,100], labels=[AGE_BABY, AGE_TEEN, AGE_MIDDLE, AGE_OLD])
for dataframe in datas_proc:
    dataframe['FareRange'] = pd.cut(dataframe['Fare'], [0,50,100,513], labels=[1,2,3])
#for dataset in datas:    
#    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
#    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
for dataset in datas_proc:    
    drop_column = ['PassengerId', 'Name', 'Cabin', 'Ticket']
    dataset.drop(drop_column, axis=1, inplace = True)
for dataset in datas_proc:    
    drop_column = ['SibSp', 'Parch', 'Fare', 'Title']
    dataset.drop(drop_column, axis=1, inplace = True)
for dataset in datas_proc:    
    drop_column = ['FareRange']
    dataset.drop(drop_column, axis=1, inplace = True)
data = df_proc
labels = data["Survived"]
data = data.drop("Survived",axis=1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(data.values, labels.values, test_size=0.33, random_state=0, stratify=labels)
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)
df.head(2)
#pp = sns.pairplot(df, hue = 'Survived', palette = 'deep', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )
#pp.set(xticklabels=[])
df.corr()
#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(df)
scatter_matrix(df, alpha=0.05, figsize=(15, 15));
rfc = RandomForestClassifier()
rfc.fit(X_train_scaled, y_train)
print(rfc.score(X_test_scaled, y_test))
feats = {}
for feature, importance in zip(data.columns, rfc.feature_importances_):
    feats[feature] = importance
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
importances = importances.sort_values(by='Gini-Importance', ascending=False)
importances = importances.reset_index()
importances = importances.rename(columns={'index': 'Features'})
sns.set(font_scale = 3)
sns.set(style="whitegrid", color_codes=True, font_scale = 1.2)
fig, ax = plt.subplots()
fig.set_size_inches(10,5)
sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
plt.xlabel('Importance', fontsize=15, weight = 'bold')
plt.ylabel('Features', fontsize=15, weight = 'bold')
plt.title('Feature Importance', fontsize=15, weight = 'bold')
plt.show()
display(importances)
data = df_proc
labels = data["Survived"]
data = data[['Age','TitleIndex','Sex','Pclass','FamilySize']]

X_train, X_test, y_train, y_test = model_selection.train_test_split(data.values, labels.values, test_size=0.33, random_state=0, stratify=labels)
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

rfc_1 = RandomForestClassifier()
rfc_1.fit(X_train_scaled, y_train)
print(rfc_1.score(X_test_scaled, y_test))
pca_test = PCA(n_components=5)
pca_test.fit(X_train_scaled)
sns.set(style='whitegrid')
plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.axvline(linewidth=4, color='r', linestyle = '--', x=3, ymin=0, ymax=1)
plt.show()
evr = pca_test.explained_variance_ratio_
cvr = np.cumsum(pca_test.explained_variance_ratio_)
pca_df = pd.DataFrame()
pca_df['Cumulative Variance Ratio'] = cvr
pca_df['Explained Variance Ratio'] = evr
display(pca_df.head(10))
pca = PCA(n_components=4)
pca.fit(X_train_scaled)
X_train_scaled_pca = pca.transform(X_train_scaled)
X_test_scaled_pca = pca.transform(X_test_scaled)

rfc_2 = RandomForestClassifier()
rfc_2.fit(X_train_scaled_pca, y_train)
print(rfc_2.score(X_test_scaled_pca, y_test))
n_estimators = [int(x) for x in np.linspace(100, 1000, 10)]
max_features = ['log2', 'sqrt']
max_depth = [int(x) for x in np.linspace(1, 15, 15)]
min_samples_split = [int(x) for x in np.linspace(2, 50, 10)]
min_samples_leaf = [int(x) for x in np.linspace(2, 50, 10)]
bootstrap = [True, False]
param_dist = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rs = RandomizedSearchCV(rfc, 
                        param_dist, 
                        n_iter = 100, 
                        cv = 3, 
                        verbose = 1, 
                        n_jobs=-1, 
                        random_state=0)
rs.fit(X_train_scaled_pca, y_train)
rs.best_params_
#rfc.get_params()
rs_df = pd.DataFrame(rs.cv_results_).sort_values('rank_test_score').reset_index(drop=True)
rs_df = rs_df.drop([
            'mean_fit_time', 
            'std_fit_time', 
            'mean_score_time',
            'std_score_time', 
            'params', 
            'split0_test_score', 
            'split1_test_score', 
            'split2_test_score', 
            'std_test_score'],
            axis=1)
rs_df.head(10)
fig, axs = plt.subplots(ncols=3, nrows=2)
sns.set(style="whitegrid", color_codes=True, font_scale = 1)
fig.set_size_inches(20,15)
sns.barplot(x='param_n_estimators', y='mean_test_score', data=rs_df, ax=axs[0,0], color='lightgrey')
axs[0,0].set_ylim([.75,.80])
axs[0,0].set_title(label = 'n_estimators', size=10, weight='bold')
sns.barplot(x='param_min_samples_split', y='mean_test_score', data=rs_df, ax=axs[0,1], color='coral')
axs[0,1].set_ylim([.75,.80])
axs[0,1].set_title(label = 'min_samples_split', size=10, weight='bold')
sns.barplot(x='param_min_samples_leaf', y='mean_test_score', data=rs_df, ax=axs[0,2], color='lightgreen')
axs[0,2].set_ylim([.75,.81])
axs[0,2].set_title(label = 'min_samples_leaf', size=10, weight='bold')
sns.barplot(x='param_max_features', y='mean_test_score', data=rs_df, ax=axs[1,0], color='wheat')
axs[1,0].set_ylim([.75,.81])
axs[1,0].set_title(label = 'max_features', size=10, weight='bold')
sns.barplot(x='param_max_depth', y='mean_test_score', data=rs_df, ax=axs[1,1], color='lightpink')
axs[1,1].set_ylim([.75,.81])
axs[1,1].set_title(label = 'max_depth', size=10, weight='bold')
sns.barplot(x='param_bootstrap',y='mean_test_score', data=rs_df, ax=axs[1,2], color='skyblue')
axs[1,2].set_ylim([.75,.81])
axs[1,2].set_title(label = 'bootstrap', size=10, weight='bold')
plt.show()
n_estimators = [300,700,1000]
max_features = ['log2']
max_depth = [3,4,13,15]
min_samples_split = [23,28]
min_samples_leaf = [2,7,12,39]
bootstrap = [True]
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
gs = GridSearchCV(rfc, param_grid, cv = 3, verbose = 1, n_jobs=-1)
gs.fit(X_train_scaled_pca, y_train)
rfc_3 = gs.best_estimator_
gs.best_params_
y_pred = rfc_1.predict(X_test_scaled)
y_pred_pca = rfc_2.predict(X_test_scaled_pca)
y_pred_gs = rfc_3.predict(X_test_scaled_pca)
conf_matrix_baseline = pd.DataFrame(confusion_matrix(y_test, y_pred), index = ['actual 0', 'actual 1'], columns = ['predicted 0', 'predicted 1'])
conf_matrix_baseline_pca = pd.DataFrame(confusion_matrix(y_test, y_pred_pca), index = ['actual 0', 'actual 1'], columns = ['predicted 0', 'predicted 1'])
conf_matrix_tuned_pca = pd.DataFrame(confusion_matrix(y_test, y_pred_gs), index = ['actual 0', 'actual 1'], columns = ['predicted 0', 'predicted 1'])
print(conf_matrix_baseline)
print('Baseline Random Forest recall score', recall_score(y_test, y_pred))
print(conf_matrix_baseline_pca)
print('Baseline Random Forest With PCA recall score', recall_score(y_test, y_pred_pca))
print(conf_matrix_tuned_pca)
print('Hyperparameter Tuned Random Forest With PCA Reduced Dimensionality recall score', recall_score(y_test, y_pred_gs))
data_train = df_proc
labels_train = data_train["Survived"]
data_train = data_train.drop("Survived",axis=1)
gs.best_estimator_.fit(data_train, labels_train)

best_res = gs.best_estimator_.predict(dft_proc)

dfFinal=pd.DataFrame({
    "PassengerId": dft['PassengerId'],
    "Survived": best_res })
dfFinal.to_csv("submission.csv",index=False)