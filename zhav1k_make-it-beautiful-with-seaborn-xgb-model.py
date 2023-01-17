
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(style = 'darkgrid') #


import os
print(os.listdir("../input"))
from pylab import rcParams
rcParams['figure.figsize'] = 25, 12.5
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.info()
categorical = []
numerical = []
for feature in test.columns:
    if test[feature].dtype == object:
        categorical.append(feature)
    else:
        numerical.append(feature)
train[categorical].head()
train[numerical].isnull().sum().sort_values(ascending = False).head(8)
test[numerical].isnull().sum().sort_values(ascending = False).head(8)
train[['meaneduc', 'SQBmeaned']].describe()
train['meaneduc'].fillna(train['meaneduc'].mean(), inplace = True)
train['SQBmeaned'].fillna(train['SQBmeaned'].mean(), inplace = True)
#the same for test
test['meaneduc'].fillna(test['meaneduc'].mean(), inplace = True)
test['SQBmeaned'].fillna(test['SQBmeaned'].mean(), inplace = True)
train['rez_esc'].fillna(0, inplace = True)
train['v18q1'].fillna(0, inplace = True)
train['v2a1'].fillna(0, inplace = True)
sns.set(style = 'darkgrid')
sns_plot = sns.palplot(sns.color_palette('Accent'))
sns_plot = sns.palplot(sns.color_palette('Accent_d'))
sns_plot = sns.palplot(sns.color_palette('CMRmap'))
sns_plot = sns.palplot(sns.color_palette('Set1'))
sns_plot = sns.palplot(sns.color_palette('Set3'))
target_values = train['Target'].value_counts()
target_values = pd.DataFrame(target_values)
target_values['Household_type'] = target_values.index
target_values
mappy = {4: "NonVulnerable", 3: "Moderate Poverty", 2: "Vulnerable", 1: "Extereme Poverty"}
target_values['Household_type'] = target_values.Household_type.map(mappy)
target_values
sns.set(style = 'whitegrid', font_scale=1.4)
fig = plt.subplots(figsize=(15, 8))
ax = sns.barplot(x = 'Household_type', y = 'Target', data = target_values, palette='Accent', ci = None).set_title('Distribution of Poverty in Households')
#Let's find out largest correlations and depict them
corrs = train.corr().abs()
corrs1 = corrs.unstack().drop_duplicates()
strongest = corrs1.sort_values(kind="quicksort", ascending = False)
strongest1 = pd.DataFrame(strongest)
temp = strongest1.index.values
first_cols = [i[0] for i in temp]
second_cols = [j[1] for j in temp]
total_cols_corr = list(set(first_cols[:20] + second_cols[:20]))
strongest.head(25)
corr = train[total_cols_corr].corr()
sns.set(font_scale=1)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
f, ax = plt.subplots(figsize=(25, 12.5))
sns.heatmap(corr, cmap=cmap, annot=True, ax=ax, fmt='.2f')
train['v2a11'] = train.v2a1.apply(lambda x: np.log(x+1))
sns.set(font_scale=1, style="darkgrid")
c =  sns.color_palette('spring_d')[4]
sns_jointplot = sns.jointplot('age', 'meaneduc', data=train, kind='kde', color=c, size=6)
for i in range(1, 5):
    sns.set(font_scale=1, style="white")
    c =  sns.color_palette('spring_d')[i]
    sns_jointplot = sns.jointplot('age', 'meaneduc', data=train[train['Target'] == i], kind='kde', color=c, size=6, stat_func=None)
def plot_distribution(df, var, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, hue = target, aspect = 4, row = row, col = col)
    facet.map(sns.kdeplot, var, shade = True)
    facet.set(xlim = (0, df[var].max()))
    facet.add_legend()
    plt.show()
#select some columns
numerical1 = ['v2a11', 'meaneduc', 'overcrowding'] #monthly pay rent, mean education, overcrowd
for numy in numerical1:
    plot_distribution(train, numy, 'Target')
#In the first graph instead of 0's should be nulls(we changed these before). So there is no info about monthly rate payment for non vulnerable households 
f, ax = plt.subplots(figsize=(20, 10))
sns.boxplot(x='Target', y = 'r4h3',ax = ax, data = train, hue = 'Target' )
ax.set_title('Number of men in households', size = 25)
f, ax = plt.subplots(figsize=(20, 10))
sns.boxplot(x='Target', y = 'r4m3',ax = ax, data = train, hue = 'Target' )
ax.set_title('Number of women in households', size = 25)
ninos = train.groupby(by = 'Target')['hogar_nin', 'Target'].sum()
ninos = pd.DataFrame(ninos)
ninos['mean_children'] = (ninos['hogar_nin']/ninos['Target'])
ninos['Target1'] = ninos.index.map({4: "NonVulnerable", 3: "Moderate Poverty", 2: "Vulnerable", 1: "Extereme Poverty"})
sns.set(style = 'whitegrid', font_scale=1.4)
fig = plt.subplots(figsize=(16, 8))
ax = sns.barplot(x = 'Target1', y = 'mean_children', data = ninos, palette='Pastel1', ci = None).set_title('Mean number on children in different households')
train['v2a1'].replace(0, np.nan, inplace = True)
train["v2a1"] = train.groupby("Target").transform(lambda x: x.fillna(x.median()))
rpd = pd.DataFrame([train['v2a1']/train['hogar_total'], train['Target']]).T
rpd['Target'] = rpd['Target'].map({4: "NonVulnerable", 3: "Moderate Poverty", 2: "Vulnerable", 1: "Extereme Poverty"})
rpd.groupby(by = 'Target').mean()
sns.set(style = 'whitegrid', font_scale=1.4)
fig = plt.subplots(figsize=(16, 8))
ax = sns.barplot(x = 'Target', y = 'Unnamed 0', data = rpd, palette='Pastel1',order = ["Extereme Poverty","Vulnerable","Moderate Poverty", "NonVulnerable"], ci = None).set_title('Montly rent payment per dweller')
#visualization of feature importance of XGB below


valuez = ['meaneduc', 'age', 'qmobilephone','Target', 'r4t3', 'tamhog', 'escolari', 'overcrowding']
tra = pd.melt(train[valuez], "Target", var_name="measurement")
f, ax = plt.subplots()
sns.despine(bottom=True, left=True)
sns.stripplot(x="value", y="measurement", hue="Target",
              data=tra, dodge=True, jitter=True,
              alpha=.05, zorder=1)
sns.pointplot(x="value", y="measurement", hue="Target",
              data=tra, dodge=.532, join=False, palette="dark",
              markers="x", scale=1, ci=None)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[4:], labels[4:], title="Target",
          handletextpad=0, columnspacing=1,
          loc="lower right", ncol=1, frameon=True)

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()
X_scaled = scaler1.fit_transform(train[numerical])
%%time
tsne2d = TSNE(random_state=13012)
tsne_representation2d = tsne2d.fit_transform(X_scaled)
%%time
tsne3d = TSNE(n_components = 3, random_state = 666)
tsne_representation3d = tsne3d.fit_transform(X_scaled)
tsne_representation2d = pd.DataFrame(tsne_representation2d, columns = ['First_col', 'Second_col'])
tsne_representation2d['Target'] = train.loc[:, 'Target']
tsne_representation3d = pd.DataFrame(tsne_representation3d, columns = ['First_col', 'Second_col', 'Third_col'])
tsne_representation3d['Target'] = train.loc[:, 'Target']
sns.set(font_scale=1, style="darkgrid") #CMRmap_r
sns.lmplot( x="First_col", y="Second_col", data=tsne_representation2d, fit_reg=False, hue='Target', legend=False, palette="Set1", size = 17)
plt.legend(loc='lower right')
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
rcParams['figure.figsize'] = 30, 20
fig = pyplot.figure()
ax = Axes3D(fig)

ax.scatter(tsne_representation3d.loc[:, 'First_col'], tsne_representation3d.loc[:, 'Second_col'], tsne_representation3d.loc[:, 'Third_col'], s = 29, c = tsne_representation3d.loc[:, 'Target'],
          edgecolors = 'black')
ax.set_title('t-SNE visualization in 3 dimensions', size = 20)
pyplot.show()
from sklearn.model_selection import train_test_split
import xgboost as xgb
y = train['Target']
train = train.drop(['Id', 'Target'] ,axis = 1)
train = train.select_dtypes(exclude=['object'])
test = test.drop('Id',axis = 1)
test = test.select_dtypes(exclude=['object'])
y.value_counts()
y = y - 1
X_train, X_test, y_train, y_test = train_test_split(train, y, stratify = y, test_size = 0.3, random_state = 666)
y.value_counts()
from sklearn.metrics import f1_score
def evaluate_macroF1(true_value, predictions):  
    pred_labels = predictions.reshape(len(np.unique(true_value)),-1).argmax(axis=0)
    f1 = f1_score(true_value, pred_labels, average='macro')
    return ('macroF1', f1, True) 
params = {
        "objective" : "multi:softmax",
        "metric" : evaluate_macroF1,
        "n_estimators": 100,
        'max_depth' : 9,
        "learning_rate" : 0.23941,
        'max_delta_step': 2,
        'min_child_weight': 9,
        'subsample': 0.72414,
        "seed": 666,
        'num_class': 4,
        'silent': True
    }
xgbtrain = xgb.DMatrix(X_train, label=y_train)
xgbval = xgb.DMatrix(X_test, label=y_test)


watchlist = [(xgbtrain, 'train'), (xgbval, 'valid')]
evals_result = {}
model = xgb.train(params, xgbtrain, 5000, 
                     watchlist,
                    early_stopping_rounds=150, verbose_eval=100)
#we don't need these for now
#xgbtest = xgb.DMatrix(test)
#p_test = model.predict(xgbtest, ntree_limit=model.best_ntree_limit)

#p_test = p_test + 1
xgb_fimp=pd.DataFrame(list(model.get_fscore().items()),columns=['feature','importance']).sort_values('importance', ascending=False)
xgb_fimp1 = xgb_fimp.iloc[0:35]

sns.set(style = 'whitegrid', font_scale=1.4)
fig = plt.subplots(figsize=(10, 15))
ax = sns.barplot(x = 'importance', y = 'feature', data = xgb_fimp1,palette='Accent', ci = None).set_title('Feature importance of XGBooost')
from sklearn.metrics import classification_report
Xgb_test = xgb.DMatrix(X_test)
y_pred = model.predict(Xgb_test, ntree_limit=model.best_ntree_limit)
print(classification_report(y_test, y_pred))




