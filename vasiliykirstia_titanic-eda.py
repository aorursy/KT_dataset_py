import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import stats
from IPython.display import display

from catboost import CatBoostClassifier, Pool, FeaturesData

plt.style.use('fivethirtyeight')
%matplotlib inline

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
train = pd.read_csv('../input/train.csv', index_col='PassengerId')
test = pd.read_csv('../input/test.csv', index_col='PassengerId')
joined = train.append(test, sort=True)
train.info()
print('-'*40)
test.info()

joined.describe()
class CabinUtils:
    _cabin_pattern = re.compile(r'[a-zA-Z]{1}[0-9]{0,}')
    
    @staticmethod
    def extract_info(value):
        if ((value is None)
            or (isinstance(value, float) and np.isnan(value))
            or (isinstance(value, str) and value.lower() in ['na', 'nan'])
        ):
            return {'Deck': np.nan, 'Side': np.nan}
        
        splited = CabinUtils._cabin_pattern.findall(value)
        if not splited:
            return {'Deck': np.nan, 'Side': np.nan}
        
        deck = splited[0][0]
        side = np.nan
        for cabin in filter(lambda s: 1 < len(s), splited):
            side = 'even' if int(cabin[1:]) % 2 == 0 else 'odd'
            break
        return {'Deck': deck, 'Side': side}
    
    @staticmethod
    def encode(series):
        return series.map(CabinUtils.extract_info).apply(pd.Series)
class TitleUtils:
    _title_pattern = re.compile(r', (?P<title>[a-zA-Z]+)\.')
    
    @staticmethod
    def extract_title(value):
        titles = TitleUtils._title_pattern.findall(value)
        return {'Title': titles[0] if titles else None}
    
    @staticmethod
    def encode(series):
        return series.map(TitleUtils.extract_title).apply(pd.Series)
def _age_bins(age):
    if age <= 18:
        return 'minor'
    else:
        return 'adult'
def _is_alone(row):
    if np.isnan(row['Parch']) or np.isnan(row['SibSp']):
        return np.nan
    return row['Parch'] == 0 or row['SibSp'] == 0
joined = (joined
    .join(CabinUtils.encode(joined['Cabin']))
    .join(TitleUtils.encode(joined['Name']))
    .join(pd.DataFrame({'AgeG': joined['Age'].map(_age_bins, na_action='ignore')}))
    .join(pd.DataFrame({'IsAlone': joined.apply(_is_alone, axis=1)}))
    .join(pd.DataFrame({'FamilySize': joined['Parch'] + joined['SibSp'] + 1})))
joined.sample(5)
corr = joined[
    ['Age', 'Fare', 'Pclass', 'Parch', 'SibSp', 'FamilySize', 'IsAlone', 'Survived']
].corr(method='kendall')
f = plt.figure(figsize=(7, 6))
sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap='RdBu_r', linewidths=5);
f = plt.figure(figsize=(13,8))
f.suptitle('Survived by Age', fontsize=16)

ax1 = plt.subplot2grid((5,1), (0,0), fig=f)
ax1.set_ylabel('Survived')
sns.boxplot(joined['Age'][joined['Survived'] == 1], ax=ax1, width=0.3, color='blue')

ax2 = plt.subplot2grid((5,1), (1,0), fig=f, sharex=ax1)
ax2.set_ylabel('Dead')
sns.boxplot(joined['Age'][joined['Survived'] == 0], ax=ax2, width=0.3, color='orange')

ax3 = plt.subplot2grid((5,1), (2,0), fig=f, sharex=ax1)
ax3.set_ylabel('All')
sns.boxplot(joined['Age'], ax=ax3, width=0.3, color='gray');

ax4 = plt.subplot2grid((5,1), (3,0), fig=f, rowspan=2, sharex=ax1)
ax4.set_ylabel('Survived')
sns.distplot(joined['Age'][joined['Survived'] == 1].dropna(), bins=np.arange(0, 82, 2), ax=ax4, kde=False);
stats.ttest_ind(
    joined['Age'][joined['Survived'] == 1],
    joined['Age'][joined['Survived'] == 0],
    nan_policy='omit', equal_var=False)
f = plt.figure(figsize=(13,7))
f.suptitle('Survived by Fare.', fontsize=16)

ax1 = plt.subplot2grid((3,1), (0,0), fig=f)
ax1.set_ylabel('Survived')
sns.boxplot(joined['Fare'][joined['Survived'] == 1], ax=ax1, width=0.3, color='green')

ax2 = plt.subplot2grid((3,1), (1,0), fig=f, sharex=ax1)
ax2.set_ylabel('Dead')
sns.boxplot(joined['Fare'][joined['Survived'] == 0], ax=ax2, width=0.3, color='red')

ax3 = plt.subplot2grid((3,1), (2,0), fig=f, sharex=ax1)
ax3.set_ylabel('All')
sns.boxplot(joined['Fare'], ax=ax3, width=0.3, color='gray');
stats.ttest_ind(
    joined['Fare'][joined['Survived'] == 1],
    joined['Fare'][joined['Survived'] == 0],
    nan_policy='omit', equal_var=False)
f = plt.figure(figsize=(13,5))
color_map = joined['Survived'].map({np.nan: np.nan, 1: 'gray', 0: 'black'})

ax1 = plt.subplot2grid((1, 5), (0, 0), fig=f)
pd.crosstab(index=joined['Embarked'], columns=joined['Survived'], normalize='index').plot.bar(
    stacked=True, title='Survived by Port', ax=ax1, color=color_map)

ax2 = plt.subplot2grid((1, 5), (0, 1), fig=f)
pd.crosstab(index=joined['Sex'], columns=joined['Survived'], normalize='index').plot.bar(
    stacked=True, title='Survived by Gender', ax=ax2, color=color_map)

ax3 = plt.subplot2grid((1, 5), (0, 2), fig=f)
pd.crosstab(index=joined['Pclass'], columns=joined['Survived'], normalize='index').plot.bar(
    stacked=True, title='Survived by PClass', ax=ax3, color=color_map);

ax4 = plt.subplot2grid((1, 5), (0, 3), fig=f)
pd.crosstab(index=joined['AgeG'], columns=joined['Survived'], normalize='index').plot.bar(
    stacked=True, title='Survived by Age Group', ax=ax4, color=color_map);

ax5 = plt.subplot2grid((1, 5), (0, 4), fig=f)
pd.crosstab(index=joined['IsAlone'], columns=joined['Survived'], normalize='index').plot.bar(
    stacked=True, title='Survived by IsAlone', ax=ax5, color=color_map);
print(
    f"Embarked: {stats.chi2_contingency(pd.crosstab(index=joined['Embarked'], columns=joined['Survived']))[1]:.3f}")
print(
    f"Sex: {stats.chi2_contingency(pd.crosstab(index=joined['Sex'], columns=joined['Survived']))[1]:.3f}")
print(
    f"PClass: {stats.chi2_contingency(pd.crosstab(index=joined['Pclass'], columns=joined['Survived']))[1]:.3f}")
print(
    f"Age Group: {stats.chi2_contingency(pd.crosstab(index=joined['AgeG'], columns=joined['Survived']))[1]:.3f}")
print(
    f"Is Alone: {stats.chi2_contingency(pd.crosstab(index=joined['IsAlone'], columns=joined['Survived']))[1]:.3f}")
f = plt.figure(figsize=(13,5))
color_map = joined['Survived'].map({np.nan: np.nan, 1: 'gray', 0: 'black'})

ax1 = plt.subplot2grid((1, 15), (0, 0), colspan=2, fig=f)
pd.crosstab(index=joined['Side'], columns=joined['Survived'], normalize='index').plot.bar(
    stacked=True, title='Survived by Side', ax=ax1, color=color_map)

ax2 = plt.subplot2grid((1, 15), (0, 2), colspan=5, fig=f, sharey=ax1)
pd.crosstab(index=joined['Deck'], columns=joined['Survived'], normalize='index').plot.bar(
    stacked=True, title='Survived by Deck', ax=ax2, color=color_map)

ax3 = plt.subplot2grid((1, 15), (0, 7), colspan=8, fig=f, sharey=ax1)
pd.crosstab(index=joined['Title'], columns=joined['Survived'], normalize='index').plot.bar(
    stacked=True, title='Survived by Title', ax=ax3, color=color_map);
print(
    f"Side: {stats.chi2_contingency(pd.crosstab(index=joined['Side'], columns=joined['Survived']))[1]:.3f}")
print(
    f"Deck: {stats.chi2_contingency(pd.crosstab(index=joined['Deck'], columns=joined['Survived']))[1]:.3f}")
print(
    f"Title: {stats.chi2_contingency(pd.crosstab(index=joined['Title'], columns=joined['Survived']))[1]:.3f}")
f = plt.figure(figsize=(13, 5))

ax1 = plt.subplot2grid((1, 3), (0, 0), fig=f)
sns.countplot(x='SibSp', hue='Survived', data=joined, ax=ax1)

ax2 = plt.subplot2grid((1, 3), (0, 1), fig=f, sharey=ax1)
sns.countplot(x='Parch', hue='Survived', data=joined, ax=ax2);

ax3 = plt.subplot2grid((1, 3), (0, 2), fig=f, sharey=ax1)
sns.countplot(x='FamilySize', hue='Survived', data=joined, ax=ax3);
print(
    f"SibSp: {stats.chi2_contingency(pd.crosstab(index=joined['SibSp'], columns=joined['Survived']))[1]:.3f}")
print(
    f"Parch: {stats.chi2_contingency(pd.crosstab(index=joined['Parch'], columns=joined['Survived']))[1]:.3f}")
print(
    f"Family Size: {stats.chi2_contingency(pd.crosstab(index=joined['FamilySize'], columns=joined['Survived']))[1]:.3f}")
sns.catplot(data=joined, x='Fare', y='Embarked', kind='box', aspect=2);
cat_features = [
    'Sex', 'Pclass', 'AgeG', 'Parch', 'SibSp', 'Side', 'Deck', 'FamilySize', 'Title', 'IsAlone', 'Embarked'
]
num_features = ['Age', 'Fare']

train_data = Pool(
    data=FeaturesData(
        num_feature_data=joined.loc[:891, num_features].astype(np.float32).values,
        num_feature_names=num_features,
        cat_feature_data=joined.loc[:891, cat_features].astype(np.unicode).values,
        cat_feature_names=cat_features),
    label=joined.loc[:891, 'Survived'].values)
model = CatBoostClassifier(task_type='CPU', learning_rate=0.1, depth=4, iterations=5000)
model.fit(train_data, verbose=False);
pd.Series(model.feature_importances_, index=model.feature_names_).sort_values().plot.barh(figsize=(10, 5));
