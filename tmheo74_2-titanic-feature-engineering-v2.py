import random
import re
from datetime import datetime

# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Model
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

pd.options.display.max_rows = 100

from IPython.display import display

%matplotlib inline
# Set a few plotting defaults
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 14
plt.rcParams['patch.edgecolor'] = 'k'
train_raw = pd.read_csv('../input/train.csv')
test_raw = pd.read_csv('../input/test.csv')
train_raw.shape, test_raw.shape
train_raw.info()
test_raw.info()
def plot_distribution_by_target(df, field):
    df = df[df[field].notnull()]

    fig = plt.figure(figsize = (14, 12))
    ax1 = plt.subplot(221)
    
    sns.kdeplot(df[field], label='Total', alpha=0.7, ax=ax1)
    sns.kdeplot(df[df.train == 1][field], label='Train', alpha=0.7, ax=ax1)
    sns.kdeplot(df[df.train == 0][field], label='Test', alpha=0.7, ax=ax1)

    plt.xlabel(field.upper())
    plt.ylabel('Density')
    
    ax2 = plt.subplot(222)

    sns.boxplot(x='train', y=field, data=df, ax=ax2)
    plt.xticks((0,1), ('Test','Train'))
    
    df = df[df.train == 1]
    
    ax3 = plt.subplot(223)

    sns.kdeplot(df[df.Survived == 1][field], label='Survived', alpha=0.7, ax=ax3)
    sns.kdeplot(df[df.Survived == 0][field], label='Not Survived', alpha=0.7, ax=ax3)

    plt.xlabel(field.upper())
    plt.ylabel('Density')

    ax4 = plt.subplot(224)

    sns.boxplot(x='Survived', y=field, data=df, ax=ax4)
    plt.xticks((0,1), ('Not Survived','Survived'))
    
    fig.suptitle(f'{field.upper()} Distribution', fontsize=20)
    
    plt.show()
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
base_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X_train = train_raw[base_features]
X_test = test_raw[base_features]
y_train = train_raw.Survived

X_train.Embarked = X_train.Embarked.fillna(X_train.Embarked.mode()[0])
X_test.Embarked = X_test.Embarked.fillna(X_train.Embarked.mode()[0])

le = LabelEncoder()
X_train.Sex = le.fit_transform(X_train[['Sex']])
X_test.Sex = le.transform(X_test[['Sex']])

X_train.Embarked = le.fit_transform(X_train[['Embarked']])
X_test.Embarked = le.transform(X_test[['Embarked']])

median_imputer = SimpleImputer(strategy='median')
X_train.Age = median_imputer.fit_transform(X_train[['Age']])
X_test.Age = median_imputer.transform(X_test[['Age']])

X_train.Fare = median_imputer.fit_transform(X_train[['Fare']])
X_test.Fare = median_imputer.transform(X_test[['Fare']])

print(f'X_train shape : {X_train[base_features].shape}, X_test shape : {X_test[base_features].shape}')
def cv_model(train, train_labels, model, name, model_results=None, cv=10, scoring='accuracy'):
    """Perform k fold cross validation of a model"""
    
    cv_scores = cross_val_score(model, train, train_labels, cv=cv, scoring=scoring, n_jobs=-1)
    print(f'{cv} Fold CV {scoring} for {name}: {round(cv_scores.mean(), 5)} with std: {round(cv_scores.std(), 5)}')
    
    model.fit(train, train_labels)
    
    if model_results is None:
        model_results = pd.DataFrame({
            'model': name,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            #'test_score': test_score
        }, index = [0])
    else:
        model_results = model_results.append(pd.DataFrame({
            'model': name,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            #'test_score': test_score
        }, index = [0]), ignore_index = True)

    return model_results, model

def show_model_results(model_results):
    display(model_results)
    df = model_results.copy().set_index('model')
    df.sort_values(by='cv_mean', inplace=True)
    df['cv_mean'].plot.bar(color = 'orange', figsize = (10, 8),
                                      yerr = list(df['cv_std']),
                                      edgecolor = 'k', linewidth = 2)
    plt.title('Model Train CV Accuracy Score Results');
    plt.ylabel('Mean CV Accuracy Score (with error bar)');
    plt.show()

def plot_feature_importances(estimator, x_cols, n=20, threshold = 0.95):
    try:
        df = pd.DataFrame({'feature': x_cols, 'importance': estimator.feature_importances_})
    except AttributeError:
        print('model does not provide feature importances')
        return
    
    # Sort features with most important at the head
    df = df.sort_values('importance', ascending = False).reset_index(drop = True)
    
    # Normalize the feature importances to add up to one and calculate cumulative importance
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    
    plt.rcParams['font.size'] = 12
    
    # Bar plot of n most important features
    df.loc[:n, :].plot.barh(y = 'importance_normalized', 
                            x = 'feature', color = 'darkgreen', 
                            edgecolor = 'k', figsize = (12, 8),
                            legend = False, linewidth = 2)

    plt.xlabel('Normalized Importance', size = 18); plt.ylabel(''); 
    plt.title(f'{min(n, len(df))} Most Important Features', size = 18)
    plt.gca().invert_yaxis()
    
    if threshold:
        # Cumulative importance plot
        plt.figure(figsize = (8, 6))
        plt.plot(list(range(1, len(df)+1)), df['cumulative_importance'], 'b-')
        plt.xlabel('Number of Features', size = 16); plt.ylabel('Cumulative Importance', size = 16); 
        plt.title('Cumulative Feature Importance', size = 18);
        
        # Number of features needed for threshold cumulative importance
        # This is the index (will need to add 1 for the actual number)
        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
        
        # Add vertical line to plot
        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.05, linestyles = '--', colors = 'red')
        plt.show();
        
        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 
                                                                                  100 * threshold))
    
    print(f'zero importance feature count : {len(df[df.importance == 0])}')
    
    return df

# Dataframe to hold results
model_results = pd.DataFrame(columns = ['model', 'cv_mean', 'cv_std'])

model_results, xgb_base = cv_model(X_train[base_features], y_train, xgb.XGBClassifier(random_state=RANDOM_SEED),
                            'XGB_Base', model_results)

plot_feature_importances(xgb_base, base_features)
train_raw = pd.read_csv('../input/train.csv')
test_raw = pd.read_csv('../input/test.csv')

train_raw.Embarked = train_raw.Embarked.fillna(train_raw.Embarked.mode()[0])
test_raw.Embarked = test_raw.Embarked.fillna(test_raw.Embarked.mode()[0])

le = LabelEncoder()
train_raw.Sex = le.fit_transform(train_raw[['Sex']])
test_raw.Sex = le.transform(test_raw[['Sex']])

train_raw.Embarked = le.fit_transform(train_raw[['Embarked']])
test_raw.Embarked = le.transform(test_raw[['Embarked']])

train_raw['train'] = 1
test_raw['train'] = 0
data_all = pd.concat([train_raw, test_raw], axis=0).reset_index(drop=True)
data_all.info()
data_all['family_size'] = data_all.SibSp + data_all.Parch + 1
def plot_by_target(df, field, col_wrap=4):
    df = df[df.train == 1]
    g = sns.catplot('Survived', col=field, data=df, kind='count', col_wrap=col_wrap)
    plt.show()
plot_by_target(data_all, 'family_size', col_wrap=3)
def calc_family_size_bin(family_size):
    if family_size == 1:
        return 0
    elif family_size <= 4: 
        return 1
    else:
        return 2
        
data_all['family_size_bin'] = data_all.family_size.map(calc_family_size_bin)
plot_by_target(data_all, 'family_size_bin')
data_all['name_title'] = data_all.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
data_all.name_title.value_counts().plot.bar()
plt.title('Name Title Count')
plt.show()
pd.crosstab(data_all.Sex, data_all.name_title)
data_all.groupby('name_title').Age.median()
data_all[data_all.train == 1].groupby('name_title').Survived.mean()
name_title_dict = {
    'Capt': 'Mr',
    'Col': 'Mr',
    'Don': 'Mr',
    'Dona': 'Mrs',    
    'Dr': 'Dr',
    'Jonkheer': 'Mr',
    'Lady': 'Mrs',
    'Major': 'Mr',
    'Master': 'Master',
    'Miss': 'Miss',
    'Mlle': 'Miss',
    'Mme': 'Miss',
    'Mr': 'Mr',
    'Mrs': 'Mrs',
    'Ms': 'Mrs',
    'Rev': 'Mr',
    'Sir': 'Mr',
    'Countess': 'Mrs'
}

data_all['name_title_cat'] = data_all.name_title.map(name_title_dict)
plot_by_target(data_all, 'name_title_cat', col_wrap=3)
data_all['last_name'] = data_all.Name.str.extract('([A-Za-z]+),', expand=False)
data_all['last_name_family_size'] = data_all.apply(lambda row: row.last_name + '_' + str(row.family_size), axis=1)
data_all['last_name_ticket'] = data_all.apply(lambda row: row.last_name + '_' + row.Ticket, axis=1)
ticket_df = data_all.groupby('Ticket', as_index=False)['PassengerId'].count()
ticket_df.columns = ['Ticket','ticket_count']
ticket_df.head()
data_all = pd.merge(data_all, ticket_df, on=['Ticket'])
data_all = data_all.sort_values('PassengerId').reset_index(drop=True)
last_name_family_size_check = data_all[data_all.family_size > 1].groupby('last_name_family_size').agg({'Survived': lambda x: x.isnull().sum()}).reset_index()
last_name_family_size_check.columns = ['last_name_family_size','last_name_family_size_feature']
last_name_family_size_check.head()
last_name_ticket_check = data_all[data_all.ticket_count > 1].groupby('last_name_ticket').agg({'Survived': lambda x: x.isnull().sum()}).reset_index()
last_name_ticket_check.columns = ['last_name_ticket','last_name_ticket_feature']
last_name_ticket_check.head()
data_all = pd.merge(data_all, last_name_family_size_check, on='last_name_family_size', how='left')
data_all = data_all.sort_values('PassengerId').reset_index(drop=True)
data_all.last_name_family_size_feature = data_all.last_name_family_size_feature.fillna(0)
data_all.head()
data_all = pd.merge(data_all, last_name_ticket_check, on='last_name_ticket', how='left')
data_all = data_all.sort_values('PassengerId').reset_index(drop=True)
data_all.last_name_ticket_feature = data_all.last_name_ticket_feature.fillna(0)
data_all.head()
data_all.loc[data_all.last_name_family_size_feature == 0, 'last_name_family_size'] = 'X'
data_all.loc[data_all.last_name_ticket_feature == 0, 'last_name_ticket'] = 'X'
data_all[['last_name_family_size','Sex','Age','last_name','family_size','Name','Ticket',
          'Survived']][data_all.last_name_family_size == 'X'].sort_values(['last_name','family_size']).head()
data_all[['last_name_ticket','Sex','Age','last_name','ticket_count','Name','Ticket',
          'Survived']][data_all.last_name_ticket == 'X'].sort_values(['last_name','Ticket']).head()
data_all[['last_name_family_size','Sex','Age','last_name','family_size','Name','Ticket',
          'last_name_family_size_feature',
          'Survived']][data_all.last_name_family_size != 'X'].sort_values(['last_name','family_size']).head(9)
data_all[['last_name_ticket','Sex','Age','last_name','ticket_count','Name','Ticket',
          'last_name_ticket_feature',
          'Survived']][data_all.last_name_ticket != 'X'].sort_values(['last_name','Ticket']).head()
family_survival = data_all.groupby(['last_name_family_size','family_size']).Survived.sum().reset_index()
family_survival.columns = ['last_name_family_size','family_size','family_survival_sum']
family_survival.head()
family_ticket_count = data_all.groupby(['last_name','Ticket']).PassengerId.count().reset_index()
family_ticket_count.columns = ['last_name','Ticket','family_ticket_count']

family_ticket_survival = data_all.groupby(['last_name','Ticket']).Survived.sum().reset_index()
family_ticket_survival.columns = ['last_name','Ticket','family_ticket_survival_sum']

family_ticket_survival = pd.merge(family_ticket_count, family_ticket_survival, on=['last_name','Ticket'])
family_ticket_survival.head()
def calc_family_survival(row):
    family_survival = 0.5
    if row['family_size'] > 1 and row['family_survival_sum'] > 0:
        family_survival = 1
    elif row['family_size'] > 1 and row['family_survival_sum'] == 0:
        family_survival = 0
        
    return family_survival

family_survival['family_survival'] = family_survival.apply(calc_family_survival, axis=1)
family_survival[family_survival['family_size'] > 1].head()
def calc_family_ticket_survival(row):
    family_ticket_survival = 0.5
    if row['family_ticket_count'] > 1 and row['family_ticket_survival_sum'] > 0:
        family_ticket_survival = 1
    elif row['family_ticket_count'] > 1 and row['family_ticket_survival_sum'] == 0:
        family_ticket_survival = 0
        
    return family_ticket_survival

family_ticket_survival['family_ticket_survival'] = family_ticket_survival.apply(calc_family_ticket_survival,
                                                                                axis=1)
family_ticket_survival[family_ticket_survival['family_ticket_count'] > 1].head()
data_all = pd.merge(data_all, family_survival, on=['last_name_family_size','family_size'], how='left')
data_all = data_all.sort_values('PassengerId').reset_index(drop=True)
data_all = pd.merge(data_all, family_ticket_survival, on=['last_name','Ticket'], how='left')
data_all = data_all.sort_values('PassengerId').reset_index(drop=True)
plot_by_target(data_all, 'family_survival')
plot_by_target(data_all, 'family_ticket_survival')
plot_by_target(data_all[data_all.family_ticket_count <= 2], 'family_ticket_count', col_wrap=4)
plot_by_target(data_all[data_all.family_ticket_count > 2], 'family_ticket_count')
def calc_family_ticket_count_bin(family_ticket_count):
    if family_ticket_count == 1:
        return 0
    elif family_ticket_count <= 4: 
        return 1
    else:
        return 2
        
data_all['family_ticket_count_bin'] = data_all.family_ticket_count.map(calc_family_ticket_count_bin)
plot_by_target(data_all, 'family_ticket_count_bin')
data_all.info()
fe_1_features = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
    'family_size', 'family_size_bin', 'name_title_cat', 
    'last_name_family_size', 'last_name_ticket',
    'family_survival', 'family_ticket_survival',
    'family_ticket_count', 'family_ticket_count_bin'
]

# label encoding
le = LabelEncoder()

data_all.name_title_cat = le.fit_transform(data_all[['name_title_cat']])
data_all.last_name_family_size = le.fit_transform(data_all[['last_name_family_size']])
data_all.last_name_ticket = le.fit_transform(data_all[['last_name_ticket']])

X_train = data_all[data_all.train == 1][fe_1_features]
X_test = data_all[data_all.train == 0][fe_1_features]

print(f'X_train shape : {X_train.shape}, X_test shape : {X_test.shape}')
def make_prediction(train, target, test, features, model_name, model_results=None,
                    model=xgb.XGBClassifier(random_state=RANDOM_SEED)):
    model_results, model = cv_model(train, target, model, 
                                         model_name, model_results)

    show_model_results(model_results)
    fi = plot_feature_importances(model, features)
    display(fi)
    
    model.fit(train, target)
    pred = model.predict(test)

    output = f'{model_name}_submission_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
    submit_df = pd.DataFrame()
    submit_df['PassengerId'] = test_raw.PassengerId
    submit_df['Survived'] = pred
    
    submit_df[['PassengerId','Survived']].to_csv(output, index=False)
    print(f'submission file {output} is generated.')
    
    return model_results
model_results = make_prediction(X_train, y_train, X_test, fe_1_features, 'XGB_FE_1', model_results)
data_all['fare_fixed'] = data_all.Fare/data_all.ticket_count
data_all[['Ticket','Fare','fare_fixed','ticket_count','Pclass']][data_all.ticket_count > 1].sort_values('Ticket').head(6)
fare_median = data_all.fare_fixed.median()
data_all['fare_fixed'] = data_all.fare_fixed.fillna(fare_median)
plot_distribution_by_target(data_all, 'fare_fixed')
data_all['fare_fixed_log'] = np.log1p(data_all.fare_fixed)
plot_distribution_by_target(data_all, 'fare_fixed_log')
age_median_by_sex_title = data_all.groupby(['Sex', 'name_title'], as_index=False).Age.median()
age_median_by_sex_title
data_all = pd.merge(data_all, age_median_by_sex_title, on=['Sex', 'name_title'])
data_all['Age'] = data_all.apply(lambda row: row.Age_x if not np.isnan(row.Age_x) else row.Age_y, axis=1)
data_all = data_all.drop(['Age_x','Age_y'], axis=1).sort_values('PassengerId').reset_index(drop=True)
data_all.info()
plot_distribution_by_target(data_all, 'Age')
def calc_age_bin(age):
    if age <= 15:
        return 0
    elif age <= 30:
        return 1
    elif age <= 60:
        return 2
    else:
        return 3
        
data_all['age_bin'] = data_all.Age.map(calc_age_bin)
fe_2_features = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
    'family_size', 'family_size_bin', 'name_title_cat', 
    'last_name_family_size', 'last_name_ticket',
    'family_survival', 'family_ticket_survival',
    'family_ticket_count', 'family_ticket_count_bin',
    'fare_fixed_log', 'age_bin',
]


X_train = data_all[data_all.train == 1][fe_2_features]
X_test = data_all[data_all.train == 0][fe_2_features]

print(f'X_train shape : {X_train.shape}, X_test shape : {X_test.shape}')
model_results = make_prediction(X_train, y_train, X_test, fe_2_features, 'XGB_FE_2', model_results)
def parse_ticket_str(ticket):
    arr = ticket.split()
    if not arr[0].isdigit():
        txt = arr[0].replace('.', '')
        txt = txt.split('/')[0]
        return re.findall('[a-zA-Z]+', txt)[0]
    else:
        return None
        
data_all['ticket_str'] = data_all.Ticket.map(parse_ticket_str)
plot_by_target(data_all, 'ticket_str')
g = sns.catplot(x="Pclass", y="fare_fixed_log", col="ticket_str", hue='Survived',
                col_wrap=4, data=data_all, kind="strip")
plt.show()
def parse_ticket_number(ticket):
    arr = ticket.split()
    if len(arr) == 1 and arr[0].isdigit():
        return int(arr[0])
    elif len(arr) == 2 and arr[1].isdigit():
        return int(arr[1])
    else:
        if arr[-1].isdigit():
            return int(arr[-1])
        else:
            return np.nan
    
data_all['ticket_number'] = data_all.Ticket.map(parse_ticket_number)
def parse_ticket_num_len(ticket):
    arr = ticket.split()
    if len(arr) == 1 and arr[0].isdigit():
        return len(arr[0])
    elif len(arr) == 2 and arr[1].isdigit():
        return len(arr[1])
    else:
        if arr[-1].isdigit():
            return len(arr[-1])
        else:
            return -1
    
data_all['ticket_num_len'] = data_all.Ticket.map(parse_ticket_num_len)
plot_by_target(data_all, 'ticket_num_len')
g = sns.catplot(x="Pclass", y="fare_fixed_log", col="ticket_num_len", hue='Survived',
                col_wrap=4, data=data_all, kind="strip")
plt.show()
data_all['ticket_num_len_4_prefix'] = data_all[data_all.ticket_num_len == 4].ticket_number.map(lambda x: int(str(x)[0]))
plot_by_target(data_all, 'ticket_num_len_4_prefix', col_wrap=3)
g = sns.catplot(x="Pclass", y="fare_fixed_log", col="ticket_num_len_4_prefix", hue='Survived',
                col_wrap=3, data=data_all, kind="strip")
plt.show()
data_all['ticket_num_len_4_prefix_2'] = data_all[data_all.ticket_num_len == 4].ticket_number.map(lambda x: int(str(x)[:2]))
plot_by_target(data_all, 'ticket_num_len_4_prefix_2')
g = sns.catplot(x="Pclass", y="fare_fixed_log", col="ticket_num_len_4_prefix_2", hue='Survived',
                col_wrap=4, data=data_all, kind="strip")
plt.show()
data_all['ticket_num_len_5_prefix'] = data_all[data_all.ticket_num_len == 5].ticket_number.map(lambda x: int(str(x)[0]))
plot_by_target(data_all, 'ticket_num_len_5_prefix', col_wrap=3)
g = sns.catplot(x="Pclass", y="fare_fixed_log", col="ticket_num_len_5_prefix", hue='Survived',
                col_wrap=3, data=data_all, kind="strip")
plt.show()
data_all['ticket_num_len_5_prefix_2'] = data_all[data_all.ticket_num_len == 5].ticket_number.map(lambda x: int(str(x)[:2]))
plot_by_target(data_all, 'ticket_num_len_5_prefix_2')
g = sns.catplot(x="Pclass", y="fare_fixed_log", col="ticket_num_len_5_prefix_2", hue='Survived',
                col_wrap=4, data=data_all, kind="strip")
plt.show()
data_all['ticket_num_len_6_prefix'] = data_all[data_all.ticket_num_len == 6].ticket_number.map(lambda x: int(str(x)[0]))
plot_by_target(data_all, 'ticket_num_len_6_prefix', col_wrap=3)
g = sns.catplot(x="Pclass", y="fare_fixed_log", col="ticket_num_len_6_prefix", hue='Survived',
                col_wrap=3, data=data_all, kind="strip")
plt.show()
data_all['ticket_num_len_6_prefix_2'] = data_all[data_all.ticket_num_len == 6].ticket_number.map(lambda x: int(str(x)[:2]))
plot_by_target(data_all, 'ticket_num_len_6_prefix_2')
g = sns.catplot(x="Pclass", y="fare_fixed_log", col="ticket_num_len_6_prefix_2", hue='Survived',
                col_wrap=4, data=data_all, kind="strip")
plt.show()
data_all['ticket_num_len_6_prefix_3'] = data_all[data_all.ticket_num_len == 6].ticket_number.map(lambda x: int(str(x)[:3]))
plot_by_target(data_all, 'ticket_num_len_6_prefix_3', col_wrap=5)
g = sns.catplot(x="Pclass", y="fare_fixed_log", col="ticket_num_len_6_prefix_3", hue='Survived',
                col_wrap=4, data=data_all, kind="strip")
plt.show()
data_all[data_all.ticket_num_len_6_prefix == 1].groupby(['ticket_num_len_6_prefix_3','train'])[['PassengerId','Survived']].agg(['mean','count'])
data_all[data_all.ticket_num_len_6_prefix == 3].groupby(['ticket_num_len_6_prefix_3','train'])[['PassengerId','Survived']].agg(['mean','count'])
data_all.info()
fe_3_features = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
    'family_size', 'family_size_bin', 'name_title_cat', 
    'last_name_family_size', 'last_name_ticket',
    'family_survival', 'family_ticket_survival',
    'family_ticket_count', 'family_ticket_count_bin',
    'fare_fixed_log', 'age_bin',
    'ticket_str', 'ticket_num_len',
    'ticket_num_len_4_prefix', 'ticket_num_len_4_prefix_2',
    'ticket_num_len_5_prefix', 'ticket_num_len_5_prefix_2',
    'ticket_num_len_6_prefix', 'ticket_num_len_6_prefix_2', 'ticket_num_len_6_prefix_3',
]

# fill missing values
data_all.ticket_str = data_all.ticket_str.fillna('X')

data_all.ticket_num_len_4_prefix = data_all.ticket_num_len_4_prefix.fillna(-1)
data_all.ticket_num_len_4_prefix_2 = data_all.ticket_num_len_4_prefix_2.fillna(-1)
data_all.ticket_num_len_5_prefix = data_all.ticket_num_len_5_prefix.fillna(-1)
data_all.ticket_num_len_5_prefix_2 = data_all.ticket_num_len_5_prefix_2.fillna(-1)
data_all.ticket_num_len_6_prefix = data_all.ticket_num_len_6_prefix.fillna(-1)
data_all.ticket_num_len_6_prefix_2 = data_all.ticket_num_len_6_prefix_2.fillna(-1)
data_all.ticket_num_len_6_prefix_3 = data_all.ticket_num_len_6_prefix_3.fillna(-1)

data_all.ticket_num_len_4_prefix = data_all.ticket_num_len_4_prefix.astype(int)
data_all.ticket_num_len_4_prefix_2 = data_all.ticket_num_len_4_prefix_2.astype(int)
data_all.ticket_num_len_5_prefix = data_all.ticket_num_len_5_prefix.astype(int)
data_all.ticket_num_len_5_prefix_2 = data_all.ticket_num_len_5_prefix_2.astype(int)
data_all.ticket_num_len_6_prefix = data_all.ticket_num_len_6_prefix.astype(int)
data_all.ticket_num_len_6_prefix_2 = data_all.ticket_num_len_6_prefix_2.astype(int)
data_all.ticket_num_len_6_prefix_3 = data_all.ticket_num_len_6_prefix_3.astype(int)

# label encoding
le = LabelEncoder()

data_all.ticket_str = le.fit_transform(data_all[['ticket_str']])

X_train = data_all[data_all.train == 1][fe_3_features]
X_test = data_all[data_all.train == 0][fe_3_features]

print(f'X_train shape : {X_train.shape}, X_test shape : {X_test.shape}')
model_results = make_prediction(X_train, y_train, X_test, fe_3_features, 'XGB_FE_3', model_results)
data_all.Cabin.sort_values().unique()
data_all['cabin_cat'] = data_all.Cabin.fillna('X').str[0]
plot_by_target(data_all[~data_all.cabin_cat.isin(['X'])], 'cabin_cat')
def calc_cabin_len(cabin):
    if type(cabin) == float:
        return 0
    else:
        return len(cabin.split())

data_all['cabin_len'] = data_all.Cabin.map(calc_cabin_len)
plot_by_target(data_all[data_all.cabin_len > 0], 'cabin_len')
fe_4_features = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
    'family_size', 'family_size_bin', 'name_title_cat', 
    'last_name_family_size', 'last_name_ticket',
    'family_survival', 'family_ticket_survival',
    'family_ticket_count', 'family_ticket_count_bin',
    'fare_fixed_log', 'age_bin',
    'ticket_str', 'ticket_num_len',
    'ticket_num_len_4_prefix', 'ticket_num_len_4_prefix_2',
    'ticket_num_len_5_prefix', 'ticket_num_len_5_prefix_2',
    'ticket_num_len_6_prefix', 'ticket_num_len_6_prefix_2', 'ticket_num_len_6_prefix_3',
    'cabin_cat', 'cabin_len',
]

# label encoding
le = LabelEncoder()

data_all.cabin_cat = le.fit_transform(data_all[['cabin_cat']])

X_train = data_all[data_all.train == 1][fe_4_features]
X_test = data_all[data_all.train == 0][fe_4_features]

print(f'X_train shape : {X_train.shape}, X_test shape : {X_test.shape}')
model_results = make_prediction(X_train, y_train, X_test, fe_4_features, 'XGB_FE_4', model_results)
def plot_correlation_heatmap(df, variables):
    # Calculate the correlations
    corr_mat = df[variables].corr().round(2)

    # Draw a correlation heatmap
    plt.figure(figsize = (18, 16))
    sns.heatmap(corr_mat, vmin=-0.6, vmax=0.6, center=0, cmap='viridis', annot=True)
    plt.title('Feature Correlation Heatmap\n')
    plt.show()
plot_correlation_heatmap(data_all[data_all.train == 1], ['Survived'] + list(fe_4_features))
fe_4_sel_features = [
    'Pclass', 'Sex', 'Age', 
    'family_size_bin',
    'name_title_cat', 
    'last_name_family_size',
    'last_name_ticket',
    'family_ticket_survival',
    'fare_fixed_log',
    'ticket_str',
    'ticket_num_len_4_prefix_2',
    'ticket_num_len_5_prefix_2',
    'ticket_num_len_6_prefix_3',
]

plot_correlation_heatmap(data_all, ['Survived'] + list(fe_4_sel_features))
X_train = data_all[data_all.train == 1][fe_4_sel_features]
X_test = data_all[data_all.train == 0][fe_4_sel_features]

print(f'X_train shape : {X_train.shape}, X_test shape : {X_test.shape}')

model_results = make_prediction(X_train, y_train, X_test, fe_4_sel_features, 'XGB_FE_4_SEL', 
                                model_results=model_results)
