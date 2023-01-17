import pandas as pd

import numpy as np

import math

import re

import warnings



import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.ticker import FuncFormatter



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



import statsmodels.formula.api as sm

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression



from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score



sns.set(font_scale = 1.4)

warnings.simplefilter(action = 'ignore', category = FutureWarning)



g_colors = ['red', 'green']

g_sex = ['male', 'female']
train_raw = pd.read_csv('../input/titanic/train.csv')

train_raw.columns = [col.lower() for col in train_raw.columns]

train_raw.columns
train_raw.iloc[:, 2:].dtypes
train_raw.iloc[:, 2:].describe()
train_raw.iloc[:, 2:].head(10)
pd.DataFrame(dict(counts = train_raw['survived'].value_counts(), 

                  percent = train_raw['survived'].value_counts(normalize=True)))
fig, axes = plt.subplots(1, 6, sharey = True, figsize = (15, 3))



# Because we are skipping certain columns, we cannot use enumerate for a counter

k = 0



for col in train_raw.columns.values:

    if len(train_raw[col].unique()) < 10:

        ax_cur = axes.ravel()[k]

        sns.barplot(x = train_raw[col].value_counts(normalize = True).index,

                    y = train_raw[col].value_counts(normalize = True),

                    palette = "Dark2",

                    ax = ax_cur)

        ax_cur.set_title(col)

        ax_cur.set_ylabel('')

        

        k += 1
fig, axes = plt.subplots(1, 2, sharey = True, figsize = (15, 3))



sns.distplot(train_raw.loc[train_raw.age.notnull(), 'age'],

             ax = axes[0],

             color = '#632de9') #purple blue



sns.distplot(train_raw.loc[train_raw.fare.notnull(), 'fare'],

             ax = axes[1],

             color = '#76cd26') #apple green



axes[0].set_title('Age Distribution')



eat = axes[1].set_title('Fare Distribution')
fare_boundary = 110

fig, axes = plt.subplots(1, 2, figsize = (15, 3))



sns.distplot(train_raw.loc[train_raw.fare < fare_boundary, 'fare'],

             ax = axes[0],

             color = '#0cff0c') #neon green



sns.distplot(train_raw.loc[train_raw.fare >= fare_boundary, 'fare'],

             ax = axes[1],

             color = 'green')



axes[0].set_title('Fare < $' + str(fare_boundary))

eat = axes[1].set_title('Fare >= $' + str(fare_boundary))
train_raw[train_raw.fare > 400]
nans = [(col, sum(train_raw[col].isnull())) for col in train_raw.columns.values if sum(train_raw[col].isnull()) > 0]

pd.DataFrame(nans, columns = ['Column', 'NaNs'])
x_train, x_test, y_train, y_test = train_test_split(train_raw, train_raw.survived, random_state = 1)
fig, axes = plt.subplots(2, 3, figsize = (15, 8))

plt.subplots_adjust(hspace = .3)



for k, col in enumerate(['pclass', 'sex', 'embarked', 'sibsp', 'parch']):

    pd.crosstab(x_train['survived'], x_train[col]).T.plot(kind = 'bar',

                                                          color = g_colors,

                                                          ax = axes.ravel()[k],

                                                          rot = 0)

    

plt.suptitle('Survival Counts By Categorical Feature', fontsize = 20)

fig.delaxes(axes[1][2])
fig, axes = plt.subplots(2, 3, sharey = True, figsize = (15, 8))

plt.subplots_adjust(hspace = .3)



for k, col in enumerate(['pclass', 'sex', 'embarked', 'sibsp', 'parch']):



    d = pd.crosstab(train_raw['survived'], train_raw[col])



    colors = np.array([g_colors[c] for c in (d.iloc[0] < d.iloc[1]) * 1])



    axes.ravel()[k].bar(x = d.columns,

                        height = (d.iloc[1] - d.iloc[0]) / (d.iloc[1] + d.iloc[0]),

                        color = colors)

    

    axes.ravel()[k].set_title(col)

    axes.ravel()[k].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 

    

plt.suptitle('Survival Percentages By Categorical Feature', fontsize = 20)

fig.delaxes(axes[1][2])
fig, axes = plt.subplots(1, 1, figsize = (15, 4))

axes.set_xticks(range(0, 100, 10))



for i in range(0, 2):

    sns.distplot(x_train.loc[(x_train.age.notnull()) & (x_train.survived == i), 'age'],

                 color = g_colors[i],

                 hist = False,

                 ax = axes)



axes.set_title('Survival By Age')

z = axes.set_xlim(x_train.age.min(), x_train.age.max())
fig, axes = plt.subplots(1, 2, sharey = True, figsize = (15, 4))

axes[0].set_xticks(range(0, 100, 10))



for i in range(0, 4):

    var_survived = i % 2

    var_sex = math.floor(i / 2)

    

    sns.distplot(x_train.loc[(x_train.age.notnull()) & (x_train.sex == g_sex[var_sex]) & (x_train.survived == var_survived), 'age'],

                 color = g_colors[var_survived],

                 hist = False,

                 ax = axes[var_sex])

    

    axes[var_sex].set_title(g_sex[var_sex].title() + ' Survival By Age')

    

    axes[var_sex].set_xlim(x_train.loc[x_train.sex == g_sex[var_sex], 'age'].min(), x_train.loc[x_train.sex == g_sex[var_sex], 'age'].max())
fig, axes = plt.subplots(1, 1, sharex = True, sharey = True, figsize = (15, 5))



for i in range(0, 2):

    sns.distplot(x_train.loc[(x_train.fare.notnull()) & (x_train.survived == i), 'fare'],

                 color = g_colors[i],

                 hist = False,

                 ax = axes)



axes.set_title('Survival By Fare')

z = axes.set_xlim(0, x_train.fare.max())
fare_lt = x_train[(x_train.fare.notnull()) & (x_train.fare < 50)]



fig, axes = plt.subplots(1, 1, figsize = (15, 5))



for i in range(0,2):

    sns.distplot(fare_lt.loc[(fare_lt.survived == i), 'fare'],

                 color = g_colors[i],

                 hist = False)



axes.set_xlim(fare_lt.fare.min(), fare_lt.fare.max())

z = axes.set_title('Survival for Fares Less Than $50')
fare_gt = x_train[(x_train.fare.notnull()) & (x_train.fare >= 50)]



fig, axes = plt.subplots(1, 1, figsize = (15, 5))



for i in range(0, 2):

    sns.distplot(fare_gt.loc[(fare_gt.survived == i), 'fare'],

                 color = g_colors[i],

                 hist = False)



axes.set_xlim(fare_gt.fare.min(), fare_gt.fare.max())

z = axes.set_title('Survival for Fares Greater Than $50')
fares = x_train[(x_train.fare.notnull()) & (x_train.fare > 0)].copy()



fares['fare_log'] = np.log(fares.fare)



fig, axes = plt.subplots(1, 1, figsize = (15, 5))



for i in range(0, 2):

    sns.distplot(fares.loc[(fares.survived == i), 'fare_log'],

                 color = g_colors[i],

                 hist = False)



z = axes.set_title('Survival By Log Transformation Of Fare')
fig = plt.figure(figsize = (15, 8))

cols = ['pclass', 'age', 'fare', 'survived']

colors = g_colors

pd.plotting.scatter_matrix(train_raw[cols],

                           figsize=[13,13],

                           alpha=0.2,

                           c = train_raw.survived.apply(lambda x:colors[x]))



plt.tight_layout()
def show_dot_plot(p_x, p_y, p_ax):

    

    sns.stripplot(x = p_x,

                  y = p_y,

                  data = x_train,

                  jitter = True,

                  alpha = .7,

                  hue = 'survived',

                  palette = g_colors,

                  ax = p_ax)

    

    return



fig, axes = plt.subplots(2, 3, figsize=(15,8))



cols = ['pclass', 'sex', 'embarked', 'sibsp', 'parch']



for k, col in enumerate(cols):

    show_dot_plot(p_x = col,

                  p_y = 'age',

                  p_ax = axes.ravel()[k])

    

fig.delaxes(axes[1][2])

z = plt.suptitle('Features By Age', fontsize = 24)
def show_dot_plot(p_x, p_y, p_ax):

    

    sns.stripplot(x = p_x,

                  y = p_y,

                  data = x_train[x_train.fare < 100],

                  jitter = True,

                  alpha = .7,

                  hue = 'survived',

                  palette = g_colors,

                  ax = p_ax)

    

    return



fig, axes = plt.subplots(2, 3, figsize=(15,8))



cols = ['pclass', 'sex', 'embarked', 'sibsp', 'parch']



for k, col in enumerate(cols):

    show_dot_plot(p_x = col,

                  p_y = 'fare',

                  p_ax = axes.ravel()[k])



fig.delaxes(axes[1][2])

    

z = plt.suptitle('Features By Fare (Less Than $100)', fontsize = 24)
eat = sns.factorplot(x = "pclass",

                     y = "survived",

                     data = x_train,

                     hue = 'sex',

                     col = 'embarked',

                     palette = [sns.xkcd_rgb['medium blue'], sns.xkcd_rgb['carmine']])
def tidy_data(df):

    data = df.copy()

    

    # Perform all data cleaning / feature engineering against data

    

    return data
x_train_tidy = x_train.copy()
x_train_tidy['name_length'] = x_train_tidy.name.str.len()



fig, axes = plt.subplots(1, 1, figsize = (15, 4))

axes.set_xticks(range(0, 100, 10))



for i in range(0, 2):

    sns.distplot(x_train_tidy.loc[(x_train.survived == i), 'name_length'],

                 color = g_colors[i],

                 hist = False,

                 ax = axes)



axes.set_title('Survival By Name Length')

eat = axes.set_xlim(x_train_tidy.name_length.min(), x_train_tidy.name_length.max())
x_train_tidy['title'] = x_train_tidy.name.str.extract(' ([A-Za-z]+)\.').str.lower()

pd.DataFrame(x_train_tidy.title.value_counts())
mrs = ['mme', 'lady', 'countess']

miss = ['mlle']

rare = ['dr', 'rev', 'col', 'capt', 'major', 'don']



x_train_tidy.loc[x_train_tidy.title.isin(mrs), 'title'] = 'mrs'

x_train_tidy.loc[x_train_tidy.title.isin(miss), 'title'] = 'miss'

x_train_tidy.loc[x_train_tidy.title.isin(rare), 'title'] = 'rare'



x_train_tidy.loc[~x_train_tidy.title.isin(['mr', 'mrs', 'miss', 'master', 'rare']), 'title'] = ''



pd.DataFrame(x_train_tidy.title.value_counts())
fig, axes = plt.subplots(1, 1, figsize=(12,6))



g = sns.stripplot(x = 'title',

                  y = 'age',

                  data = x_train_tidy,

                  jitter = True,

                  alpha = .7,

                  hue = 'survived',

                  palette = g_colors)



eat = g.set_title('Age By Title')
x_train_tidy['cabin_floor'] = x_train_tidy.cabin.str.replace('[0-9]| ', '').str.get(0).str.lower()

x_train_tidy.loc[x_train_tidy.cabin_floor.isnull(), 'cabin_floor'] = 'z'



fig, axes = plt.subplots(1, 1, figsize=(13,6))



g = sns.stripplot(x = 'cabin_floor',

                  y = 'fare',

                  data = x_train_tidy,

                  jitter = True,

                  alpha = .7,

                  hue = 'survived',

                  palette = g_colors)



g.set_ylim(0, 300)



eat = g.set_title('Fare By Cabin Floor (High Fare Outliers Excluded)')
x_train_tidy['ticket_alpha'] = x_train_tidy.ticket.str.extract('([A-Za-z\.\/]+)').str.replace('\.', '').str.lower()

x_train_tidy['ticket_num'] = x_train_tidy.ticket.str.extract('([0-9\.\/]+)').str.replace('\.', '')

pd.DataFrame(x_train_tidy.ticket_alpha.value_counts())
fig, axes = plt.subplots(1, 1, figsize=(13,6))



top_tickets = x_train_tidy.ticket_alpha.value_counts().index.values[:10]



g = sns.stripplot(x = 'ticket_alpha',

                  y = 'fare',

                  data = x_train_tidy.loc[x_train_tidy.ticket_alpha.isin(top_tickets)],

                  jitter = True,

                  alpha = .7,

                  hue = 'survived',

                  palette = g_colors)



g.set_ylim(0, 300)



eat = g.set_title('Fare By Ticket (High fare outliers excluded)')
encode = pd.get_dummies(x_train_tidy['ticket_alpha'], prefix = 'ticket')

test = pd.concat([x_train_tidy, encode], axis = 1)

test_corr = test.corr().survived

pd.DataFrame(test_corr[[x for x in test_corr.index.values if 'ticket_' in x]].sort_values())
x_train_tidy['fam_size'] = x_train_tidy.sibsp + x_train_tidy.parch + 1

x_train_tidy['fam_size_large'] = (x_train_tidy.fam_size > 3) * 1
x_train_tidy.loc[x_train_tidy.embarked.isnull(), 'embarked'] = 'S'

x_train_tidy['embarked'] = x_train_tidy.embarked.str.lower()
x_train_tidy.columns
def tidy_data(df):

    data = df.copy()

    

    data['name_length'] = data.name.str.len()

    data['title'] = data.name.str.extract(' ([A-Za-z]+)\.')

    mr = ['Rev', 'Dr', 'Col', 'Capt', 'Don', 'Major']

    mrs = ['Mme', 'Countess', 'Lady']

    miss = ['Mlle']



    data.loc[data.title.isin(mr), 'title'] = 'mr'

    data.loc[data.title.isin(mrs), 'title'] = 'mrs'

    data.loc[data.title.isin(miss), 'title'] = 'miss'

    data.loc[data.title=='Master', 'title'] = 'master'

    data.loc[~data.title.isin(['mr', 'mrs', 'miss', 'master']), 'title'] = ''



    data['cabin_floor'] = data.cabin.str.replace('[0-9]| ', '').str.get(0).str.lower()

    data.loc[data.cabin_floor.isnull(), 'cabin_floor'] = 'z'



    data['ticket_alpha'] = data.ticket.str.extract('([A-Za-z\.\/]+)').str.replace('\.', '').str.lower()

    data['ticket_num'] = data.ticket.str.extract('([0-9\.\/]+)').str.replace('\.', '')



    data['fam_size'] = data.sibsp + data.parch + 1



    data.loc[data.embarked.isnull(), 'embarked'] = 'S'

    data['embarked'] = data.embarked.str.lower()



    for col in ['pclass', 'sex', 'sibsp', 'parch', 'embarked', 'title', 'cabin_floor', 'fam_size']:

        encode = pd.get_dummies(data[col], prefix = col)

        data = pd.concat([data, encode], axis = 1)

    

    return data
x_train_tidy = tidy_data(x_train)

x_train_tidy.columns
age_corr = x_train_tidy.corr().age

vals = age_corr[[x for x in age_corr.index.values if x != 'age']]

pd.DataFrame(vals[abs(vals)>.25].sort_values())
def model_age(df):

    formula = 'age ~ title_master + pclass_3 + parch_2 + parch_0 + pclass_1'



    model = sm.ols(formula, data = x_train_tidy[x_train_tidy.age.notnull()]).fit()



    age_pred = model.predict(df)

    

    return np.ceil(age_pred)
def model_age_correct(df):

    dt = x_train.copy()

    

    dt['title_master'] = dt.name.str.contains('Master') * 1

    for col in ['pclass', 'parch']:

        encode = pd.get_dummies(dt[col], prefix = col)

        dt = pd.concat([dt, encode], axis = 1)



    formula = 'age ~ title_master + pclass_3 + parch_2 + parch_0 + pclass_1'



    model = sm.ols(formula, data = dt[dt.age.notnull()]).fit()



    age_pred = model.predict(df)

    

    return np.ceil(age_pred)
x_train_tidy['age_gen'] = x_train_tidy.age

x_train_tidy.loc[x_train_tidy.age_gen.isnull(), 'age_gen'] = model_age_correct(x_train_tidy[x_train_tidy.age_gen.isnull()])
fig, axes = plt.subplots(1, 2, sharey = True, figsize = (15, 4))

axes[0].set_xticks(range(0, 100, 10))



for i in range(0, 2):

    var_survived = i % 2

    

    sns.distplot(x_train_tidy.loc[(x_train_tidy.age.notnull()) & (x_train_tidy.survived == var_survived), 'age'], 

                 color = g_colors[var_survived], 

                 hist = False, 

                 ax = axes[0])



    sns.distplot(x_train_tidy.loc[(x_train_tidy.survived == var_survived), 'age_gen'], 

                 color = g_colors[var_survived], 

                 hist = False, 

                 ax = axes[1])



axes[0].set_title('Survival By Age')

axes[1].set_title('Survival By Age (Derived)')

axes[0].set_xlim(x_train_tidy.age.min(), x_train_tidy.age.max())

eat = axes[1].set_xlim(x_train_tidy.age_gen.min(), x_train_tidy.age_gen.max())
bins = [0, 14, 32, 99]



x_train_tidy['age_bin'] = pd.cut(x_train_tidy['age_gen'], bins, labels = ['age_0_14', 'age_14_32', 'age_32_99'])

enc_age = pd.get_dummies(x_train_tidy.age_bin)

x_train_tidy = pd.concat([x_train_tidy, enc_age], axis = 1)
age_corr = x_train_tidy.corr().fare

vals = age_corr[[x for x in age_corr.index.values if x != 'fare']]

pd.DataFrame(vals[abs(vals) > .2].sort_values())
def model_fare(df):

    data = x_train.copy()

    

    data['cabin_floor'] = data.cabin.str.replace('[0-9]| ', '').str.get(0).str.lower()

    data.loc[data.cabin_floor.isnull(), 'cabin_floor'] = 'z'

    

    data['is_alone'] = ((data.sibsp + data.parch) == 0) * 1

    

    for col in ['pclass', 'cabin_floor', 'parch']:

        encode = pd.get_dummies(data[col], prefix = col)

        data = pd.concat([data, encode], axis = 1)



    formula = 'fare ~ pclass_1 + pclass_3 + is_alone + cabin_floor_b + cabin_floor_c + cabin_floor_z'



    model = sm.ols(formula, data = data[data.fare.notnull()]).fit()



    fare_pred = model.predict(df)

    

    return round(fare_pred, 2)
def tidy_data(df):

    data = df.copy()

    

    data['name_length'] = data.name.str.len()

    data['title'] = data.name.str.extract(' ([A-Za-z]+)\.')

    mr = ['Rev', 'Dr', 'Col', 'Capt', 'Don', 'Major']

    mrs = ['Mme', 'Countess', 'Lady']

    miss = ['Mlle']



    data.loc[data.title.isin(mr), 'title'] = 'mr'

    data.loc[data.title.isin(mrs), 'title'] = 'mrs'

    data.loc[data.title.isin(miss), 'title'] = 'miss'

    data.loc[data.title=='Master', 'title'] = 'master'

    data.loc[~data.title.isin(['mr', 'mrs', 'miss', 'master']), 'title'] = ''



    data['cabin_floor'] = data.cabin.str.replace('[0-9]| ', '').str.get(0).str.lower()

    data.loc[data.cabin_floor.isnull(), 'cabin_floor'] = 'z'



    data['ticket_alpha'] = data.ticket.str.extract('([A-Za-z\.\/]+)').str.replace('\.', '').str.lower()

    data['ticket_num'] = data.ticket.str.extract('([0-9\.\/]+)').str.replace('\.', '')



    data['fam_size'] = data.sibsp + data.parch

    data['is_alone'] = ((data.sibsp + data.parch) == 0) * 1



    data.loc[data.embarked.isnull(), 'embarked'] = 'S'

    data['embarked'] = data.embarked.str.lower()



    for col in ['pclass', 'sex', 'sibsp', 'parch', 'embarked', 'title', 'cabin_floor', 'fam_size']:

        encode = pd.get_dummies(data[col], prefix = col)

        data = pd.concat([data, encode], axis = 1)



    data['fare_gen'] = data.fare

    data.loc[(data.fare == 0) | (data.fare.isnull()), 'fare_gen'] = model_fare(data[(data.fare == 0) | (data.fare.isnull())])



    data['age_gen'] = data.age

    data.loc[data.age_gen.isnull(), 'age_gen'] = model_age_correct(data[data.age_gen.isnull()])



    bins = [0, 14, 32, 99]



    data['age_bin'] = pd.cut(data['age_gen'], bins, labels = ['age_0_14', 'age_14_32', 'age_32_99'])

    enc_age = pd.get_dummies(data.age_bin)

    data = pd.concat([data, enc_age], axis = 1)



    std_scale = StandardScaler().fit(data[['age_gen', 'fare_gen', 'name_length']])

    data['age_scaled'] = 0

    data['fare_scaled'] = 0

    data['name_scaled'] = 0

    data[['age_scaled', 'fare_scaled', 'name_scaled']] = std_scale.transform(data[['age_gen', 'fare_gen', 'name_length']])



    return data
x_train_tidy = tidy_data(x_train)

x_test_tidy = tidy_data(x_test)

x_train_tidy.columns
COLUMNS_TO_MODEL = [#'passengerid', 'survived', 

    #'pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked',

    #'name_length',

    #'title',

    #'cabin_floor',

    #'ticket_alpha',

    #'ticket_num',

    #'fam_size',

    #'fam_size_large',

    #'is_alone',

    'pclass_1', 'pclass_2', 'pclass_3',

    'sex_female',

    'sex_male',

    #'sibsp_0', 'sibsp_1', 'sibsp_2', 'sibsp_3', 'sibsp_4', 'sibsp_5', 'sibsp_8',

    #'parch_0', 'parch_1', 'parch_2', 'parch_3', 'parch_4', 'parch_5', 'parch_6',

    'embarked_c', #'embarked_q', 

    'embarked_s',

    #'title_',

    'title_master', 'title_miss', 'title_mr', 'title_mrs',

    #'cabin_floor_a',

    'cabin_floor_b',

    'cabin_floor_c',

    'cabin_floor_d',

    'cabin_floor_e',

    #'cabin_floor_f',

    #'cabin_floor_g',

    'cabin_floor_z',

    'fam_size_0', 'fam_size_1', 'fam_size_2', 'fam_size_3', #'fam_size_4', 'fam_size_5', 'fam_size_6', 'fam_size_7', 'fam_size_10', 

    #'fare_gen',

    #'age_gen',

    #'age_bin',

    'age_0_14', 'age_14_32', #'age_32_99',

    #'age_scaled',

    'fare_scaled',

    'name_scaled']
def get_feature_set(name):

    features = pd.read_csv('../input/titanic-features/features.csv')

    

    return features.loc[features[name]==1, 'feature'].values



get_feature_set('baseline')
corr = x_train_tidy.corr().survived

pd.DataFrame(corr[abs(corr) > .1])
def add_missing_columns(df, features):

    missing = [col for col in features if col not in df.columns]

    

    for x in missing:

        df[x] = 0

    

    return
def get_thin_data(df, features):



    add_missing_columns(df, features)



    df_thin = df[features]



    df_thin = df_thin[df_thin.columns[~(df_thin.dtypes == 'object')]]

    

    return df_thin
get_thin_data(x_train_tidy, get_feature_set('all')).columns
model_results = []



def run_model(mod, features, feature_set = '', back_elim = 'N', cross_val = 'N'):

    

    x_train_thin = get_thin_data(x_train_tidy, features)

    x_test_thin = get_thin_data(x_test_tidy, features)

        

    mod.fit(x_train_thin, y_train)



    y_train_pred = mod.predict(x_train_thin)

    y_test_pred = mod.predict(x_test_thin)

    

    output = (mod.__class__.__name__, 

              accuracy_score(y_train, y_train_pred), 

              accuracy_score(y_test, y_test_pred),

              feature_set,

              back_elim,

              cross_val)

    

    model_results.append(output)

    

    return
def show_model_results():

    df = pd.DataFrame(model_results).reset_index(drop = True)

    df.columns = ['model', 'training', 'test', 'feature_set', 'back_elim', 'cross_val']

    #df = df.sort_values(['test'], ascending = False)

    

    fig, axes = plt.subplots(1, 2, sharey = True, figsize = (15, 5))



    for i in range(0, 2):

        sns.barplot(y = df.feature_set + '|' + df.back_elim + '|' + df.cross_val + ' | '+ df.model,

                    x = df[['training', 'test'][i]],

                    ax = axes[i], 

                    palette = sns.color_palette("deep", 20))



        axes[i].set_ylabel('')

        axes[i].set_xlabel('Accuracy')

        axes[i].set_title(['training', 'test'][i].title() + ' Set Results')

        

    axes[0].set_xlim((.70, 1))

    axes[1].set_xlim((.70, .85))

    

    return df
lr = LogisticRegression(random_state = 1)

run_model(lr, get_feature_set('all'), 'all')



rf = RandomForestClassifier(random_state = 1)

run_model(rf, get_feature_set('all'), 'all')
show_model_results()
def backward_feature_elimination(model, x, y):

    

    bwd_results = []

    

    features = x.columns



    while len(features) > 4:

        model.fit(x[features], y)



        score = accuracy_score(y, model.predict(x[features]))

        

        #   Retrieve feature importance and store values

        values = dict(zip(features, model.feature_importances_))

        values['score'] = score

        bwd_results.append(values)

        

        #   Eliminate feature

        low_import = min(values.values())

        

        features = [k for k, v in values.items() if (v > low_import) & (k != 'score')]



    bwd_features = pd.DataFrame(bwd_results)

    

    # Identify the row with the highest accuracy but reverse array order

    # so the smaller feature set is returned in the case of a tie

    best = bwd_features[bwd_features.index == np.argmax(bwd_features.score[::-1])]

    

    best_result = best.T.reset_index()

    best_result.columns = ['feature', 'importance']

    best_result = best_result[(best_result.importance.notnull()) & (best_result.feature != 'score')].sort_values(['importance'], ascending = False)



    return best.columns[(best.notnull().values.ravel()) & (best.columns != 'score')].values, best_result, bwd_features
def run_backward_elimination(model, feature_set):

    x_train_thin = get_thin_data(x_train_tidy, get_feature_set(feature_set))

    x_test_thin = get_thin_data(x_test_tidy, get_feature_set(feature_set))



    feat, res, results = backward_feature_elimination(model, x_train_thin, y_train)

    

    fig, axes = plt.subplots(1, 1, figsize = (12, math.ceil(len(res) / 3.5)))



    eat = sns.barplot(y = res.feature,

                      x = res.importance,

                      color = 'blue',

                      ax = axes)

    

    run_model(model, feat, feature_set, back_elim = 'Y')

    

    return feat
features_all_back_rf = run_backward_elimination(rf, 'all')
show_model_results()
run_model(lr, get_feature_set('baseline'), 'baseline')



run_model(rf, get_feature_set('baseline'), 'baseline')

features_base_back_rf = run_backward_elimination(rf, 'baseline')
show_model_results()
gbm = GradientBoostingClassifier(random_state = 1)



run_model(gbm, get_feature_set('all'), 'all')

run_model(gbm, get_feature_set('baseline'), 'baseline')



features_all_back_gbm = run_backward_elimination(gbm, 'all')

features_base_back_gbm = run_backward_elimination(gbm, 'baseline')
show_model_results()
def cross_validate(model, x, y, param, n_range, metric = 'accuracy'):

    

    all_scores = []

    

    for val in n_range:

        

        setattr(model, param, val)

        scores = cross_val_score(model, x, y, cv = 5, scoring = metric)

        all_scores.append((param, val, scores.mean(), scores.std(), scores.min(), scores.max()))

    

    return pd.DataFrame(all_scores, columns = ['param', 'value', 'mean', 'std', 'min', 'max'])



def cross_all(model, x, y, params, metric = 'accuracy'):

    

    while len(params) > 0:

        

        cv_best = pd.DataFrame()

        cv_results = {}

    

        for k in params:

            results = cross_validate(model, x, y, k, params[k])

            

            cv_results[k] = results

            

            cv_best = cv_best.append(results[results.index==results['mean'].idxmax()])

        

        cv_best.reset_index(inplace = True)

        

        #   Select best param and set in model. Remove from params

        param, value = cv_best.loc[cv_best.index == cv_best['mean'].idxmax(), ['param', 'value']].values.ravel()

        

        if param in ('max_features', 'n_estimators'):

            value = int(value)

            

        print(cv_best)

        print(param)

        print(value)

        

        setattr(model, param, value)

        

        del params[param]

        

    return model
mod_gbm = GradientBoostingClassifier()



cross_thin = get_thin_data(x_train_tidy, features_base_back_gbm)



params = {'n_estimators': range(20, 150, 10),

          'learning_rate': np.arange(.02, .1, .01),

          'subsample': np.arange(7, 10.1, .5) / 10,

          'max_features': range(4, len(cross_thin.columns)),

          'max_depth': range(2, 10)}



mod_gbm = cross_all(mod_gbm, cross_thin, y_train, params)



run_model(mod_gbm, features_base_back_gbm, 'baseline', back_elim = 'Y', cross_val = 'Y')
mod_gbm = GradientBoostingClassifier()



cross_thin = get_thin_data(x_train_tidy, get_feature_set('baseline'))



params = {'n_estimators': range(20, 150, 10),

          'learning_rate': np.arange(.02, .1, .01),

          'subsample': np.arange(7, 10.1, .5) / 10,

          'max_features': range(4, len(cross_thin.columns)),

          'max_depth': range(2, 10)}



mod_gbm = cross_all(mod_gbm, cross_thin, y_train, params)



run_model(mod_gbm, get_feature_set('baseline'), 'baseline', back_elim = 'N', cross_val = 'Y')
show_model_results()
mod_rf = RandomForestClassifier()



cross_thin = get_thin_data(x_train_tidy, features_base_back_rf)



params = {'n_estimators': range(20, 150, 10),

          'max_features': range(4, len(cross_thin.columns))}



mod_rf = cross_all(mod_rf, cross_thin, y_train, params)



run_model(mod_rf, features_base_back_rf, 'baseline', back_elim = 'Y', cross_val = 'Y')
mod_rf = RandomForestClassifier()



cross_thin = get_thin_data(x_train_tidy, get_feature_set('baseline'))



params = {'n_estimators': range(20, 150, 10),

          'max_features': range(4, len(cross_thin.columns))}



mod_rf = cross_all(mod_rf, cross_thin, y_train, params)



run_model(mod_rf, get_feature_set('baseline'), 'baseline', back_elim = 'N', cross_val = 'Y')
show_model_results()