import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
from plotly import tools
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import missingno as msno

from scipy.stats import gaussian_kde

plt.style.use('seaborn')
sns.set(font_scale=1.5)

# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()
import random

# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")
application_train = pd.read_csv('../input/application_train.csv')
# POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
previous_application = pd.read_csv('../input/previous_application.csv')
# installments_payments = pd.read_csv('../input/installments_payments.csv')
# credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
# bureau = pd.read_csv('../input/bureau.csv')
# application_test = pd.read_csv('../input/application_test.csv')
print('Size of application_tra data', application_train.shape)
# print('Size of POS_CASH_balance data', POS_CASH_balance.shape)
# print('Size of bureau_balance data', bureau_balance.shape)
# print('Size of previous_application data', previous_application.shape)
# print('Size of installments payments data', installments_payments.shape)
# print('Size of credit_card_balance data', credit_card_balance.shape)
# print('Size of bureau data', bureau.shape)
application_train.head()
msno.matrix(df=application_train, figsize=(10, 8), color=(0, 0.6, 1))
# checking missing data
total = application_train.isnull().sum().sort_values(ascending = False)
percent = (application_train.isnull().sum()/application_train.isnull().count()*100).sort_values(ascending = False)
missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_application_train_data.head(20)
application_train.info()
def make_meta_dataframe(df):
    data = []
    for col in df.columns:
        if col == 'TARGET':
            role = 'target'
        elif col == 'SK_ID_CURR':
            role = 'id'
        else:
            role = 'input'

        if df[col].dtype == 'float64':
            level = 'interval'
        elif df[col].dtype == 'int64':
            level = 'ordinal'
        elif df[col].dtype == 'object':
            level = 'categorical'

        col_dict = {
            'varname': col,
            'role': role,
            'level': level,
            'dtype': df[col].dtype,
            'response_rate': 100 * df[col].notnull().sum() / df.shape[0]
        }
        data.append(col_dict)

    meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'dtype', 'response_rate'])
    meta.set_index('varname', inplace=True)
    return meta
meta = make_meta_dataframe(application_train)
def random_color_generator(number_of_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color
cnt_srs = application_train['TARGET'].value_counts()
text = ['{:.2f}%'.format(100 * (value / cnt_srs.sum())) for value in cnt_srs.values]

trace = go.Bar(
    x = cnt_srs.index,
    y = (cnt_srs / cnt_srs.sum()) * 100,
    marker = dict(
        color = random_color_generator(2),
        line = dict(color='rgb(8, 48, 107)',
                   width = 1.5
                   )
    ), 
    opacity = 0.7
)

data = [trace]

layout = go.Layout(
    title = 'Target distribution(%)',
    margin = dict(
        l = 100
    ),
    xaxis = dict(
        title = 'Labels (0: repay, 1: not repay)'
    ),
    yaxis = dict(
        title = 'Account(%)'
    ),
    width=800,
    height=500
)
annotations = []
for i in range(2):
    annotations.append(dict(
        x = cnt_srs.index[i],
        y = ((cnt_srs / cnt_srs.sum()) * 100)[i],
        text = text[i],
        font = dict(
            family = 'Arial',
            size = 14,
        ),
        showarrow = True
    ))
    layout['annotations'] = annotations

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
def get_percent(df, temp_col, width=800, height=500):
    cnt_srs = df[[temp_col, 'TARGET']].groupby([temp_col], as_index=False).mean().sort_values(by=temp_col)

    trace = go.Bar(
        x = cnt_srs[temp_col].values[::-1],
        y = cnt_srs['TARGET'].values[::-1],
        text = cnt_srs.values[::-1],
        textposition = 'auto',
        textfont = dict(
            size=12,
            color='rgb(0, 0, 0)'
        ),
        orientation = 'v',
            marker = dict(
                color = random_color_generator(100),
                line=dict(color='rgb(8,48,107)',
                  width=1.5,)
            ),
            opacity = 0.7,
    )    
    return trace
#     fig = go.Figure(data=data, layout=layout)
#     py.iplot(fig)


def get_count(df, temp_col, width=800, height=500):
    cnt_srs = df[temp_col].value_counts().sort_index()

    trace = go.Bar(
        x = cnt_srs.index[::-1],
        y = cnt_srs.values[::-1],
        text = cnt_srs.values[::-1],
        textposition = 'auto',
        textfont = dict(
            size=12,
            color='rgb(0, 0, 0)'
        ),
        name = 'Percent',
        orientation = 'v',
            marker = dict(
                color = random_color_generator(100),
                line=dict(color='rgb(8,48,107)',
                  width=1.5,)
            ),
            opacity = 0.7,
    )    
    return trace
#     fig = go.Figure(data=data, layout=layout)
#     py.iplot(fig)
def plot_count_percent_for_object(df, temp_col, height=500):
    trace1 = get_count(df, temp_col)
    trace2 = get_percent(df, temp_col)

    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Count', 'Percent'), print_grid=False)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)

    fig['layout']['yaxis1'].update(title='Count')
    fig['layout']['yaxis2'].update(range=[0, 1], title='% TARGET')
    fig['layout'].update(title='{} (Response rate: {:.2f}%)'.format(temp_col, meta[(meta.index == temp_col)]['response_rate'].values[0]), margin=dict(l=100), width=800, height=height, showlegend=False)

    py.iplot(fig)
features_dtype_object = meta[meta['dtype'] == 'object'].index
features_dtype_int = meta[meta['dtype'] == 'int64'].index
features_dtype_float = meta[meta['dtype'] == 'float64'].index
application_object_na_filled = application_train[features_dtype_object].fillna('null')
application_object_na_filled['TARGET'] = application_train['TARGET']
temp_col = features_dtype_object[0]
plot_count_percent_for_object(application_train, temp_col)
temp_col = features_dtype_object[1]
plot_count_percent_for_object(application_train, temp_col)
temp_col = features_dtype_object[2]
plot_count_percent_for_object(application_train, temp_col)
temp_col = features_dtype_object[3]
plot_count_percent_for_object(application_train, temp_col)
temp_col = features_dtype_object[4]
plot_count_percent_for_object(application_train, temp_col)
plot_count_percent_for_object(application_object_na_filled, temp_col)
temp_col = features_dtype_object[5]
plot_count_percent_for_object(application_train, temp_col)
temp_col = features_dtype_object[6]
plot_count_percent_for_object(application_train, temp_col)
temp_col = features_dtype_object[7]
plot_count_percent_for_object(application_train, temp_col)
temp_col = features_dtype_object[8]
plot_count_percent_for_object(application_train, temp_col)
temp_col = features_dtype_object[9]
plot_count_percent_for_object(application_train, temp_col)
plot_count_percent_for_object(application_object_na_filled, temp_col)
temp_col = features_dtype_object[10]
plot_count_percent_for_object(application_train, temp_col)
temp_col = features_dtype_object[11]
plot_count_percent_for_object(application_train, temp_col)
temp_col = features_dtype_object[12]
plot_count_percent_for_object(application_train, temp_col)
plot_count_percent_for_object(application_object_na_filled, temp_col)
temp_col = features_dtype_object[13]
plot_count_percent_for_object(application_train, temp_col)
plot_count_percent_for_object(application_object_na_filled, temp_col)
temp_col = features_dtype_object[14]
plot_count_percent_for_object(application_train, temp_col)
plot_count_percent_for_object(application_object_na_filled, temp_col)
temp_col = features_dtype_object[15]
plot_count_percent_for_object(application_train, temp_col)
plot_count_percent_for_object(application_object_na_filled, temp_col)
def plot_count_percent_for_int(df, temp_col, height=500):
    trace1 = get_count(df, temp_col)
    trace2 = get_percent(df, temp_col)

    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Count', 'Percent'), print_grid=False)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)

    fig['layout']['xaxis1'].update(tickvals=[i for i in range(20)])
    fig['layout']['xaxis2'].update(tickvals=[i for i in range(20)])
    fig['layout']['yaxis1'].update(title='Count')
    fig['layout']['yaxis2'].update(range=[0, 1], title='% TARGET')
    fig['layout'].update(title='{} (Response rate: {:.2f}%)'.format(temp_col, meta[(meta.index == temp_col)]['response_rate'].values[0]), margin=dict(l=100), width=800, height=height, showlegend=False)
    
    py.iplot(fig)
application_train_int = application_train[meta[meta['dtype'] == 'int64'].index]
application_train_int['TARGET'] = application_train['TARGET']
features_dtype_int
temp_col = features_dtype_int[2]
plot_count_percent_for_int(application_train_int, temp_col)
temp_col = features_dtype_int[6]
plot_count_percent_for_int(application_train_int, temp_col)
temp_col = features_dtype_int[7]
plot_count_percent_for_int(application_train, temp_col)
temp_col = features_dtype_int[8]
plot_count_percent_for_int(application_train, temp_col)
temp_col = features_dtype_int[9]
plot_count_percent_for_int(application_train, temp_col)
temp_col = features_dtype_int[10]
plot_count_percent_for_int(application_train, temp_col)
temp_col = features_dtype_int[12]
plot_count_percent_for_int(application_train, temp_col)
temp_col = features_dtype_int[13]
plot_count_percent_for_int(application_train, temp_col)
temp_col = features_dtype_int[14]
plot_count_percent_for_int(application_train, temp_col)
temp_col = features_dtype_int[15]
plot_count_percent_for_int(application_train, temp_col)
temp_col = features_dtype_int[16]
plot_count_percent_for_int(application_train, temp_col)
temp_col = features_dtype_int[17]
plot_count_percent_for_int(application_train, temp_col)
temp_col = features_dtype_int[18]
plot_count_percent_for_int(application_train, temp_col)
temp_col = features_dtype_int[19]
plot_count_percent_for_int(application_train, temp_col)
temp_col = features_dtype_int[20]
plot_count_percent_for_int(application_train, temp_col)
for i in range(21, 40):
    temp_col = features_dtype_int[i]
    plot_count_percent_for_int(application_train, temp_col)
data = [
    go.Heatmap(
        z = application_train_int.corr().values,
        x = application_train_int.columns.values,
        y = application_train_int.columns.values,
        colorscale='Viridis',
        reversescale = False,
        text = True ,
    )
]
layout = go.Layout(
    title='Pearson Correlation of float-type features',
    xaxis = dict(ticks=''),
    yaxis = dict(ticks='' ),
    width = 900, height = 700,
    margin = dict(
        l = 250
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='labelled-heatmap')
correlations = application_train_int.corr()['TARGET'].sort_values()
correlations[correlations.abs() > 0.05]
temp_col = 'DAYS_BIRTH'
sns.kdeplot((application_train_int.loc[application_train_int['TARGET'] == 0, temp_col]/365).abs(), label='repay(0)', color='r')
sns.kdeplot((application_train_int.loc[application_train_int['TARGET'] == 1, temp_col]/365).abs(), label='not repay(1)', color='b')
plt.xlabel('Age(years)')
plt.title('KDE for {} splitted by target'.format(temp_col))
plt.show()
sns.lmplot(x='FLAG_DOCUMENT_6', y='FLAG_EMP_PHONE', data=application_train_int)
col1 = 'FLAG_DOCUMENT_6'
col2 = 'FLAG_EMP_PHONE'
xy = np.vstack([application_train[col1].dropna().values[:100000], application_train[col2].dropna().values[:100000]])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
im = ax.scatter(application_train[col1].dropna().values[:100000], application_train[col2].dropna().values[:100000], c=z, s=50, cmap=plt.cm.jet)
ax.set_xlabel(col1)
ax.set_ylabel(col2)
fig.colorbar(im)
sns.lmplot(x='DAYS_BIRTH', y='FLAG_EMP_PHONE', data=application_train_int)
col1 = 'DAYS_BIRTH'
col2 = 'FLAG_EMP_PHONE'
xy = np.vstack([np.abs((application_train[col1].dropna().values[:100000]/365)), application_train[col2].dropna().values[:100000]])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
im = ax.scatter(np.abs((application_train[col1].dropna().values[:100000]/365)), application_train[col2].dropna().values[:100000], c=z, s=50, cmap=plt.cm.jet)
ax.set_xlabel(col1)
ax.set_ylabel(col2)
fig.colorbar(im)
sns.lmplot(x='DAYS_EMPLOYED', y='FLAG_EMP_PHONE', data=application_train_int.dropna().loc[:100000, :])
col1 = 'DAYS_EMPLOYED'
col2 = 'FLAG_EMP_PHONE'
xy = np.vstack([np.abs((application_train[col1].dropna().values[:100000]/365)), application_train[col2].dropna().values[:100000]])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
im = ax.scatter(np.abs((application_train[col1].dropna().values[:100000]/365)), application_train[col2].dropna().values[:100000], c=z, s=50, cmap=plt.cm.jet)
ax.set_xlabel(col1)
ax.set_ylabel(col2)
fig.colorbar(im)
application_train_float = application_train[meta[meta['dtype'] == 'float64'].index]
application_train_float['TARGET'] = application_train['TARGET']
data = [
    go.Heatmap(
        z = application_train_float.corr().values,
        x = application_train_float.columns.values,
        y = application_train_float.columns.values,
        colorscale='Viridis',
        reversescale = False,
        text = True ,
    )
]
layout = go.Layout(
    title='Pearson Correlation of float-type features',
    xaxis = dict(ticks=''),
    yaxis = dict(ticks='' ),
    width = 1200, height = 1200,
    margin = dict(
        l = 250
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='labelled-heatmap')
correlations = application_train_float.corr()['TARGET'].sort_values()
correlations[correlations.abs() > 0.05]
temp_col = 'EXT_SOURCE_1'
sns.kdeplot(application_train_float.loc[application_train_float['TARGET'] == 0, temp_col], label='repay(0)', color='r')
sns.kdeplot(application_train_float.loc[application_train_float['TARGET'] == 1, temp_col], label='not repay(1)', color='b')
plt.title('KDE for {} splitted by target'.format(temp_col))
plt.show()
temp_col = 'EXT_SOURCE_2'
sns.kdeplot(application_train_float.loc[application_train_float['TARGET'] == 0, temp_col], label='repay(0)', color='r')
sns.kdeplot(application_train_float.loc[application_train_float['TARGET'] == 1, temp_col], label='not repay(1)', color='b')
plt.title('KDE for {} splitted by target'.format(temp_col))
plt.show()
temp_col = 'EXT_SOURCE_3'
sns.kdeplot(application_train_float.loc[application_train_float['TARGET'] == 0, temp_col], label='repay(0)', color='r')
sns.kdeplot(application_train_float.loc[application_train_float['TARGET'] == 1, temp_col], label='not repay(1)', color='b')
plt.title('KDE for {} splitted by target'.format(temp_col))
plt.show()
corr_matrix = application_train_float.corr().abs()
corr_matrix.head()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.head()
threshold = 0.9
count = 1
combinations = []
for name, column in upper.iteritems():
    if (column > threshold).any():
        for col, value in column[column > 0.9].iteritems():
            print(count, name, col, value)
            combinations.append((name, col, value))
            count += 1
fig, ax = plt.subplots(28, 2, figsize=(20, 400))
count = 0
for i in range(28):
    for j in range(2):
        sns.regplot(x=combinations[count][0], y=combinations[count][1], data=application_train_float[application_train_float['TARGET'] == 0], ax=ax[i][j], color='r')
        sns.regplot(x=combinations[count][0], y=combinations[count][1], data=application_train_float[application_train_float['TARGET'] == 1], ax=ax[i][j], color='b')
        ax[i][j].set_title('{} and {}, corr:{:.2f} '.format(combinations[count][0], combinations[count][1], combinations[count][2]))
        ax[i][j].legend(['repay', 'not repay'], loc=0)
        count += 1
def multi_features_kde_plot(col1, col2):
    fig, ax = plt.subplots(3, 2, figsize=(14, 20))
    g = sns.kdeplot(application_train_float.loc[application_train['TARGET'] == 0, :].dropna().loc[:50000, :][col1], application_train_float.loc[application_train['TARGET'] == 0, :].dropna().loc[:50000, :][col2], ax=ax[0][0], cmap="Reds")
    g = sns.kdeplot(application_train_float.loc[application_train['TARGET'] == 1, :].dropna().loc[:50000, :][col1], application_train_float.loc[application_train['TARGET'] == 1, :].dropna().loc[:50000, :][col2], ax=ax[0][1], cmap='Blues')
    ax[0][0].set_title('mutivariate KDE: target == repay')
    ax[0][1].set_title('mutivariate KDE: target == not repay')

    temp_col = col1
    sns.kdeplot(application_train.loc[application_train['TARGET'] == 0, temp_col].dropna(), label='repay(0)', color='r', ax=ax[1][0])
    sns.kdeplot(application_train.loc[application_train['TARGET'] == 1, temp_col].dropna(), label='not repay(1)', color='b', ax=ax[1][0])
    ax[1][0].set_title('KDE for {}'.format(temp_col))

    sns.kdeplot(np.log(application_train.loc[(application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[1][1])
    sns.kdeplot(np.log(application_train.loc[(application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[1][1])
    ax[1][1].set_title('KDE for {} with log'.format(temp_col))

    temp_col = col2
    sns.kdeplot(application_train.loc[application_train['TARGET'] == 0, temp_col].dropna(), label='repay(0)', color='r', ax=ax[2][0])
    sns.kdeplot(application_train.loc[application_train['TARGET'] == 1, temp_col].dropna(), label='not repay(1)', color='b', ax=ax[2][0])
    ax[2][0].set_title('KDE for {}'.format(temp_col))

    sns.kdeplot(np.log(application_train.loc[(application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[2][1])
    sns.kdeplot(np.log(application_train.loc[(application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[2][1])
    ax[2][1].set_title('KDE for {} with log'.format(temp_col))
col1 = 'OBS_60_CNT_SOCIAL_CIRCLE'
col2 = 'OBS_30_CNT_SOCIAL_CIRCLE'
multi_features_kde_plot(col1, col2)
col1 = 'NONLIVINGAREA_MEDI'
col2 = 'NONLIVINGAREA_MODE'
multi_features_kde_plot(col1, col2)
# Read in bureau
bureau = pd.read_csv('../input/bureau.csv')
bureau.head()
msno.matrix(df=bureau, figsize=(10, 8), color=(0, 0.6, 1))
bureau.head()
print('Applicatoin train shape before merge: ', application_train.shape)
application_train = application_train.merge(bureau.groupby('SK_ID_CURR').mean().reset_index(), 
                                            left_on='SK_ID_CURR', right_on='SK_ID_CURR', 
                                            how='left', validate='one_to_one')
print('Applicatoin train shape after merge: ', application_train.shape)
meta= make_meta_dataframe(application_train)
bureau.info()
temp_col = 'CREDIT_ACTIVE'
plot_count_percent_for_object(application_train, temp_col)
temp_col = 'CREDIT_CURRENCY'
plot_count_percent_for_object(application_train, temp_col)
temp_col = 'CREDIT_TYPE'
plot_count_percent_for_object(application_train, temp_col)
temp_col = 'DAYS_CREDIT'
plt.figure(figsize=(10, 6))
sns.distplot(application_train.loc[(application_train['TARGET'] == 0), temp_col], bins=100, label='repay(0)', color='r')
sns.distplot(application_train.loc[(application_train['TARGET'] == 1), temp_col], bins=100, label='not repay(1)', color='b')
plt.title('Distplot for {} splitted by target'.format(temp_col))
plt.legend()
plt.show()
temp_col = 'CREDIT_DAY_OVERDUE'
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.kdeplot(application_train.loc[application_train['TARGET'] == 0, temp_col].dropna(), label='repay(0)', color='r', ax=ax[0])
sns.kdeplot(application_train.loc[application_train['TARGET'] == 1, temp_col].dropna(), label='not repay(1)', color='b', ax=ax[0])
ax[0].set_title('KDE for {}'.format(temp_col))

sns.kdeplot(np.log(application_train.loc[(application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[1])
sns.kdeplot(np.log(application_train.loc[(application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[1])
ax[1].set_title('KDE for {} with log'.format(temp_col))
plt.show()
temp_col = 'CREDIT_DAY_OVERDUE'
fig, ax = plt.subplots(2, 2, figsize=(16, 16))

sns.kdeplot(application_train.loc[(application_train[temp_col] > 200) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[0][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] > 200) & (application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[0][0])
ax[0][0].set_title('KDE for {}'.format(temp_col))

application_train.loc[(application_train[temp_col] > 200) & (application_train['TARGET'] == 0), temp_col].dropna().hist(bins=100, ax=ax[0][1], normed=True, color='r', alpha=0.5)
application_train.loc[(application_train[temp_col] > 200) & (application_train['TARGET'] == 1), temp_col].dropna().hist(bins=100, ax=ax[0][1], normed=True, color='b', alpha=0.5)


sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > 200) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[1][0])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > 200) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[1][0])
ax[1][0].set_title('KDE for {} with log'.format(temp_col))


np.log(application_train.loc[(application_train[temp_col] > 200) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001).hist(bins=100, ax=ax[1][1], normed=True, color='r', alpha=0.5)
np.log(application_train.loc[(application_train[temp_col] > 200) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001).hist(bins=100, ax=ax[1][1], normed=True, color='b', alpha=0.5)
def overdue(x):
    if x < 30:
        return 'A'
    elif x < 60:
        return 'B'
    elif x < 90:
        return 'C'
    elif x < 180:
        return 'D'
    elif x < 365:
        return 'E'
    else:
        return 'F'
application_train['CREDIT_DAY_OVERDUE_cat'] = application_train['CREDIT_DAY_OVERDUE'].apply(overdue)
meta = make_meta_dataframe(application_train)
temp_col = 'CREDIT_DAY_OVERDUE_cat'
plot_count_percent_for_object(application_train, temp_col)
temp_col = 'CREDIT_DAY_OVERDUE'
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.kdeplot(application_train.loc[(application_train['TARGET'] == 0) & (application_train['CREDIT_DAY_OVERDUE'] > 30), temp_col].dropna(), label='repay(0)', color='r', ax=ax[0])
sns.kdeplot(application_train.loc[(application_train['TARGET'] == 1) & (application_train['CREDIT_DAY_OVERDUE'] > 30), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[0])
ax[0].set_title('KDE for {} (>30)'.format(temp_col))

sns.kdeplot(np.log(application_train.loc[(application_train['TARGET'] == 0) & (application_train['CREDIT_DAY_OVERDUE'] > 30), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[1])
sns.kdeplot(np.log(application_train.loc[(application_train['TARGET'] == 1) & (application_train['CREDIT_DAY_OVERDUE'] > 30), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[1])
ax[1].set_title('KDE for {} with log (>30)'.format(temp_col))
plt.show()
temp_col = 'CNT_CREDIT_PROLONG'
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
sns.kdeplot(application_train.loc[application_train['TARGET'] == 0, temp_col], label='repay(0)', color='r', ax=ax[0])
sns.kdeplot(application_train.loc[application_train['TARGET'] == 1, temp_col], label='not repay(1)', color='b', ax=ax[0])
plt.title('KDE for {} splitted by target'.format(temp_col))

sns.kdeplot(application_train.loc[(application_train['TARGET'] == 0) & (application_train[temp_col] > 3), temp_col], label='repay(0)', color='r', ax=ax[1])
sns.kdeplot(application_train.loc[(application_train['TARGET'] == 1) & (application_train[temp_col] > 3), temp_col], label='not repay(1)', color='b', ax=ax[1])
plt.title('KDE for {} splitted by target (>3)'.format(temp_col))
plt.show()
temp_col = 'AMT_CREDIT_SUM'
fig, ax = plt.subplots(2, 2, figsize=(16, 16))
threshold = 2 * 10e6
sns.kdeplot(application_train.loc[(application_train[temp_col] < threshold) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[0][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] < threshold) &(application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[0][0])
ax[0][0].set_title('KDE for {} (< {})'.format(temp_col, threshold))

sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] < threshold) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[0][1])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] < threshold) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[0][1])
ax[0][1].set_title('KDE for {} with log (< {})'.format(temp_col, threshold))

sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[1][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold) &(application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[1][0])
ax[1][0].set_title('KDE for {} (> {})'.format(temp_col, threshold))

sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[1][1])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[1][1])
ax[1][1].set_title('KDE for {} with log (> {})'.format(temp_col, threshold))
plt.show()
temp_col = 'AMT_CREDIT_SUM_DEBT'
fig, ax = plt.subplots(2, 2, figsize=(16, 16))
threshold = 2 * 10e6
sns.kdeplot(application_train.loc[(application_train[temp_col] < threshold) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[0][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] < threshold) &(application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[0][0])
ax[0][0].set_title('KDE for {} (< {})'.format(temp_col, threshold))

sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] < threshold) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[0][1])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] < threshold) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[0][1])
ax[0][1].set_title('KDE for {} with log (< {})'.format(temp_col, threshold))

sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[1][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold) &(application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[1][0])
ax[1][0].set_title('KDE for {} (> {})'.format(temp_col, threshold))

sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[1][1])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[1][1])
ax[1][1].set_title('KDE for {} with log (> {})'.format(temp_col, threshold))
plt.show()
temp_col = 'AMT_CREDIT_SUM_LIMIT'
fig, ax = plt.subplots(3, 2, figsize=(16, 24))
threshold1 = 1e4
threshold2 = 1e6

sns.kdeplot(application_train.loc[(application_train[temp_col] < threshold1) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[0][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] < threshold1) &(application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[0][0])
ax[0][0].set_title('KDE for {} (< {})'.format(temp_col, threshold1))

sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] < threshold1) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[0][1])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] < threshold1) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[0][1])
ax[0][1].set_title('KDE for {} with log (< {})'.format(temp_col, threshold1))


sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold1) & (application_train[temp_col] < threshold2) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[1][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold1) & (application_train[temp_col] < threshold2) &(application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[1][0])
ax[1][0].set_title('KDE for {} ({} < and < {})'.format(temp_col, threshold1, threshold2))

sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold1) & (application_train[temp_col] < threshold2) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[1][1])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold1) & (application_train[temp_col] < threshold2) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[1][1])
ax[1][1].set_title('KDE for {} with log ({} < and < {})'.format(temp_col, threshold1, threshold2))


sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold2) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[2][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold2) &(application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[2][0])
ax[2][0].set_title('KDE for {} (> {})'.format(temp_col, threshold2))

sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold2) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[2][1])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold2) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[2][1])
ax[2][1].set_title('KDE for {} with log (> {})'.format(temp_col, threshold2))


plt.show()
temp_col = 'AMT_CREDIT_SUM_OVERDUE'
fig, ax = plt.subplots(3, 2, figsize=(16, 24))
threshold1 = 1e3
threshold2 = 1e5

sns.kdeplot(application_train.loc[(application_train[temp_col] < threshold1) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[0][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] < threshold1) &(application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[0][0])
ax[0][0].set_title('KDE for {} (< {})'.format(temp_col, threshold1))

sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] < threshold1) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[0][1])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] < threshold1) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[0][1])
ax[0][1].set_title('KDE for {} with log (< {})'.format(temp_col, threshold1))


sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold1) & (application_train[temp_col] < threshold2) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[1][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold1) & (application_train[temp_col] < threshold2) &(application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[1][0])
ax[1][0].set_title('KDE for {} ({} < and < {})'.format(temp_col, threshold1, threshold2))

sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold1) & (application_train[temp_col] < threshold2) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[1][1])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold1) & (application_train[temp_col] < threshold2) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[1][1])
ax[1][1].set_title('KDE for {} with log ({} < and < {})'.format(temp_col, threshold1, threshold2))


sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold2) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[2][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold2) &(application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[2][0])
ax[2][0].set_title('KDE for {} (> {})'.format(temp_col, threshold2))

sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold2) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[2][1])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold2) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[2][1])
ax[2][1].set_title('KDE for {} with log (> {})'.format(temp_col, threshold2))


plt.show()












