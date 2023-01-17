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
# checking missing data
total = application_train.isnull().sum().sort_values(ascending = False)
percent = (application_train.isnull().sum()/application_train.isnull().count()*100).sort_values(ascending = False)
missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_application_train_data.head(20)
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
features_dtype_int = meta[meta['dtype'] == 'int64'].index
for i in range(0, features_dtype_int.shape[0]):
    for j in range(i, features_dtype_int.shape[0]):
        col1 = features_dtype_int[i]
        col2 = features_dtype_int[j]
        if col1 == col2 or col1 == 'TARGET' or col2 == 'TARGET':
            continue
        if application_train[col1].value_counts().shape[0] == 2 and application_train[col2].value_counts().shape[0] == 2:
            temp_col = '{}_{}'.format(col1, col2)
            application_train[temp_col] = application_train[col1] * application_train[col2]
            max_rate = application_train[[temp_col, 'TARGET']].groupby(temp_col).mean().max().values[0]
            if max_rate > 0.1 and application_train[temp_col].sum() > 5:
                print('Column 1: ', col1, '    Column 2:', col2)
                meta = make_meta_dataframe(application_train)
                plot_count_percent_for_int(application_train, temp_col)