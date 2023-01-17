import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy.stats import norm, probplot

%matplotlib inline
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
fc = lambda x: datetime.datetime.strptime(x,"%Y-%m-%d")
data = pd.read_csv('/kaggle/input/Bitcoin_dataset_updated.csv', index_col = 'Date', parse_dates = True, date_parser = fc)
print('The size of the data set: {} rows and {} columns'.format(data.shape[0], data.shape[1]))
data.isnull().sum()
data.fillna(method = 'ffill', inplace=True)
data.isnull().sum()
data.head(10)
data.describe().round(2)
datemin = datetime.date(data.index.min().year, data.index.min().day-1, 1)
datemax = datetime.date(data.index.max().year, data.index.max().month + 2, 1)
sns.set(style="whitegrid", color_codes=True)

for i in data.columns:

    fig,ax = plt.subplots(figsize = (9, 5))      
    data[i].plot(kind = 'line', ax=ax)
    plt.gcf().autofmt_xdate()
    ax.set_xlabel('Date', fontsize = 15)
    ax.set_ylabel(str(i), fontsize = 15)
    ax.set_xlim([datemin, datemax])
    plt.show()
data2 = data.copy()
mms = MinMaxScaler()
for i in data2.columns:
    data2[i] = mms.fit_transform(data[[i]])
    
for i in data2.iloc[:,1:].columns:
    fig, ax = plt.subplots(figsize = (10, 5))
    data2[i].plot(ax = ax, label = i)
    data2.iloc[:,0].plot(ax = ax, label = 'BTC price')
    plt.gcf().autofmt_xdate()
    ax.set_xlabel('Date', fontsize = 15)
    ax.set_xlim([datemin, datemax])
    ax.legend()
    plt.show()
fig, ax = plt.subplots(figsize = (8, 8))
ax = sns.heatmap(data2.corr(), vmin = 0, vmax = 1, annot = True, cmap = 'hot')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_title('correlation matrix')
plt.show()
data.corr().round(4)
sns.set(style = 'white', font_scale = 3)
sns.pairplot(data,
             aspect = 1,
             diag_kind = 'kde', 
             height = 10, 
             diag_kws = {'color': 'red'}, 
             plot_kws = {'s': 100, 'marker': 'o'})
plt.xticks(rotation = 45)
fig.tight_layout()
plt.show()
sns.set(style = 'whitegrid', font_scale = 1)
fig, ax = plt.subplots(nrows = 2, figsize=(10,10))
sns.distplot(data['BTC price [USD]'], 
             fit = norm,  
             kde_kws = {'lw': 3}, 
             fit_kws = {'color': 'red', 'lw': 3},
             ax = ax[0]);
ax[0].legend(['normal distribution','real distribution'])
probplot(data['BTC price [USD]'], plot=ax[1])
fig.tight_layout()
def vif(dataset):
    X = data2.drop(columns = ['BTC price [USD]'])    
    X = sm.add_constant(X)
    return pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index = X.columns).iloc[1:]

ds=vif(data)
print(ds)
print("\n")
print('variables with large (>5) VIF:')
ds[ds > 5]
sns.set(style = 'whitegrid', font_scale = 1.5)
col_in_rows = 2
for i in range(1, len(data.columns), col_in_rows):
    g = sns.PairGrid(data, y_vars = 'BTC price [USD]', x_vars = data.columns[i:i+col_in_rows], height = 6)
    g.map(sns.regplot, order = 1, line_kws = {'color': 'red'})