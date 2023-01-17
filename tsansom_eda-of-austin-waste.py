import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
data = pd.read_csv('../input/austin_waste_and_diversion.csv', parse_dates=[2, 5])

data.head()
data.drop('report_date', axis=1, inplace=True)
plt.figure(figsize=(12,6))

load_type_counts = data['load_type'].value_counts()

ax1 = load_type_counts.plot(kind='bar');#, logy=True);

labels = []

for i, label in enumerate(load_type_counts.index):

    labels.append('{} - ({})'.format(label, load_type_counts[label]))

ax1.set_xticklabels(labels);
plt.figure(figsize=(12, 6))

dropoff_site_counts = data['dropoff_site'].value_counts()

ax2 = dropoff_site_counts.plot(kind='bar');

labels = []

for i, label in enumerate(dropoff_site_counts.index):

    labels.append('{} - ({})'.format(label, dropoff_site_counts[label]))

ax2.set_xticklabels(labels);
missing = data[data['load_weight'].isnull()]

missing_perc = len(missing) / len(data) * 100

print('Missing Total: {}\nMissing Percentage: {:.2f}%'.format(len(missing), missing_perc))
missing['load_type'].value_counts()
ax3 = data.boxplot(by='load_type', column='load_weight', figsize=(12, 6))

ax3.set_ylim(-1000,40000);

ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90);

data.groupby('load_type')['load_weight'].describe()
routes = data.groupby('route_number')

routes_by_route_type = routes['route_type'].nunique()

routes_by_load_type = routes['load_type'].nunique()

print('Redundant Route Types: {}'.format((routes_by_route_type > 1).sum()))

print('Redundant Load Types: {}'.format((routes_by_load_type > 1).sum()))
print('BULK: \n{}\n'.format(data[data['route_type'] == 'BULK']['route_number'].unique()))

print('DEAD ANIMAL: \n{}\n'.format(data[data['route_type'] == 'DEAD ANIMAL']['route_number'].unique()))
data_ts = data.sort_values('load_time')

data_ts.index = data_ts['load_time']

data_ts.drop('load_time', axis=1, inplace=True)

data_ts.head()
load_types = data_ts['load_type'].unique()

skip_plots = []

fig = plt.figure(figsize=(12,12))

for i, lt in enumerate(load_types):

    #resample the data to get monthly totals

    tmp = data_ts[data_ts['load_type'] == lt]['load_weight'].resample('M').sum()

    ax = fig.add_subplot(6, 3, i+1)

    plt.plot(tmp.index, tmp.values)

    ax.set_title(lt)

    ax.set_xlim(data_ts.index.min(), data_ts.index.max())

fig.tight_layout()
yt = data_ts[data_ts['load_type'] == 'YARD TRIMMING']

yt = yt.resample('M').sum()

yt['load_weight'].plot()
plt.plot(yt.groupby(yt.index.month).mean()['load_weight'])
da = data_ts[data_ts['load_type'] == 'DEAD ANIMAL']

da.resample('M').sum()['load_weight'].plot()