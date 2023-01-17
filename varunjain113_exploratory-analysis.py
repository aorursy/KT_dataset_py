import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt



%matplotlib inline
data = pd.read_csv("../input/ip-network-traffic-flows-labeled-with-87-apps/Dataset-Unicauca-Version2-87Atts.csv", parse_dates=True)

data.head()
data.shape
data.columns
data.info()
non_num_cols = [col for col in data.columns if data[col].dtype == 'O']

non_num_data = data[non_num_cols]

non_num_data
[(col, non_num_data[col].nunique()) for col in non_num_cols]
def summarize_cat(col_name):

    sorted_values = sorted(non_num_data[col_name].value_counts().iteritems(), key = lambda x:x[1], reverse=True)

    remaining_per = 100

    for (value, count) in sorted_values:

        per = count / len(non_num_data) * 100

        if per >= 1:

            print(f'{value} : {per:.2f}%')

        else :

            print(f'Others : {remaining_per:.2f}%')

            break

        remaining_per = remaining_per - per
for col in non_num_cols:

    print(f"Summary of {col} column : ")

    summarize_cat(col)

    print('\n')
num_cols = list(set(data.columns) - set(non_num_cols))

num_cols
data[num_cols].describe()
[col for col in num_cols if data[col].isnull().any()]
print("range and no. of unique values in numeric columns")

for col in num_cols:

    print(f'{col}\tRange : {max(data[col]) - min(data[col])}, No. of unique values : {data[col].nunique()}')
cols_for_hist = [col for col in num_cols if data[col].nunique() <= 50]

cols_for_hist, len(cols_for_hist)
cols_for_desc = [col for col in num_cols if data[col].nunique() > 50]

cols_for_desc
data[cols_for_hist].hist(layout = (7,3), figsize = (12, 20))

plt.tight_layout()
corr = data[num_cols].corr()
f = plt.figure(figsize = (25,25))

plt.matshow(corr, fignum=f.number)

plt.title('Correlation Matrix of Numeric columns in the dataset', fontsize = 20)

plt.xticks(range(len(num_cols)), num_cols, fontsize = 14, rotation = 90)

plt.yticks(range(len(num_cols)), num_cols, fontsize = 14)

plt.gca().xaxis.set_ticks_position('bottom')

cb = plt.colorbar(fraction = 0.0466, pad = 0.02)

cb.ax.tick_params(labelsize=10)

plt.show()
ipdata = data.copy()
print("No. of unique values in Timestamp column :",ipdata['Timestamp'].nunique())

print("No. of unique values in FlowID column :",ipdata['Flow.ID'].nunique())
ipdata.drop(['Timestamp', 'Flow.ID'], axis = 1, inplace = True)
single_unique_cols = [col for col in ipdata.columns if ipdata[col].nunique() == 1]

single_unique_cols
ipdata.drop(single_unique_cols, axis = 1, inplace = True)
ip_add_cols = ['Source.IP', 'Source.Port', 'Destination.IP', 'Destination.Port']

ipdata[ip_add_cols]
ipdata.drop(ip_add_cols, axis = 1, inplace = True)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder().fit(ipdata['ProtocolName'])

ipdata['ProtocolName'] = encoder.fit_transform(ipdata['ProtocolName'])

ipdata['ProtocolName']
ipdata.head(10)
ipdata.shape