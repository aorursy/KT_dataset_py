import pandas as pd

from datetime import datetime
df = pd.read_csv('/kaggle/input/groceries-dataset/Groceries_dataset.csv')
df.head()
df['Date'] = df['Date'].apply(lambda x : datetime.strptime(x, "%d-%m-%Y"))
first_date = df['Date'].describe()['first']
df['timedelta'] = df['Date'].apply(lambda x : (x - first_date).days)
def id_maker(x):

    return str(x[0]) + '-' + str(x[1])
df['id'] = df[['Member_number', 'timedelta']].apply(id_maker, axis=1)
item_freq = df.itemDescription.value_counts()

item_freq = item_freq.reset_index()

item_freq = item_freq.rename(columns={'index' : 'item', 'itemDescription' : 'frequency'})

item_freq
df.drop(['Member_number', 'Date', 'timedelta'], axis=1, inplace=True)
by_purchase = df.groupby('id')
df.head()
gb = df.groupby('id')

transactions = gb.agg({'itemDescription' : lambda x : x.ravel().tolist()})

transactions
item_freq
freq_dict = {}

def freq_counter(transaction):

    for i in range(0, len(transaction)):

        x = transaction[i]

        for j in range(len(transaction) - 1, i, -1):

            y = transaction[j]

            x_id = item_freq[item_freq['item'] == x].index[0]

            y_id = item_freq[item_freq['item'] == y].index[0]

            sorted_ids = sorted([x_id, y_id])

            pair_id = str(sorted_ids[0]) + '-' + str(sorted_ids[1])

            dict_val = freq_dict.get(pair_id)

            freq_dict.update({pair_id : 1 if dict_val is None else dict_val + 1})
transactions['itemDescription'].apply(freq_counter)
freq_dict
n = 14963 #number of transactions

support = {}

for key, value in freq_dict.items():

    support.update({key : value/n})
support
d = {}

count = 0

for k in support.keys():

    items = k.split('-')

    i1 = items[0]

    i2 = items[1]

    d_i1 = d.get(i1)

    d_i2 = d.get(i2)

    s = support.get(k)

    if d_i1 is None:

        d.update({i1 : {i2 : s}})

    else:

        d_i1.update({i2 : s})

        d.update({i1 : d_i1})

    if d_i2 is None:

        d.update({i2 : {i1 : s}})

    else:

        d_i2.update({i1 : s})

        d.update({i2: d_i2})
d
item = []

supps = []

for k, v in d.items():

    item.append(k)

    supps.append(v)
supp_df = pd.DataFrame({'item' : item, 'support_dict' : supps})
supp_df.head()
def conf_calc(x):

    conf_dict = {}

    for k, v in x.items():

        k_freq = item_freq.iloc[int(k)].frequency/14963

        conf = v/k_freq

        conf_dict.update({k : conf})

    return conf_dict
supp_df['confidence_dict'] = supp_df['support_dict'].apply(conf_calc)
supp_df.confidence_dict
supp_df
def lift_calc(row):

    lift_dict = {}

    x = row[0]

    x_freq = item_freq.iloc[int(x)].frequency/14963

    for k, v in row[1].items():

        lift_dict.update({k : v/x_freq})

    return lift_dict
supp_df['lift_dict'] = supp_df[['item', 'confidence_dict']].apply(lift_calc, axis=1)
supp_df.head()