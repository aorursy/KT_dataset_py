# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy import array

from numpy import hstack



import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

submission_file = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
train_sales.head(3)
sell_prices.head(3)
calendar.head(10)
submission_file.head(3)
train_sales['total_sales'] = train_sales.sum(axis=1)

sns.catplot(x="cat_id", y="total_sales",

                hue="state_id",

                data=train_sales, kind="bar",

                height=8, aspect=1);
sns.catplot(x="store_id", y="total_sales",

                hue="cat_id",

                data=train_sales, kind="bar",

                height=8, aspect=1);
hobbies_state = train_sales.loc[(train_sales['cat_id'] == 'HOBBIES')].groupby(['state_id']).mean().T

hobbies_state = hobbies_state.rename({'CA': 'HOBBIES_CA', 'TX': 'HOBBIES_TX', 'WI': 'HOBBIES_WI'}, axis=1)

household_state = train_sales.loc[(train_sales['cat_id'] == 'HOUSEHOLD')].groupby(['state_id']).mean().T

household_state = household_state.rename({'CA': 'HOUSEHOLD_CA', 'TX': 'HOUSEHOLD_TX', 'WI': 'HOUSEHOLD_WI'}, axis=1)

foods_state = train_sales.loc[(train_sales['cat_id'] == 'FOODS')].groupby(['state_id']).mean().T

foods_state = foods_state.rename({'CA': 'FOODS_CA', 'TX': 'FOODS_TX', 'WI': 'FOODS_WI'}, axis=1)

nine_example = pd.concat([hobbies_state, household_state, foods_state], axis=1)

nine_example = nine_example.drop('total_sales')
from itertools import cycle

color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])



fig, axs = plt.subplots(3,3, figsize=(10,10))

axs = axs.flatten()

ax_idx = 0

for item in nine_example.columns:

    nine_example[item].plot(title=item, color=next(color_cycle), ax=axs[ax_idx])

    ax_idx += 1

plt.tight_layout()

plt.show()
nine_example.loc[nine_example['HOBBIES_CA'] == 0]
calendar.loc[calendar['d'].isin(['d_331', 'd_697', 'd_1062', 'd_1427', 'd_1792'])]
event_date = calendar.loc[calendar['event_name_1'].isin(calendar.event_name_1.unique()[1:])].d

HOBBIES_event = train_sales.loc[(train_sales['cat_id'] == 'HOBBIES')].groupby(['state_id']).mean().T.reset_index()

HOBBIES_event = HOBBIES_event.loc[HOBBIES_event['index'].isin(event_date)]

plt.figure(figsize=(15, 10))

plt.subplot(3,1,1)

nine_example['HOBBIES_CA'].plot(title='HOBBIES_CA', color=next(color_cycle))

plt.scatter(HOBBIES_event.reset_index().level_0, HOBBIES_event['CA'],color=next(color_cycle), zorder=10)

plt.subplot(3,1,2)

nine_example['HOBBIES_TX'].plot(title='HOBBIES_TX', color=next(color_cycle))

plt.scatter(HOBBIES_event.reset_index().level_0, HOBBIES_event['TX'],color=next(color_cycle), zorder=10)

plt.subplot(3,1,3)

nine_example['HOBBIES_WI'].plot(title='HOBBIES_WI', color=next(color_cycle))

plt.scatter(HOBBIES_event.reset_index().level_0, HOBBIES_event['WI'],color=next(color_cycle), zorder=10)

plt.tight_layout()

plt.show()
cal = calendar[['d', 'wday', 'month', 'year']]

cal = cal.rename(columns={'d': 'index'})

hobbies_state = train_sales.loc[(train_sales['cat_id'] == 'HOBBIES')].groupby(['state_id']).sum().T

hobbies_state = hobbies_state.reset_index()

hobbies_state = pd.merge(hobbies_state,cal, on='index')

household_state = train_sales.loc[(train_sales['cat_id'] == 'HOUSEHOLD')].groupby(['state_id']).sum().T

household_state = household_state.reset_index()

household_state = pd.merge(household_state,cal, on='index')

foods_state = train_sales.loc[(train_sales['cat_id'] == 'FOODS')].groupby(['state_id']).sum().T

foods_state = foods_state.reset_index()

foods_state = pd.merge(foods_state,cal, on='index')
plt.figure(figsize=(18, 18))

plt.subplot(3,3,1)

plt.title('hobbies')

plt.plot(range(1, 7 + 1 ,1), hobbies_state.groupby(['wday']).mean().CA, label='CA')

plt.plot(range(1, 7 + 1 ,1), hobbies_state.groupby(['wday']).mean().TX, label='TX')

plt.plot(range(1, 7 + 1 ,1), hobbies_state.groupby(['wday']).mean().WI, label='WI')

plt.legend(loc='upper right')

plt.subplot(3,3,2)

plt.title('household')

plt.plot(range(1, 7 + 1 ,1), household_state.groupby(['wday']).mean().CA, label='CA')

plt.plot(range(1, 7 + 1 ,1), household_state.groupby(['wday']).mean().TX, label='TX')

plt.plot(range(1, 7 + 1 ,1), household_state.groupby(['wday']).mean().WI, label='WI')

plt.legend(loc='upper right')

plt.subplot(3,3,3)

plt.title('foods')

plt.plot(range(1, 7 + 1 ,1), foods_state.groupby(['wday']).mean().CA, label='CA')

plt.plot(range(1, 7 + 1 ,1), foods_state.groupby(['wday']).mean().TX, label='TX')

plt.plot(range(1, 7 + 1 ,1), foods_state.groupby(['wday']).mean().WI, label='WI')

plt.legend(loc='upper right')

plt.subplot(3,3,4)

plt.title('hobbies')

plt.plot(range(1, 12 + 1 ,1), hobbies_state.groupby(['month']).mean().CA, label='CA')

plt.plot(range(1, 12 + 1 ,1), hobbies_state.groupby(['month']).mean().TX, label='TX')

plt.plot(range(1, 12 + 1 ,1), hobbies_state.groupby(['month']).mean().WI, label='WI')

plt.legend(loc='upper right')

plt.subplot(3,3,5)

plt.title('household')

plt.plot(range(1, 12 + 1 ,1), household_state.groupby(['month']).mean().CA, label='CA')

plt.plot(range(1, 12 + 1 ,1), household_state.groupby(['month']).mean().TX, label='TX')

plt.plot(range(1, 12 + 1 ,1), household_state.groupby(['month']).mean().WI, label='WI')

plt.legend(loc='upper right')

plt.subplot(3,3,6)

plt.title('foods')

plt.plot(range(1, 12 + 1 ,1), foods_state.groupby(['month']).mean().CA, label='CA')

plt.plot(range(1, 12 + 1 ,1), foods_state.groupby(['month']).mean().TX, label='TX')

plt.plot(range(1, 12 + 1 ,1), foods_state.groupby(['month']).mean().WI, label='WI')

plt.legend(loc='upper right')

plt.subplot(3,3,7)

plt.title('hobbies')

plt.plot(range(2011, 2016 + 1 ,1), hobbies_state.groupby(['year']).mean().CA, label='CA')

plt.plot(range(2011, 2016 + 1 ,1), hobbies_state.groupby(['year']).mean().TX, label='TX')

plt.plot(range(2011, 2016 + 1 ,1), hobbies_state.groupby(['year']).mean().WI, label='WI')

plt.legend(loc='upper right')

plt.subplot(3,3,8)

plt.title('household')

plt.plot(range(2011, 2016 + 1 ,1), household_state.groupby(['year']).mean().CA, label='CA')

plt.plot(range(2011, 2016 + 1 ,1), household_state.groupby(['year']).mean().TX, label='TX')

plt.plot(range(2011, 2016 + 1 ,1), household_state.groupby(['year']).mean().WI, label='WI')

plt.legend(loc='upper right')

plt.subplot(3,3,9)

plt.title('foods')

plt.plot(range(2011, 2016 + 1 ,1), foods_state.groupby(['year']).mean().CA, label='CA')

plt.plot(range(2011, 2016 + 1 ,1), foods_state.groupby(['year']).mean().TX, label='TX')

plt.plot(range(2011, 2016 + 1 ,1), foods_state.groupby(['year']).mean().WI, label='WI')

plt.legend(loc='upper right')



plt.show()
plt.figure(figsize=(12, 12))

plt.subplot(2,1,1)

hobbies_1_prices = sell_prices.loc[sell_prices['item_id'].str.contains('HOBBIES_1')]

hobbies_1_prices_CA = hobbies_1_prices.loc[hobbies_1_prices['store_id'].str.contains('CA')]

hobbies_1_prices_TX = hobbies_1_prices.loc[hobbies_1_prices['store_id'].str.contains('TX')]

hobbies_1_prices_WI = hobbies_1_prices.loc[hobbies_1_prices['store_id'].str.contains('WI')]

grouped_CA = hobbies_1_prices_CA.groupby(['wm_yr_wk'])['sell_price'].mean()

plt.plot(grouped_CA.index, grouped_CA.values, label="CA")

grouped_TX = hobbies_1_prices_TX.groupby(['wm_yr_wk'])['sell_price'].mean()

plt.plot(grouped_TX.index, grouped_TX.values, label="TX")

grouped_WI = hobbies_1_prices_WI.groupby(['wm_yr_wk'])['sell_price'].mean()

plt.plot(grouped_WI.index, grouped_WI.values, label="WI")

plt.legend(loc=(1.0, 0.5))

plt.title('HOBBIES_1 mean sell prices by state');

plt.subplot(2,1,2)

cal = calendar[['wm_yr_wk', 'd']]

cal = cal.rename(columns={"d": "index"})

hobbies_1 = train_sales.loc[train_sales['item_id'].str.contains('HOBBIES_1')]

hobbies_1_CA = hobbies_1.loc[hobbies_1['store_id'].str.contains('CA')].drop(columns = ['id','item_id','dept_id','cat_id','store_id','state_id']).sum().reset_index().drop(1913)

hobbies_1_TX = hobbies_1.loc[hobbies_1['store_id'].str.contains('TX')].drop(columns = ['id','item_id','dept_id','cat_id','store_id','state_id']).sum().reset_index().drop(1913)

hobbies_1_WI = hobbies_1.loc[hobbies_1['store_id'].str.contains('WI')].drop(columns = ['id','item_id','dept_id','cat_id','store_id','state_id']).sum().reset_index().drop(1913)

hobbies_1_CA = pd.merge(hobbies_1_CA, cal, on='index')

hobbies_1_TX = pd.merge(hobbies_1_TX, cal, on='index')

hobbies_1_WI = pd.merge(hobbies_1_WI, cal, on='index')

grouped_CA = hobbies_1_CA.drop(columns = "index").groupby(['wm_yr_wk']).sum()

plt.plot(grouped_CA.index, grouped_CA.values, label="CA")

grouped_TX = hobbies_1_TX.drop(columns = "index").groupby(['wm_yr_wk']).sum()

plt.plot(grouped_TX.index, grouped_TX.values, label="TX")

grouped_WI = hobbies_1_WI.drop(columns = "index").groupby(['wm_yr_wk']).sum()

plt.plot(grouped_WI.index, grouped_WI.values, label="WI")

plt.legend(loc=(1.0, 0.5))

plt.title('HOBBIES_1 sum sales by state');

def SMA(days, n):

    total = 0

    for i in range(n):

        total = total + days[i]

    return total/n



def count_SMA(orig, n):

    ret = np.zeros(len(orig) - n)

    for i in range(len(ret)):

        ret[i] = SMA(np.array(orig[i:i+n]), n)

    return ret



def WMA(days, n):

    total = 0

    dev = 0

    for i in range(n):

        total = total + (n-i)*days[i]

        dev = dev + (n-i)

    return total/dev



def count_WMA(orig, n):

    ret = np.zeros(len(orig) - n)

    for i in range(len(ret)):

        ret[i] = WMA(np.array(orig[i:i+n]), n)

    return ret



def EMA(days, n):

    total = 0

    a = 2/(n+1)

    for i in range(n):

        total = total + a*(days[i] - total)

    return total



def count_EMA(orig, n):

    ret = np.zeros(len(orig) - n)

    for i in range(len(ret)):

        ret[i] = EMA(np.array(orig[i:i+n]), n)

    return ret
hobbies_1_CA = hobbies_1_CA.rename(columns={0: "sales"})
CA_SMA_28 = count_SMA(hobbies_1_CA['sales'], 28)

CA_WMA_28 = count_WMA(hobbies_1_CA['sales'], 28)

CA_EMA_28 = count_EMA(hobbies_1_CA['sales'], 28)
plt.figure(figsize=(12, 6))

plt.subplot(3,1,1)

plt.plot(range(len(hobbies_1_CA['sales'])), hobbies_1_CA['sales'], label="original")

plt.plot(range(len(CA_SMA_28)), CA_SMA_28, label="SMA")

plt.legend(loc=(1.0, 0.5))

plt.subplot(3,1,2)

plt.plot(range(len(hobbies_1_CA['sales'])), hobbies_1_CA['sales'], label="original")

plt.plot(range(len(CA_WMA_28)), CA_WMA_28, label="WMA")

plt.legend(loc=(1.0, 0.5))

plt.subplot(3,1,3)

plt.plot(range(len(hobbies_1_CA['sales'])), hobbies_1_CA['sales'], label="original")

plt.plot(range(len(CA_EMA_28)), CA_EMA_28, label="EMA")

plt.legend(loc=(1.0, 0.5))

plt.show()
def melt_sales(df):

    df = df.drop(["item_id", "dept_id", "cat_id", "store_id", "state_id", "total_sales"], axis=1).melt(

        id_vars=['id'], var_name='d', value_name='demand')

    return df



sales = melt_sales(train_sales)
def map_f2d(d_col, id_col):

    eval_flag = id_col.str.endswith("evaluation")

    return "d_" + (d_col.str[1:].astype("int") + 1913 + 28 * eval_flag).astype("str")
submission = submission_file.melt(id_vars="id", var_name="d", value_name="demand").assign( demand=np.nan, d = lambda df: map_f2d(df.d, df.id))

submission.head()
sales_trend = train_sales.drop(columns = ['id','item_id','dept_id','cat_id','store_id','state_id', 'total_sales']).mean().reset_index()

sales_trend.plot()
sales_trend.rename(columns={'index':'d', 0: 'sales'}, inplace=True)

sales_trend = sales_trend.merge(calendar[["wday","month","year","d"]], on="d",how='left')

sales_trend = sales_trend.drop(columns = ["d"])
def split_sequences(sequences, n_steps):

    X, y = list(), list()

    for i in range(len(sequences)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the dataset

        if end_ix > len(sequences):

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)
in_seq1 = array(sales_trend['wday'])

in_seq2 = array(sales_trend['month'])

in_seq3 = array(sales_trend['year'])

out_seq = array(sales_trend['sales'])

in_seq1 = in_seq1.reshape((len(in_seq1), 1))

in_seq2 = in_seq2.reshape((len(in_seq2), 1))

in_seq3 = in_seq3.reshape((len(in_seq3), 1))

out_seq = out_seq.reshape((len(out_seq), 1))

dataset = hstack((in_seq1, in_seq2, in_seq3, out_seq))

n_steps = 7

X, y = split_sequences(dataset, n_steps)
train_x = X[:-30]

train_y = y[:-30]

test_x = X[-30:]

test_y = y[-30:]
n_features = train_x.shape[2]
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Flatten

from keras.layers import TimeDistributed

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D
model = Sequential()

model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(50, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(train_x, train_y, epochs=400, verbose=0)
last_30_400 = np.zeros(30)

i = 0

for test in test_x:

    test = test.reshape((1, n_steps, n_features))

    last_30_400[i] = model.predict(test, verbose=0)

    i = i + 1
subs = submission.groupby(['d']).mean().reset_index()

result = subs 
subs = subs.merge(calendar[["wday","month","year","d"]], on="d",how='left')

subs = subs.drop(columns = ["d", "demand"])

subs = pd.concat([sales_trend, subs], ignore_index=True, sort=False)
in_seq1 = array(subs['wday'])

in_seq2 = array(subs['month'])

in_seq3 = array(subs['year'])

out_seq = array(np.zeros(1969))

in_seq1 = in_seq1.reshape((len(in_seq1), 1))

in_seq2 = in_seq2.reshape((len(in_seq2), 1))

in_seq3 = in_seq3.reshape((len(in_seq3), 1))

out_seq = out_seq.reshape((len(out_seq), 1))

dataset = hstack((in_seq1, in_seq2, in_seq3, out_seq))

n_steps = 7

X, y = split_sequences(dataset, n_steps)
subs = X[-56:]
i = 0

for sub in subs:

    sub = sub.reshape((1, n_steps, n_features))

    result['demand'][i] = model.predict(sub, verbose=0)

    i = i + 1
for i in range(1,29):

    submission_file.loc[submission_file.id.str.contains("validation"), "F" + str(i)] = result["demand"][i-1]

    submission_file.loc[submission_file.id.str.contains("evaluation"), "F" + str(i)] = result["demand"][i + 28-1]
submission_file.to_csv('submission.csv', index=False)