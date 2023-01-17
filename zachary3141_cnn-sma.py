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
cats = train_sales.cat_id.unique()

stores = train_sales.store_id.unique()

mean_array = []
for cat in cats:

    for store in stores:

        mean_array.append(train_sales.loc[train_sales['store_id'] == store].groupby(['cat_id']).mean().loc[cat])

        mean_array.append(count_SMA(train_sales.loc[train_sales['store_id'] == store].groupby(['cat_id']).mean().loc[cat], 28))
from itertools import cycle

color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

plt.figure(figsize=(14, 7))

for i in range(4):

    temp = np.zeros(30)

    plt.subplot(4,1,i+1)

    plt.plot(range(len(mean_array[i*2])), mean_array[i*2], color=next(color_cycle))

    plt.plot(range(30, len(mean_array[i*2+1])+30), mean_array[i*2+1], color=next(color_cycle))

    plt.title(cats[0] + "_" + stores[i])

plt.tight_layout()

plt.show()
i = 0

for cat in cats:

    for store in stores:

        submission_file.loc[submission_file['id'].str.contains(cat) & submission_file['id'].str.contains(store),1:] = mean_array[i*2+1][-1]

        i += 1
submission_file.to_csv('submission.csv', index=False)
def melt_sales(df):

    df = df.drop(["item_id", "dept_id", "cat_id", "store_id", "state_id"], axis=1).melt(

        id_vars=['id'], var_name='d', value_name='demand')

    return df



sales = melt_sales(train_sales)



def map_f2d(d_col, id_col):

    eval_flag = id_col.str.endswith("evaluation")

    return "d_" + (d_col.str[1:].astype("int") + 1913 + 28 * eval_flag).astype("str")



submission = submission_file.melt(id_vars="id", var_name="d", value_name="demand").assign( demand=np.nan, d = lambda df: map_f2d(df.d, df.id))



sales_trend = train_sales.drop(columns = ['id','item_id','dept_id','cat_id','store_id','state_id']).mean().reset_index()



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
mean = result["demand"].mean()

result["demand"] = result["demand"] - mean



for i in range(1,29):

    submission_file.loc[submission_file.id.str.contains("validation"), "F" + str(i)] += result["demand"][i-1]

    submission_file.loc[submission_file.id.str.contains("evaluation"), "F" + str(i)] += result["demand"][i + 28-1]
submission_file.to_csv('submission.csv', index=False)
last_30_400 = np.zeros(30)

i = 0

for test in test_x:

    test = test.reshape((1, n_steps, n_features))

    last_30_400[i] = model.predict(test, verbose=0)

    i = i + 1

mean = last_30_400.mean()

last_30_400 = last_30_400 - mean

last_30_400_cb = last_30_400 + mean_array[1][-30]



plt.figure(figsize=(16, 8))

plt.plot(range(30), mean_array[0][-30:], label="original")

#plt.plot(range(30), mean_array[0][-(30+ 7*4*12):-(7*4*12)], label="last year")

plt.plot(range(30), last_30_400_cb, label="predicted + SMA")

plt.plot(range(30), last_30_400, label="predicted")

plt.legend(loc=(1.0, 0.5))