import pandas as pd
pd.__version__
import numpy as np
mylist=list('abcdefghijklmnopqrstuvwxyz') 
myarr=np.arange(26)
mydict=dict(zip(mylist,myarr))
mydict
myser=pd.Series(mydict)
myser
df=myser.reset_index()
df
se1=pd.Series(mylist)
se2=pd.Series(myarr)
df=pd.concat([se1,se2],axis=1)
df.head(5)
se1.name='Alphabets'
se1.head(5)
ser1 = pd.Series([1, 2, 3, 4, 5])

ser2 = pd.Series([4, 5, 6, 7, 8])
ser1[~ser1.isin(ser2)]
ser_u = pd.Series(np.union1d(ser1, ser2))  # union

ser_i = pd.Series(np.intersect1d(ser1, ser2))  # intersect

ser_u[~ser_u.isin(ser_i)]
ser = pd.Series(np.random.normal(10, 5, 25))
ser.min()
np.percentile(ser,0.25)
ser.median()
np.percentile(ser,0.75)
ser.max()
np.percentile(ser,[0.25,0.50,0.75,100])
ser = pd.Series(np.take(list('abcdefgh'), np.random.randint(8, size=30)))
ser.value_counts()
np.random.RandomState(100)

ser = pd.Series(np.random.randint(1, 5, [12]))
ser
ser.value_counts()
ser.value_counts().index[:2]
ser[~ser.isin(ser.value_counts().index[:2])]='Other'
ser
ser = pd.Series(np.random.random(20))

ser.head(5)
a,b=pd.qcut(ser,q=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]).head(),pd.qcut(ser,q=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],

                                                                   labels=['1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th']).head()

print(a,'\n',b)
ser = pd.Series(np.random.randint(1, 10, 35))
ser1=pd.DataFrame(ser.values.reshape(7,5))
ser1
ser=pd.Series(np.random.randint(1,10,7))
ser
np.argwhere(ser%3==0)
ser = pd.Series(list('abcdefghijklmnopqrstuvwxyz'))
pos=[0,4,8,14,20]
ser[pos]
ser1=pd.Series(range(5))

ser2=pd.Series(list('abcde'))
serh=pd.DataFrame(np.hstack([ser1,ser2]))

serh
serv=pd.concat([ser1, ser2], axis=1)

serv
ser1 = pd.Series([10, 9, 6, 5, 3, 1, 12, 8, 13])

ser2 = pd.Series([1, 3, 10, 13])
[np.where(i==ser1)[0].tolist()[0] for i in ser2]
truth = pd.Series(range(10))

pred = pd.Series(range(10)) + np.random.random(10)
np.mean((truth-pred)**2)
ser = pd.Series(['how', 'to', 'kick', 'ass?'])
ser.map(lambda x:x.title())
ser.map(lambda x:len(x))
ser = pd.Series([1, 3, 6, 10, 15, 21, 27, 35])
print(ser.diff().tolist())

print(ser.diff().diff().tolist())
ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])
pd.to_datetime(ser)
ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])
from dateutil.parser import parse

ser_ts = ser.map(lambda x: parse(x))
print("Date:",ser_ts.dt.day.tolist())

print("Week",ser_ts.dt.week.tolist())

print("Day of Year:",ser_ts.dt.dayofyear.tolist())

print("Date:",ser_ts.dt.weekday_name.tolist())
ser = pd.Series(['Jan 2010', 'Feb 2011', 'Mar 2012'])
ser_t=ser.map(lambda x:parse(x))

format=ser_t.dt.year.astype('str')+"-"+ser_t.dt.month.astype('str')+"-"+"04"

[parse(i).strftime('%Y-%m-%d') for i in format]
ser.map(lambda x:parse('04'+x))
ser = pd.Series(['Apple', 'Orange', 'Plan', 'Python', 'Money'])
from collections import Counter

mask = ser.map(lambda x: sum([Counter(x.lower()).get(i, 0) for i in list('aeiou')]) >= 2)

ser[mask]
emails = pd.Series(['buying books at amazom.com', 'rameses@egypt.com', 'matt@t.co', 'narendra@modi.com'])

pattern ='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4}'
import re

mask = emails.map(lambda x: bool(re.match(pattern, x)))

emails[mask]
fruit=pd.Series(np.random.choice(['apple', 'banana', 'carrot'], 10))
weights = pd.Series(np.linspace(1, 10, 10))
print(weights.tolist())

print(fruit.tolist())
weights.groupby(fruit).mean()
p = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

q = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
sum((p-q)**2)**.5
ser = pd.Series([2, 10, 3, 4, 9, 10, 2, 7, 3])
dd = np.diff(np.sign(np.diff(ser)))

peak_locs = np.where(dd == -2)[0] + 1

peak_locs
my_str = 'dbc deb abed gade'
ser=pd.Series(list('dbc deb abed gade'))
freq=ser.value_counts()
least_freq = freq.dropna().index[-1]
"".join(ser.replace(' ', least_freq))
ser=pd.Series(np.random.randint(1,10,10),pd.date_range('2000-01-01',periods=10,freq='W-SAT'))

ser
ser = pd.Series([1,10,3, np.nan], index=pd.to_datetime(['2000-01-01', '2000-01-03', '2000-01-06', '2000-01-08']))

ser.resample('D').bfill().ffill()
ser = pd.Series(np.arange(20) + np.random.normal(1, 10, 20))
autocorrelations = [ser.autocorr(i).round(2) for i in range(11)]

print(autocorrelations[1:])

print('Lag having highest correlation: ', np.argmax(np.abs(autocorrelations[1:]))+1)
df=pd.read_csv('../input/house-prices-advanced-regression-techniques/housetrain.csv',chunksize=50)
df2 = pd.DataFrame()

for chunk in df:

    df2 = df2.append(chunk.iloc[0,:])
df2.head()
L = pd.Series(range(15))
def gen_strides(a, stride_len=5, window_len=5):

    n_strides = ((a.size-window_len)//stride_len) + 1

    return np.array([a[s:(s+window_len)] for s in np.arange(0, a.size, stride_len)[:n_strides]])



gen_strides(L, stride_len=2, window_len=4)
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', usecols=['crim', 'medv'])

print(df.head())