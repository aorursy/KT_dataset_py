# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpimport unittest
import pandas
import numpy as np

pandas.set_option('display.float_format', lambda x: '%.6f' % x)

btcMax = 0
ethMax = 0

def fun(x):
    return x + 1

def fun(x):
    if ("/BTC" in str(x.name)):
        return x["maxV"] * btcMax
    if ("/ETH" in str(x.name)):
        return x["maxV"] * ethMax
    return x["maxV"]

df = pandas.read_csv("../input/prices.csv", index_col=0)
btcMax = df[df.index.str.contains("BTC/USD")]["maxV"][0]
ethMax = df[df.index.str.contains("ETH/USD")]["maxV"][0]
b = df[df.index.str.contains("/BTC")]
b['usd'] = b["maxV"].apply(lambda x: x*btcMax)
#print(b)

e = df[df.index.str.contains("/ETH")]
e['usd'] = e["maxV"].apply(lambda x: x * ethMax)
#print(e)

df['usd'] = df.apply(fun, axis=1)
#df['usd'] = df.apply((lambda x: x["maxV"] * btcMax if "/BTC" in str(x.index) else x["maxV"] * ethMax if "/ETH" in str(x.index) else None), axis=1)

print(df)