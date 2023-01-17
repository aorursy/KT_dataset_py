# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/FINAL_FROM_DF.csv')
data.head()
#Preprocess and create new features
data['TRADE_DATE']= pd.to_datetime(data.TIMESTAMP,dayfirst=True)
data['DAILY_ABS_PCNT_CHANGE'] = (data.CLOSE-data.OPEN)*100/data.OPEN
data.head()
data.set_index('TIMESTAMP',inplace=True)
data.head()
#Find all banking stocks
list_size = len(data.SYMBOL)
bank_stocks = []
i=0
while i < list_size:
    string_symbol = data.SYMBOL[i]
    #print(string_symbol)
    if string_symbol.find("BANK") is not -1: 
         #We found a bank but is it already in our list if not we append it else skip it
            if string_symbol not in bank_stocks:
                bank_stocks.append(string_symbol)
    i+=1
    if i ==1000:
        break

# Create a data frame of all banking stocks - Index= Date, Column= Stock Ticker, Value= Closing price
temp = pd.DataFrame()
for bank in bank_stocks:
   # print('Extracting  data for :{}'.format(bank))
    a = data[data.SYMBOL == bank]
    temp = temp.append(a)
temp = temp[['SYMBOL','CLOSE']]
temp.sort_index(inplace=True)
#Import relevant libraries
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as py #ifthis is not offline then plotly will ask for a account.
py.init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings('ignore')

Banks_to_plot = []

for stock_ticker in bank_stocks:
    bank = go.Scatter(
            x = temp[temp.SYMBOL==stock_ticker].index,
            y = temp[temp.SYMBOL==stock_ticker].CLOSE,
            name= stock_ticker,
            mode='lines')
    Banks_to_plot.append(bank)

#Plot all bank stocks                      
py.iplot(Banks_to_plot)
temp1 = temp.reset_index()
temp1 = temp1.pivot_table(values='CLOSE',columns='SYMBOL',index='TIMESTAMP')
temp1.head()
plt.figure(figsize=(15,10))
sns.heatmap(temp1.corr(method='pearson'),annot=True,linewidth=.5)
#Next Steps - Visualize Volatility of stocks
