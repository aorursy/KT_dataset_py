# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import numpy as np
import pandas_datareader as web
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Apple AAPL
# Altın GC=F
# Gümüş SI=F
# Dolar TRY=X
# Bitcoin BTC-USD
# Ethereum ETHUSD=X
# LINK LINK-USD
# XRP XRP-USD
# Şişecam SISE.IS


data = web.DataReader('SI=F', data_source='yahoo', start='2018-01-01', end='2020-10-15') 
data.shape
data


period = 20
data['SMA'] = data['Close'].rolling(window=period).mean()
data['STD'] = data['Close'].rolling(window=period).std()
data['Yukarı'] = data['SMA'] + (data['STD'] * 2)
data['Aşağı'] = data['SMA'] - (data['STD'] * 2)


sutun_liste = ['Close', 'SMA', 'Yukarı', 'Aşağı']
data[sutun_liste].plot(figsize=(25,12))


yeni_veri = data[period-1:]
yeni_veri


def sinyal_al(yeniveriler):
    al_sinyal = []
    sat_sinyal = []
    
    for i in range(len(yeniveriler['Close'])):
        if yeniveriler['Close'][i] < yeniveriler['Aşağı'][i]:
            al_sinyal.append(yeniveriler['Close'][i])
            sat_sinyal.append(np.nan)
        elif yeniveriler['Close'][i] > yeniveriler['Yukarı'][i]:
            al_sinyal.append(np.nan)
            sat_sinyal.append(yeniveriler['Close'][i])
        else:
            al_sinyal.append(np.nan)
            sat_sinyal.append(np.nan)
    return(al_sinyal,sat_sinyal)

yeni_veri['Al'] = sinyal_al(yeni_veri)[0]
yeni_veri[-20:]







