# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")

data.head()

#veri import edildi ilk 5 elemana bakıldı
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True,fmt=".1f",ax=ax)

# verinin korelasyon haritası oluşturuldu
data.shape

#verinin öznitelik sayısı öğrenildi
data.High.plot(kind="line",color="red",linestyle=":",label="High")

plt.legend()

#verinin high özniteliği için lineplot çizimi yapıldı
data.plot(kind="scatter",x="Open",y="Close")

plt.xlabel("Open")

plt.ylabel("Close")

plt.title("scatter plot")

#verinin Open ve Close öznitelikleri çizdirildi.Korelasyonunun 1 e yakın olduğu yorumlandı.
data.Low.plot(kind='hist',bins=50,figsize=(5,5))
data_frame=data[["Open"]]   # data frame oluşturduk

print(type(data_frame))
data[np.logical_and(data["Open"]>200,data["Volume_(Currency)"]<100)] 

#filtre oluşturuldu veriye uygulandı