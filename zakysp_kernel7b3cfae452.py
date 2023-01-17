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
ayamgoreng1=pd.read_csv('../input/car-consume/measurements.csv')
ayamgoreng1['distance']=ayamgoreng1['distance'].str.replace(',','.').astype(float)
ayamgoreng1['consume']=ayamgoreng1['consume'].str.replace(',','.').astype(float)
ayamgoreng1['temp_inside']=ayamgoreng1['temp_inside'].str.replace(',','.').astype(float)
ayamgoreng1
df.info()
coba2=ayamgoreng1.drop(['specials','temp_outside','refill liters','refill gas'],axis=1)
coba2
#fuelconsumption dari penggunaan AC
coba2['distance'].plot.hist()
plt.show()
coba2.describe()
coba3min=coba2.query('speed==14.000000')
coba3min
coba3max=coba2.query('speed==90.000000')
coba3max
#penggunaan bhan bakar di ujan yg minus
minus=coba2.query('rain==0.000000')
minus
#penggunaan bhan bakar di ujan MINIMUM

sns.lmplot('distance','consume',data=minus)
plt.show()
#penggunaan bahakan bakar dengan AC minumum!
minusac=coba2.query('AC==0.000000')
minusac
#penggunaan bahakan bakar dengan AC minumum!
sns.lmplot('distance','consume',data=minusac)
plt.show()