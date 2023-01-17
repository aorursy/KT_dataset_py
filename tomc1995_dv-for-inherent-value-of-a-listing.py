import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from statsmodels.formula.api import ols

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_json('../input/train.json')



for a in df["bedrooms"]:

    if a == 0:

        df['Studio']=1

    else: 

        df['Studio']=0

   

df['size'] = (6*6*df["bathrooms"] + 9*9*df["bedrooms"] + 50 + 80 + 50).astype(int)



df["size"] = df["size"].astype(int)

		

df['price_size'] = (df["price"]) / (df["size"])





df["num_photos"] = (df["photos"].apply(len)).astype(int)



df["num_photos"] = df["num_photos"].astype(int)



x1= df["bathrooms"]

x2=df["bedrooms"]

x3=df["num_photos"]

x4=df['price_size']

y= df["price"]



#yhat= 7.4100  + 0.4087 *x1 + 0.1256*x2 

yhat= 2.0346 + 0.3552 *x1 - 0.1157*x2 





yl=np.log(y)

yll=yl/df["size"]

deltay= yll - yhat

deltay = np.random.normal(size = 49000)

plt.hist(deltay, normed=True, bins=100)



for a in deltay:

    if a <= 1.5:

        df['inherent']=1

    else: 

        df['inherent']=0