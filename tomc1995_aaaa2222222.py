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



yhat=-2646.6911 + 1083.3994*x1 + 810.3251*x2 -15.2368*x3 +365.407*x4



deltay= y - yhat

deltay = np.random.normal(size = 49000)

plt.hist(deltay, normed=True, bins=100)