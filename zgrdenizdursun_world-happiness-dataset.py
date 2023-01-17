# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/world-happiness/2016.csv")
data.head()
data.info()
f,ax = plt.subplots(figsize = (10,10))

sns.heatmap(data.corr() , cmap = "BuPu" , annot = True , linewidth = .5 , fmt= ".2f" , vmin= 0 , vmax= 1 );

#cmap = colormap , renk belirtecidir. Yanındaki kodları internettern kolayca kopyala yapıştır yapabiliriz.

#vmin, değer çubuğunun minimum değeri vmax ise maximum değeridir.

data.head()
sns.set_style("whitegrid") 

#grid ekledik.

f, ax = plt.subplots(figsize=(6.5 ,6.5))

sns.despine(f, left=True, bottom=True ) 

#grafikteki sol ve alt çizgileri çıkarttık.

sns.scatterplot(data=data, x = "Happiness Score" , y = "Health (Life Expectancy)" ,hue = "Freedom" ,size = "Economy (GDP per Capita)",palette="Spectral",linewidth=0);
plt.figure(figsize=(8,8))

x = data["Region"].value_counts().index

y = data["Region"].value_counts().values

sns.barplot(x=x , y=y, palette =sns.color_palette("PuOr" , 10) )

plt.xticks(rotation = 90)

plt.show()
data.head()
plt.figure(figsize = (7,7))

data["Trust (Government Corruption)"].plot(kind="line", label = "Trust" , color = "#52575d", grid = True , linestyle = "-.")

data["Generosity"].plot(label="Generosity" , color = "#fddb3a" , grid = True , linestyle = "-")

plt.legend()

plt.show()