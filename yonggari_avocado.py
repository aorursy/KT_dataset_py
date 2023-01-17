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
import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import pylab as pl



avoc = pd.read_csv('../input/avocado-prices/avocado.csv')



#SEE IF NAN VALUES EXISTS

print(avoc.isna().any())



#COUNT MISSING VALUES

print(avoc.isna().sum())

#IF WE HAD NaN VALUES, WE COULD USE avoc.dropna() TO DROP THEM











print(avoc.head())



#DELETES THE 'UNNAME' COLUMN

av = avoc.drop(columns = ['Unnamed: 0'])

print(av.info())

print(av.describe())

print(av.shape)

#RETURNS NUMPY REPRESENTATION

print(av.values)







#SORTS VALUES BASED ON ASCENDING AVG PRICE AND DESCENDING YEAR

print(av.sort_values(["AveragePrice", "year"], ascending=[True, False]))



#SUBSETTING (GET A SLICE OF O.G. DATAFRAME) MULTIPLE COLUMNS

print(av[["AveragePrice","year"]])





#PRINTS ALL THE ROWS WITH TYPE = ORGANIC

print(av[av["type"]=="organic"])





#PRINT ONLY THE ROWS WITH PRICE < 1

print(av[av["AveragePrice"]<1])





#PRINTS ALL THE ROWS WITH TYPE = ORGANIC AND AVG. PRICE BELOW $1.00

print(av[(av["AveragePrice"]< 1) & (av["type"]=="organic")])









#SUBSET THOSE FROM CALIFORNIA OR CHICAGO

regionFilter = av["region"].isin(["California", "Chicago"])

print(av[regionFilter])



#SUBSET THOSE FROM CALIFORNIA OR CHICAGO IN THE YEAR 2016 OR 2017

regionFilter = av["region"].isin(["Chicago", "California"])

yearFilter = av["year"].isin(["2016", "2017"])

print(av[regionFilter & yearFilter])







#DELETES ALL DUPLICATE NAMES

temp = av.drop_duplicates(subset=["year"])

print(temp)









#COUNT NUMBER OF AVOCADO IN EACH YEAR IS DESCENDING ORDER

av_year = av["year"].value_counts(sort=True, ascending = False)



av_year.plot(kind = "bar",rot=45,title="Count Per Year")









#GROUP BY MULTIPLE COLUMNS AND PERFORM MULTIPLE SUMMARY STATISTIC OPERATIONS

print(av.pivot_table(index=["year","type"], aggfunc=[min,max,np.mean,np.median], values="AveragePrice"))









#HISTOGRAPH

av["AveragePrice"].hist(bins=20)

plt.show()











#BAR PLOT

regionFilter = av.groupby("region")["AveragePrice"].mean().head(10)

print(regionFilter)



regionFilter.plot(kind = "bar",rot=45,title="Average price in 10 regions")











#DISTRIBUTION OF PRICE

pl.figure(figsize=(12,5))

pl.title("Distribution Price")

ax = sns.distplot(av["AveragePrice"], color = 'g')















#SCATTER PLOT

av.plot(x="AveragePrice", y="Total Volume", kind="scatter")











#BOX PLOT OF AVOCADO TYPES

sns.boxplot(y="type", x="AveragePrice", data=av, palette = 'pink')















#FACTOR PLOT OF REGION VS AVG PRICE OVER THE YEARS FOR ORGANIC

org = av['type']=='organic'

orga = sns.factorplot('AveragePrice','region',data=av[org],

                   hue='year',

                   size=13,

                   aspect=0.8,

                   palette='magma',

                   join=False,)





#FACTOR PLOT OF REGION VS AVG PRICE OVER THE YEARS FOR CONVENTIONAL

con = av['type']=='conventional'

conv = sns.factorplot('AveragePrice','region',data=av[con],

                   hue='year',

                   size=13,

                   aspect=0.8,

                   palette='magma',

                   join=False,

              )











#HEATMAP OF ORGANIC VS CONVENTIONAL AVOCADOS

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

dicts = {}

label.fit(av.type.drop_duplicates()) 

dicts['type'] = list(label.classes_)

av.type = label.transform(av.type)





cols = ['AveragePrice','type','year','Total Volume','Total Bags']

cm = np.corrcoef(av[cols].values.T)

sns.set(font_scale = 1.7)

hm = sns.heatmap(cm,cbar = True, annot = True,square = True, fmt = '.2f', annot_kws = {'size':15}, yticklabels = cols, xticklabels = cols)


