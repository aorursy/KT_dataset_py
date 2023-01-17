# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


# Modules for importing data 

import pandas as pd

import numpy as np



# Modules for data visualizaion 

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use(["seaborn-ticks", "seaborn-paper"])

sns.set_palette('husl')



# Modules for regression 

from sklearn.model_selection import train_test_split,  cross_val_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestClassifier 

from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier

from wordcloud import WordCloud
import pandas as pd

af_crises = pd.read_csv("../input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv")
# Rows and Columns of Dataset

rows, columns = af_crises.shape

print(f'There are {rows} of rows and {columns} columns in data') 
# Type of data in each columns

af_crises.info()
# Basic arthimetics of data

af_crises.describe()
# Null data check

af_crises.isnull().sum()
# Column headings in the data 

for i in af_crises.head():

    print(i)
Currency_crisis = af_crises.query("currency_crises == '1'").groupby("country")["currency_crises"].count()



Inflation_crisis = af_crises.query("inflation_crises == '1'").groupby("country")["inflation_crises"].count()



Bank_crisis = af_crises.query("banking_crisis == 'crisis'").groupby("country")["banking_crisis"].count()



sns.set(style="white", context="talk")

rs = np.random.RandomState(8)



#Set up the matplotlib figure

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)



# Generate some sequential data

x = Currency_crisis.index

y1 = Currency_crisis.values

sns.barplot(x=x, y=y1,palette="rocket", ax=ax1)

ax1.axhline(0, color="k", clip_on=False)

ax1.set_ylabel("Currency crisis")



# Center the data to make it diverging

y2 = Inflation_crisis.values

sns.barplot(x=x, y=y2, palette="vlag", ax=ax2)

ax2.axhline(0, color="k", clip_on=False)

ax2.set_ylabel("Inflation crisis")



# Randomly reorder the data to make it qualitative

y3 = Bank_crisis.values

sns.barplot(x=x, y=y3, palette="deep", ax=ax3)

ax3.axhline(0, color="k", clip_on=False)

ax3.set_ylabel("Bank crisis")



# Finalize the plot

sns.despine(bottom=True)

plt.setp(f.axes, yticks=[])

plt.tight_layout(h_pad=2)

plt.xticks(rotation = 90)
sns.set(style="white", context="talk")

currency_cloud = WordCloud(background_color = "Red", max_words = 200).generate(str(Currency_crisis))

fig = plt.figure(figsize = (12, 8))

plt.imshow(currency_cloud, interpolation = 'bilinear')

plt.axis("off")

plt.tight_layout(pad=0)
sns.set(style="white", context="talk")

bank_cloud = WordCloud(background_color = "yellow", max_words = 200).generate(str(Bank_crisis))

fig = plt.figure(figsize = (12, 8))

plt.imshow(bank_cloud, interpolation = 'bilinear')

plt.axis("off")

plt.tight_layout(pad=0)
sns.set(style="white", context="talk")

bank_cloud = WordCloud(background_color = "pink", max_words = 200).generate(str(Inflation_crisis))

fig = plt.figure(figsize = (12, 8))

plt.imshow(bank_cloud, interpolation = 'bilinear')

plt.axis("off")

plt.tight_layout(pad=0)
Egypt = af_crises.query("country == 'Egypt'")

Angola = af_crises.query("country == 'Angola'")

Central_African_Republic = af_crises.query("country == 'Central African Republic'")

Ivory_Coast = af_crises.query("country == 'Ivory Coast'")

Kenya = af_crises.query("country == 'Kenya'")

Mauritius = af_crises.query("country == 'Mauritius'")

Morocco = af_crises.query("country == 'Morocco'")

Nigeria = af_crises.query("country == 'Nigeria'")

South_Africa = af_crises.query("country == 'South Africa'")

Tunisia = af_crises.query("country == 'Tunisia'")

Zambia = af_crises.query("country == 'Zambia'")

Zimbabwe = af_crises.query("country == 'Zimbabwe'")



plt.style.use("seaborn-ticks")

sns.set(style="whitegrid")

fig, axs = plt.subplots(4,3, sharex=True, figsize = (12, 8))

sns.lineplot(x= 'year', y='exch_usd', style="independence", data=Egypt, ax=axs[0,0], color = "r")

sns.lineplot(x= 'year', y='exch_usd', style="independence" , data=Angola, ax = axs[0,1], color = "b")

sns.lineplot(x= 'year', y='exch_usd', style="independence" , data=Central_African_Republic, ax=axs[0,2], color = "g")

sns.lineplot(x= 'year', y='exch_usd', style="independence" , data=Ivory_Coast, ax=axs[1,0], color = "y")

sns.lineplot(x= 'year', y='exch_usd', style="independence" , data=Kenya, ax=axs[1,1], color = "c")

sns.lineplot(x= 'year', y='exch_usd', style="independence" , data=Mauritius, ax=axs[1,2], color = "m")

sns.lineplot(x= 'year', y='exch_usd', style="independence" , data=Morocco, ax=axs[2,0],color = "w")

sns.lineplot(x= 'year', y='exch_usd', style="independence" , data=Nigeria, ax=axs[2,1], color = "k")

sns.lineplot(x= 'year', y='exch_usd', style="independence" , data=South_Africa, ax=axs[2,2], color = "r")

sns.lineplot(x= 'year', y='exch_usd', style="independence" , data=Tunisia, ax=axs[3,0], color = "b")

sns.lineplot(x= 'year', y='exch_usd', style="independence" , data=Zambia, ax=axs[3,1], color = "y")

sns.lineplot(x= 'year', y='exch_usd', style="independence" , data=Zimbabwe, ax=axs[3,2], color = "c")

fig.set_figwidth(25)

fig.set_figheight(25)

fig.tight_layout()
# Model building 

def encode(af_crises):

    for column in af_crises.columns[af_crises.columns.isin(["cc3", "country", "banking_crisis"])]:

        af_crises[column] = af_crises[column].factorize()[0]

    return af_crises



af_crises_en = encode(af_crises.copy())
# Prediction model 

x = np.array(af_crises_en.iloc[:,[3,4,5,6,7,8,9,10,11,12]])

y = np.array(af_crises_en["banking_crisis"])
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 52)
# Random Forest:

rf = RandomForestClassifier(n_estimators = 100)



rf_model = rf.fit(x_train, y_train)



rf_predict = rf.predict(x_test)



r2_score(y_test, rf_predict)
# Classification Matrix

confusion_matrix(y_test,rf_predict)
# Model Accuracy 

accuracy_score(y_test, rf_predict)
# Model report 

classification_report(y_test, rf_predict)
tree = DecisionTreeClassifier(min_samples_leaf = 0.001)



dt_model = tree.fit(x_train, y_train)



tree_predict = tree.predict(x_test)



r2_score(y_test, tree_predict)
# Classification Matrix

confusion_matrix(y_test,tree_predict)
# Model Accuracy 

accuracy_score(y_test,tree_predict)
# Model report

classification_report(y_test,tree_predict)