# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

%matplotlib inline

from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.ensemble import ExtraTreesRegressor
df = pd.read_csv("../input/playstore-analysis/googleplaystore.csv")
df.head()
df.shape
df.describe()
df.info()
#dropping rows with null values

df.dropna(how='any',inplace=True)
df['Reviews'] = df['Reviews'].astype(int)
df.head()
df['Size'].unique()
def Kb_to_Mb(size):

    if size.endswith('M'):

        return float(size[:-1])

    elif size.endswith('k'):

        return float(size[:-1])/1000

    else:

        return size
df['Size'] = df['Size'].apply(lambda x: Kb_to_Mb(x))
df['Size'].value_counts()
df['Size'].fillna(method="bfill",inplace=True)
df['Size'].replace({'Varies with device':11.0},inplace=True)
df.head()
df.rename(columns={'Size_MB':'Size_MB'},inplace=True)
df.head()
df["Installs"] = df['Installs'].str[:-1]
df['Installs'] = df['Installs'].apply(lambda x: x.replace(",",""))
df['Installs'] = df['Installs'].astype(int)
df.head()
df['Type'].value_counts()
df['Price'] = df['Price'].apply(lambda x: x if x == '0' else x[1:])
df.Price= df.Price.astype(float)
df.rename(columns={'Price':'Price_in_$'},inplace=True)
df.head()
df['Content Rating'].unique()
df['Category'].unique()
df['Current Ver'].value_counts()
df['Android Ver'].value_counts()
def update_version(version):

    if version.endswith('up'):

        ver = version.split(' ')

        ver = ver[0]

        return ver

    elif version == "Varies with device":

        ver = '4.1.0'

        return str(ver)

    else:

        ver = version.split('-')

        ver = ver[0]

        return str(ver)

        
df['Android Ver'] = df['Android Ver'].apply(lambda x: update_version(x))
df.head()
df['Genres'].unique()
plt.rcParams['figure.figsize'] = (11,9)

df.hist()

plt.show()
#correlation between variables

plt.rcParams['figure.figsize'] = (12,9)

sns.heatmap(df.corr(),annot=True,cmap="Reds")

plt.show()
df.head()
plt.rcParams['figure.figsize'] = (12,9)

sns.barplot(x='App',y='Installs',hue='Reviews',data = df.sort_values('Installs',ascending=False)[:10])

plt.legend(loc='center')

plt.xticks(rotation=90)

plt.show()
plt.rcParams['figure.figsize'] = (12,9)

sns.barplot(x='Rating',y='App',data = df.sort_values('Rating',ascending=False)[:10])

plt.show()
plt.rcParams['figure.figsize'] = (12,9)

sns.barplot(x='Rating',y='Category',data = df.sort_values('Rating',ascending=False)[:10])

plt.show()
plt.rcParams['figure.figsize'] = (12,9)

sns.countplot(x=df['Type'],data = df)

plt.show()
plt.rcParams['figure.figsize'] = (50,20)

sns.countplot(x=df['Category'],hue='Type',data = df)

plt.show()
plt.rcParams['figure.figsize'] = (20,9)

sns.countplot(x='App',hue='Installs',data = df.sort_values('Installs',ascending=False)[:5])

plt.show()
plt.rcParams['figure.figsize'] = (25,9)

sns.barplot(x='App',y='Price_in_$',data = df.sort_values('Price_in_$',ascending=False)[:10])

plt.show()
plt.rcParams['figure.figsize'] = (20,12)

sns.barplot(x='Installs',y='Category',data = df.sort_values('Installs',ascending=False))

plt.show()
df.columns
x = df[['Reviews','Size','Installs','Price_in_$']]

y = df[['Rating']]
bstfeatures = ExtraTreesRegressor()

fit = bstfeatures.fit(x,y)

print(fit.feature_importances_)

feat_imp = pd.Series(bstfeatures.feature_importances_,index=x.columns)

feat_imp.plot(kind='barh')

plt.title("Most important features")

plt.show()