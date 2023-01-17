# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
['googleplaystore_user_reviews.csv', 'googleplaystore.csv']

import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import os as o


columns= [
    'Apps',
    'Category',
    'Rating',
    'Reviews',
    'Size',
    'Installs',
    'Type',
    'Price',
    'Content Rating',
    'Genres',
    'Last Updated',
    'Current Ver',
    'Android Ver'
]
df=pd.read_csv('../input/googleplaystore.csv')

kfold = KFold(3, True, 1)
#•	Veri Keşfi ve Görselleştirme
#o	Describe, info, head, tail, shape kullanımı
#o	Histogram bakma
#o	Korelasyon görme ve sonuçlarını yorumlama
#	Düz metin ve seaborn ısı haritası ile görülmeli
#	Korelasyonları yüksek olan 2 tane öznitelik için plotting (çizim işlemi) gerçekleştirilmelidir

df.head()
df.dtypes
df.isnull().any()
df.isnull().any()
df.describe()
mapping = {'no': 0., 'yes':1., 'False.':0., 'True.':1.}
df.replace({'international plan' : mapping, 'voice mail plan' : mapping, 'googleplaystore' : mapping}, regex = True, inplace = True)
#Veri ön işleme adım 2 
#df.drop('', axis = 1, inplace = True)
df.shape
df.describe()
df.info()
df.tail(10)
df.sample(11)
#histogram
import matplotlib.pyplot as plot
num_bins = 20
df.hist(bins=num_bins, figsize=(5,5))
#korelasyon
import matplotlib.pyplot as plot
plot.matshow(df.corr())
import seaborn as sns
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#on işlerme
df.isnull().sum()
# boş null değer bulma
df.isnull().sum().sum()
def eksik_deger_tablosu(df): 
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum()/len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    return mis_val_table_ren_columns
eksik_deger_tablosu(df)
#null değerleri doldur
df.isnull().sum()
#group by 
df.groupby('App')['Category'].apply(lambda x: x.count())
df.groupby('App')['Rating'].apply(lambda x: np.mean(x))
#istatsilik fonk
df['Rating'].mean()
df.mean(axis=0,skipna=True)
df.mean(axis=0,skipna=True)
df['Rating'].mode()
df['Rating'].std()
df.cov()
df.corr()
df.plot(x='Type', y='Rating', style='o')
def eksik_deger_tablosu(df): 
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum()/len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    return mis_val_table_ren_columns
df
tr = len(df) * .3
df.dropna(thresh = tr, axis = 1, inplace = True)
df
df['Rating'] = df['Rating'].fillna('boş')
df
df['Rating'] = df['Rating'].fillna('boş')
df
df
df1=df[df['Type']=='Free']
df2=df[df['Type']=='Paid']
df1.shape
df2.shape
df = df2.append(df1[:800])
df['popularity']= df['Installs']
df
df.shape
df.corr()
import pandas
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection

#Eğitim için ilgili öznitlelik değerlerini seçilmesi
X = df.iloc[:, :-1].values

#Sınıflandırma öznitelik değerlerini seçilmesi
Y = df.iloc[:, -1].values
X
#model eğitimi
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
X_train
X_validation
Y_validation
Y_train
X
Y
# navie bayes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
from sklearn.metrics import r2_score
import pandas as pd
New_df = pd.DataFrame(df.groupby("Category")["App"].sum())
New_df.head()
New_df.tail()
#Model oluşturmamız için gereken kütüphaneler:
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#lineer Regresyon için modelimizi oluşturuyoruz.
lr = LinearRegression()

# Derecesi 4 olan bir fonksiyon kullanacağız
pf = PolynomialFeatures(degree=4)