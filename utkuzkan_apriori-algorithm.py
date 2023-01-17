# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mlxtend.frequent_patterns import apriori, association_rules 
import random
import matplotlib.pyplot as plot 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
orders =pd.read_excel("/kaggle/input/sample.xlsx")
#Şube Kodu 32 olanları filtrele
filter1=orders["Şube No"]==32
branch = orders[filter1]
branch.info()
#Dataframe içinde NAN data NA data var mı diye kontrol yapıyoruz
branch.columns[branch.isnull().any()]
branch["Stok Kodu"].value_counts()
branch["Fiş No"].value_counts()
#1025 transaction 850 farklı stok kodu var
#1025/850 neredeyse transaction başına 1.2 stok kodu düşüyor
#Böyle br veride analiz yapmak zor olduğu için mevcut veri üzerinde sentetik bir veri oluşturacağız
#Bu sentetik veri 1025 transaction da yaklaşık 10 tane seçilen Stok kodu nu random bir şekilde transaction başına 1-6 arasında farklı olacak şekilde 
# atama yapılacak.Yani her fişte 10 üründen 1 ile 6 arasında farklı olanları seçip  alış veriş yapılmış gibi bir senaryo oluşturulcak
#Bu oluşturulan yeni data Label encoding yöntemi uygulanarak bir diğer dataFrame e aktarılacak

# 10 tane ürün seçiyoruz
products = branch["Stok Kodu"][:10].values.tolist()
products
branch.head(10)
products
def getRandomProducts():
    random_products = []
    [random_products.append(x) for x in random.sample(products, np.random.randint(1,7)) if x not in random_products]
    return random_products
dict = {}
for index,row in branch.iterrows():
    key=row["Fiş No"] # Eğer key önceden eklenmişse ekleme,Fiş Noları birden çok kez geldiği için engelliyoruz   
    if key in dict:
        continue
    #check key exists or not
    if key not in dict:
        dict[row["Fiş No"]] = []
        #append value
        for i in getRandomProducts():
            dict[row["Fiş No"]].append(i)
    else:
        for i in getRandomProducts():
            dict[row["Fiş No"]].append(i)

#Transaction başına Stok Kodu ataması yaptık
#dict

#Create new DataFrame to add encoding variable
#Products we specified 10 product then we will create columns
new_df = pd.DataFrame(columns=products)
new_df
#add encoding values to new DataFrame
for key, value in dict.items():
    temp=[]
    for column in new_df.columns:
            for i in value:
                if i == column:
                    temp.append(column)
    encoded_rows = [] 
    for column in new_df.columns:
        if column in temp:
            encoded_rows.append(1)
        else:
            encoded_rows.append(0)
    new_df=new_df.append(pd.Series(encoded_rows, index=new_df.columns), ignore_index=True)
    

del dict    

new_df
#min support hesabı
#56/1025 #her ürünün hafta da 4 tane satıldığını hesap edersek elimzdeki veri 2 haftalık 14*4= 56/toplam transaction(1025) 

frequent_itemsets = apriori(new_df, min_support=0.054634146341463415, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()