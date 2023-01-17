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
import pandas  as pd 
import numpy as np 

df = pd.read_csv("../input/Advertising.csv")

data_type = df.dtypes

print(data_type)
data_null = df.empty
if data_null == False :
    print("Veri seti içerisinde boş değer yoktur.") 
else :
    print ("Veri seti içerisinde boş değer vardır.")

 
df.head()
df = df.iloc[:,1:len(df)]

df.head()
df.info()

df_istatistik = df.describe()
print (df_istatistik)
import matplotlib.pyplot as plt

import seaborn as sns 

f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(df.corr(),annot=True,fmt=".1f",linewidths=1,ax=ax)
plt.show()
import seaborn as sns 
sns.jointplot(x = "TV" , y = "sales" , data = df ,kind = "reg");
X_= df.drop(["radio","newspaper"],axis=1).astype("float64")
from sklearn.linear_model import LinearRegression
X = df [["TV"]]
X.head()
y = df[["sales"]]
reg = LinearRegression()   # linear regresyon fonksiyonundan bir model nesnesi oluşturalım.
model = reg.fit(X,y)
model
str(model)
dir(model)  # bu nesnenin içerisinden alabilecek olduğumuz bazı bilgiler sunuldu.
#intercept : sabit (b0)
#coef : katsayı modele ilşkin katsayıyı buradan çekebiliriz.(b1)

model.intercept_
model.coef_

model.score(X,y)     
import seaborn as sns 
import matplotlib.pyplot as plt 
g = sns.regplot(df["TV"] , df["sales"] , ci = None , scatter_kws = {"color" : "r", "s" :9})
g.set_title ("Model denklemi: Sales = 7.03 + TV*0.05")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10 ,310)
plt.ylim(bottom = 0);
model.predict([[165]])
model.intercept_ + model.coef_*165
model.predict(new_data)
model.predict([[400]])
y.head()
model.predict(X)[0:6]
gercek_y = y[0:10]
tahmin_edilen_y = pd.DataFrame(model.predict(X)[0:10])
hatalar = pd.concat([gercek_y , tahmin_edilen_y] ,axis = 1)
hatalar.columns = ["gercek_y","tahmin_edilen_y"]
hatalar
hatalar["hata"] = hatalar["gercek_y"] - hatalar["tahmin_edilen_y"]
hatalar
hatalar["hata_kareler"]= hatalar["hata"]**2
hatalar
np.mean(hatalar["hata_kareler"])