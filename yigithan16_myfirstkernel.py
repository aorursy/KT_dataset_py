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
df = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')
df.info()
df.shape
df.head()
df.tail()
df.describe().T #sayısal verileri istatisiksel olarak açıklar. yani bool ve object olanlar dahil değildir
df.sample(10) #rastgele 10 gözlemi görüntüledik
df.isna().sum() #değişkenlerde kaç tane eksik gözlem olduğunu gösterir
df.count() #her değişkende kaç adet değer olduğunu gördük
df.corr()
df["Type 1"].unique() #type 1 değişkeni nelerden oluşuyor onu gördük
df["Type 2"].unique()
df["Name"].unique()
df["Legendary"].unique() #bool değişken olduğunu biliyoduk ama yine de sadece efsane veya efsane değil olarak mı belirtilmiş diye baktık
df.groupby(["Legendary"]).mean() #burda efsane olanlar ve olmayanların ortalama değişken değerlerini gördük.
df.groupby(["Type 1"]).mean() #burda ise type 1 bazında ortalama olarak inceleme yaptık. 
df.groupby(["Legendary"]).describe()["HP"] #Legendary değişkeninin HP bazında istatiksel özellikleri.
df[(df["Legendary"] == 0) & (df["HP"] == 255)] #yukarda gözüme bir şey çarptı. efsane olmayanların max HP'si, efsane olanardan fazla. burda bu pokemonu bulduk
df.sort_values('HP', axis=0, ascending=False).head(10) #pokemonları HP değişkeni bazında büyükten küçüğe sıraladık ilk 10 değeri gözlemledik
df.query('Legendary==0 & HP>149') #burda da query kullanarak efsane olmayan ve hp'si 149'dan büyük olan pokemonları gördük.