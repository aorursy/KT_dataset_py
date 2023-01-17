# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/weather-dataset-rattle-package/weatherAUS.csv')



data.tail()
data.isnull().sum()
data.drop(['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'], axis=1, inplace=True)





for i in range(0, len(data.columns)):

    if i==16 or i==18:

        continue  # Çünkü raintoday ve raintomorrow column'ları yes ve no'dan oluşuyor.

    

    mean = data.iloc[:, i].mean()

    

    data.iloc[:, i] = [mean if str(each)=='nan' else each for each in data.iloc[:, i]]  # Boş data'lara, aynı column'daki diğer data'ların ortalama değerini ekledim.



    

data.RainToday = [1 if each=='Yes' else 0 for each in data.RainToday]

data.RainTomorrow = [1 if each=='Yes' else 0 for each in data.RainTomorrow]





data.isnull().sum()  # you can see alteration
y = data.RainTomorrow.values

x_head = data.drop('RainTomorrow', axis=1).values
#normalize

x = (x_head - np.min(x_head)) / (np.max(x_head) - np.min(x_head))
#split

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=3)



knn.fit(x_train, y_train)

knn.score(x_test, y_test)
n = 10

score_list = []



for i in range(1, n):

    percent = (100 / (n-1)) * i

    print(f"%{int(percent)}")

    

    knn2 = KNeighborsClassifier(n_neighbors = i)

    knn2.fit(x_train, y_train)

    

    score_list.append(knn2.score(x_test, y_test))



print("Max score:", max(score_list))

print("Best K number:", score_list.index(max(score_list)) + 1)

    

plt.plot(range(1, n), score_list)

plt.xlabel('n_neighbors')

plt.ylabel('score')

plt.show()