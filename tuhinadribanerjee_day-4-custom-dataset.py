# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#curve = pd.read_csv("../input/curve-dts/curve.csv")

#curve.head()

student = pd.read_csv("../input/student-dataset/student.csv")

student



classF = ("Intel")

classG = ("Leader")

classH = ("Speaker")

classI = ("null[]")

distance4 = 0

student.head()



plt.scatter(x = student.Communication, y = student.Class)

plt.show()
# K Means # 

k = 3

testdata = [5, 4.5]





plt.scatter(x = student.Communication, y = student.Aptitude)

plt.show()







m = (5, 6, 7)

n = (8, 9, 9)





distance = math.sqrt(sum([(m - n) ** 2 for m, n in zip(m, n)]))

distance2 = distance+k

distance3 = distance2+k

print("Test Case & Comparison Data :",k,"and",testdata)

print("--")

print("Distance | Class: ",distance,":",classF, "when K =" , k,)

print("Distance | Class: ",distance2,":",classG, "when K =" , k,)

print("Distance | Class: ",distance3,":",classH, "when K =" , k,)

print("Distance | Class: ",distance4,":",classI, "when K =" , k,)

print("--")



student = pd.read_csv("../input/student-dataset/student.csv")

subset = student.drop(['Name'], axis=1)

subset.head()



testdata = [5, 4.5]



lst_dist = []



for ind in subset.index:

    print(subset['Aptitude'][ind], subset['Communication'][ind], subset['Class'][ind])

    dist_row = np.sqrt(np.square(testdata[0] - subset['Aptitude'][ind]) + np.square(testdata[1] - subset['Communication'][ind]))

    lst_dist.append([dist_row, subset['Class'][ind]]) 

    

df = pd.DataFrame(lst_dist)

df.columns = ('Distance', 'class')

df_sorted = df.sort_values('Distance')

student



print("--")

k = 3

df_sorted_kval = df_sorted.head(k)

print(df_sorted_kval)

print("--")

selected_class_count = df_sorted_kval.groupby(['class']).count()

print(selected_class_count)






