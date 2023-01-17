# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/food.csv")

median = data['income'].median()

data['income'].replace(to_replace='NaN',value=median,inplace=True)

  

#Creating Profiles[Gender,marital_status,income] 

p1 = data.groupby(['Gender','marital_status','income'])['Gender'].count().keys()

#now, lets divide the datase based in profile

dados = []

for i in range(0,len(p1)):

         #print(i)

         valor = data[(data['Gender'] == p1[i][0]) & (data['marital_status'] == p1[i][1]) &

                      (data['income'] == p1[i][2])]

         dados.append(valor)

#the graphics below show how de interviewed are distributed among the profiles

lista = []

for i in range(0,len(dados)):

    lista.append(len(dados[i]))





ami = []

for j in range(0,len(p1)):

   ami.append(p1[j]) 







for j in range(0,(len(ami))):

    string = "{}: {}-{} {}-{} {}-{}".format(str(j+1),'Gender',ami[j][0],'Marital-status',ami[j][1],

              'income',ami[j][2])

    print(string)



prof = range(1,25)

rs = np.random.RandomState(7)



f, (ax1) = plt.subplots(1, 1, figsize=(8, 6), sharex=True)

sns.barplot(prof, lista, palette="Set3", ax=ax1)

ax1.set_ylabel("Qty")

#Let's see how the habit of exercising differs between profiles

for i in range(0,len(dados)):

    print(p1[i])

    unicos = np.unique(dados[i]['exercise'])

    dad = dados[i]

    #dad.loc[np.isnan(dad['exercise']) == True] = 'nan'

    tam = []

    for j in enumerate(unicos):

        tam.append(len(dad[dad['exercise'] == j[1]]))

    plt.pie(tam, labels=unicos, autopct = '%1.1f%%', shadow = True,startangle=60)

    plt.show()