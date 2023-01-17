# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.read_csv('../input/train.csv')



bins = []

#Get claims experience of Binary Variables

for i in filter(lambda x: x[-3:] == 'bin' ,list(data)):

    temp1 = data[['target',i]]

    for j in temp1[i].unique():

        claims = float(temp1[temp1[i]==j]['target'].sum()) / temp1['target'].count()

        bins.append([i,j,claims])



#Get claims experience by Categorical Variables

for i in filter(lambda x: x[-3:] == 'cat' ,list(data)):

    temp1 = data[['target',i]]

    for j in temp1[i].unique():

        claims = float(temp1[temp1[i]==j]['target'].sum()) / temp1['target'].count()

        bins.append([i,j,claims])



WoE = pd.DataFrame(bins,columns=['variable','bin','rate'])

plt.clf()

plot_rows = len(WoE['variable'].unique())

for i in WoE['variable'].unique():

    plt.figure(figsize=(8,4))

    x_val = WoE[WoE['variable'] == i]['bin']

    y_val = WoE[WoE['variable'] == i]['rate']

    subplt = list(WoE['variable'].unique()).index(i) + 1

    #plt.subplot(plot_rows, 1, subplt)

    plt.title(i)

    plt.xticks(x_val.values)

    plt.ylim(0,max(y_val)+0.005)

    plt.bar(x_val.values, y_val, color='blue')

    plt.xticks(x_val.values)

plt.show()

# Any results you write to the current directory are saved as output.