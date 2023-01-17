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
train_data = pd.read_csv('../input/train.csv')
#train_data[:10]
data_without_label = train_data.ix[:,1:]

train_label = train_data['label']
int1 = np.array(data_without_label.ix[0]).reshape(28,28)
int1 = pd.DataFrame(int1)
import matplotlib.pyplot as plt
fig = plt.figure(111)
#plt.gray()
plt.imshow(int1)
num=[]

for i in range(10):

    num.append(np.array(data_without_label.ix[i]).reshape(28,28))
#num[:5]
for i in range(1,5):

    plt.subplot(2,2,i)

    plt.imshow(num[i-1],cmap= 'gray')
data_sort = train_data.sort_values(by= 'label')