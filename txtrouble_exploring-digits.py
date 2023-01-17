# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib import cm

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

#print(train.head())



for i in range(10):

    t=train[train.label==i]

    

    img=t.values[:,1:].mean(axis=0)

    imgs=t.values[:,1:].std(axis=0)

    

    img=img.reshape(28,28)

    imgs=imgs.reshape(28,28)

    

    plt.figure(1) 

    plt.subplot(2,5,i+1)

    plt.title('$\mu_' + str(i)+'$')

    plt.imshow(img,cmap=cm.binary)

    

    plt.figure(2) 

    plt.subplot(2,5,i+1)

    plt.title('$\mu_' + str(i)+'+\sigma_'+ str(i)+'$')

    plt.imshow(imgs,cmap=cm.binary)

plt.show()    
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100.0,random_state=1)

lr.fit(tr_dm, train.values[0::,0])

output_lr = lr.predict(x_test_dm).astype(int)

np.savetxt('final_submission_lr.csv', np.c_[range(1,len(test)+1),output_lr], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
tr=train.values[0::,1::]

tr_dm=tr/255   



test = pd.read_csv('../input/test.csv').values

test_dm=test/255  



forest = RandomForestClassifier(n_estimators=200)

forest = forest.fit(tr_dm, train.values[0::,0] )



output_rf = forest.predict(test_dm).astype(int)

np.savetxt('final_submission_rf.csv', np.c_[range(1,len(test)+1),output_rf], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100.0,random_state=1)

lr.fit(tr_dm, train.values[0::,0])

output_lr = lr.predict(x_test_dm).astype(int)

np.savetxt('final_submission_lr.csv', np.c_[range(1,len(test)+1),output_lr], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')