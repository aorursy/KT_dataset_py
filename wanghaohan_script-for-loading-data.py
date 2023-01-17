# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



def EEGLoading():

    file = np.loadtxt('EEGdata.csv', delimiter=',')

    data = np.asarray(file)

    X = data[:, 2:13]

    Y = data[:, 13:15]

    G0 = data[:, 0].astype(int)

    G1 = data[:, 1].astype(int)

    Z0 = np.zeros([np.shape(X)[0], np.amax(G0) + 1])

    for i in range(1, np.shape(G0)[0]):

        Z0[i - 1][G0[i]] = 1

    Z1 = np.zeros([np.shape(X)[0], np.amax(G1) + 1])

    for i in range(1, np.shape(G1)[0]):

        Z1[i - 1][G1[i]] = 1

    return X, Y, Z0, Z1

    

X, Y, Z0, Z1 = EEGLoading()



# Any results you write to the current directory are saved as output.