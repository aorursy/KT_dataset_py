# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def random_mini_batches(X,Y, minibatch_size):

    

    m = Y.shape[1]            # number of examples

    

    # Lets shuffle X and Y

    permutation = list(np.random.permutation(m))            # shuffled index of examples

    shuffled_X = X[:, permutation]

    shuffled_Y = Y[:, permutation]

    

    minibatches = []                                        # we will append all minibatch_Xs and minibatch_Ys to this minibatch list 

    number_of_minibatches = int(m/minibatch_size)           # number of mini batches 

    

    for k in range(number_of_minibatches):

        minibatch_X = shuffled_X[:,k*minibatch_size: (k+1)*minibatch_size ]

        minibatch_Y = shuffled_Y[:,k*minibatch_size: (k+1)*minibatch_size ]

        minibatch_pair = (minibatch_X , minibatch_Y)                        #tuple of minibatch_X and miinibatch_Y

        minibatches.append(minibatch_pair)

    if m%minibatch_size != 0 :

        last_minibatch_X = shuffled_X[:,(k+1)*minibatch_size: m ]

        last_minibatch_Y = shuffled_Y[:,(k+1)*minibatch_size: m ]

        last_minibatch_pair = (last_minibatch_X , last_minibatch_Y)

        minibatches.append(last_minibatch_pair)

    return minibatches