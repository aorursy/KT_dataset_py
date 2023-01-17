# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install wildqat   

!pip install pyitlib
import numpy as np

import pandas as pd

from pyitlib import discrete_random_variable as drv

from sklearn.preprocessing import KBinsDiscretizer

from sklearn.preprocessing import LabelEncoder

import wildqat as wq
class MatrixCreation:

    def Mutual_Information(self,X):

        return drv.information_mutual(X.T)

    

    def Identity_Matrix(self,NOF):

        return np.identity(NOF)

    

    def Diagonal_Matrix(self,data,NOF):

        D = np.zeros([NOF,NOF])

        for i in range(NOF):

            corr = np.corrcoef(data.iloc[:,i], data.iloc[:,NOF])

            D[i,i]=corr[0,1]

        return D

    

    def mRMR(self,D,M,NOF):       #-------- minimun Redundancy Maximum Relevance

        return ((M/NOF)-D)
if __name__=="__main__":

    data = pd.read_csv('../input/cmc-dataset/CMC.csv')

    col_no = len(data.columns)

    NOF = col_no - 1

    M = data.iloc[:,0:NOF]    ##----------- Data Matrix

    M.values

    #print(M)

    

    enc = KBinsDiscretizer(n_bins=5,encode='ordinal')   

    M_binned = enc.fit_transform(M)

    #print(M_binned)

    

    

    mc = MatrixCreation()

    D = mc.Diagonal_Matrix(data,NOF)

    #print(D)



    MI = mc.Mutual_Information(M_binned.astype(int))

    mRMR = mc.mRMR(D,MI,NOF)

    #mRMR_qubo = pd.DataFrame(mRMR)

    #mRMR_qubo.to_csv("CMC_mi_mRMR_qubo.csv")
    Result = []

    for j in range(10):

        a = wq.opt()

        a.qubo = mRMR

        answer = a.sa()

        print('Solution of {}th iteration: '.format(j),answer)

        Result.append(answer)
df = pd.DataFrame(Result,columns = ["Wife's age","Wife's education","Husband's education",

                                     "Number of children ever born","Wife's religion","Wife's now working",

                                    "Husband's occupation","Standard-of-living index","Media exposure"])

#df.to_csv('CMC_mi_mRMR_sa_result.csv')

df