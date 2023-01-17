import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

stud_marks = pd.read_csv("../input/stud_marks.csv")

print(stud_marks)
nRowsRead = 1000

df1 = pd.read_csv('/kaggle/input/stud_marks.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'stud_marks.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
x = stud_marks['internal_marks']

y = stud_marks['external_marks']



import numpy as np

import array



covariance = np.cov(x,y)[0][1]

variancex = np.var(x)

variancey = np.var(y)

internalmean = np.mean(x)

externalmean = np.mean(y)



output = (covariance/variancex)

output2 = (covariance/variancey)



## Output = B

print ("Value w.r.t - x : ", (output))

print ("Value w.r.t - y : ", (output2))

## Statement End
output3 = externalmean - output*internalmean

print ("Value w.r.t - x & y : ",(output3))



## Linear Equation ##



print ("Regression of data is : ",("\n"),"Y = ",output3)

print (" +")

print (" X = ",output)