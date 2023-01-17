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
import pandas as pd

import numpy as np

import csv

# 这里只用测试序列进行计算

testfile = '../input/test.csv'

resultfile = '../input/result.csv'
SequenceDict = {} #序列字典，<id,Sequence>



with open(testfile,'r') as f:

    reader = csv.reader(f)   

    i = 0

    for row in reader:

        if i == 0: i = 1;continue 

#         if i < 5:

#             print(row[0],np.array(row[1].split(',')).astype(int))

#         i += 1

        sequence = [int(x) for x in row[1].split(',')]

        SequenceDict[row[0]] = np.array(sequence)

def autoRegression(sequence, p, k):

    '''

    sequence ： 序列

    p ：假设的参数个数

    coeff： 得到的参数

    

    检查p是否可以接受

    这里通过比较p之后的k个值，得到的预测值，是否与真实值的误差都在10^-2之内

    如果都满足就接受，否则就不接受,

    如果可以接受，就预测序列之后的值

    

    '''

    if len(sequence) < 2*p: return None

    

    x = []

    b = []

    for i in range(p):

        tmp = sequence[i:p+i]

        x.append(tmp)

    b = sequence[p:2*p]

    

    x = np.array(x)

    b = np.array(b)

    

    try: 

        if np.linalg.det(x)==0:

            return None

    except TypeError:

        return None

   

    coeff = np.linalg.inv(x).dot(b)

#     print(coeff)

    k = min(k, len(sequence) - 2*p)

    

    for i in range(k):

        t = 2*p + i

        predict = np.sum(coeff * sequence[t-p:t])

        if np.abs(predict - sequence[t]) > 10**-2:

            return None

    

    

    nextvalue = np.sum(sequence[len(sequence)-p:] * coeff)

    

    return int(round(nextvalue))

    
#预测

predicted = dict()

k = 7

j = 0

for p in range(2,10): # 这里应该是100，为了提高速度，这里设为10

    i = 0

    for key in SequenceDict.keys():

        if key in predicted.keys(): continue

    

        sequence = np.array(SequenceDict[key])

#         print(sequence)

        nextvalue = autoRegression(sequence,p,k)

        

        if nextvalue != None:

            i += 1

            predicted[key] = nextvalue

    j += i

    print("p = ",p ,"i = ",i)

    

    if j == len(SequenceDict):

        break

    
print(len(SequenceDict))

len(predicted)



'''

填充没有预测到的序列，用这个序列的前一个替代它，当然这个可能不会对结果有什么好处，

但是，提交文件要求对所有的值，进行预测。

'''



result = []

for key in predicted.keys():

    result.append([int(key),predicted[key]])

for key in SequenceDict.keys():   

    if key not in predicted.keys():

        result.append([int(key),SequenceDict[key][-1]])



print(len(result))



sorted(result)
with open(resultfile,'w') as f:

    writer = csv.writer(f)

    writer.writerow(['id','Last'])

    

    for data in result:

        writer.writerow([data[0],data[1]])

    
