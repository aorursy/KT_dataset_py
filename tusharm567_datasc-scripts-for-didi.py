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
file1 = pd.read_csv('../input/file1.txt', '\t', header=None, names = ['GCF', 'WP'])
file1.head()
file1['ind'] = file1.index + 1
file1['ind'] = 'C' + file1['ind'].astype(str) + ' '
file1.head()
file2 = pd.read_csv('../input/file2.txt', '\t', header=None, names = [
    'ind', 'col1', 'col2', 'GCF', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'WP', 'col12', 'col13', 'col14', 'col15', 'col16', 'col17', 'col18', 'col19', 'col20'])
file2.head()
file1.GCF = file1.GCF.astype(str).apply(lambda x: x.strip())
file1.WP = file1.WP.astype(str).apply(lambda x: x.strip())

file2.GCF = file2.GCF.astype(str).apply(lambda x: x.strip())
file2.WP = file2.WP.astype(str).apply(lambda x: x.strip())
pd.merge(file1, file2, on = ['ind', 'GCF', 'WP'], how = 'inner')
pd.merge(file1, file2, on = ['GCF', 'WP'], how = 'inner')
output = pd.merge(file1, file2, on = ['GCF', 'WP'], how = 'inner')
output.to_csv('output_1.csv')
pd.merge(file1, file2, on = ['ind'], how = 'inner').head()
file3 = pd.read_csv('../input/file3.txt', '\t')
file3.fillna(-9999, inplace=True)
file3
def resolve_as_arr(val):
    return str(val).split(',')
    
file3['rmlA'] = file3['rmlA'].apply(resolve_as_arr)
file3['rmlB'] = file3['rmlB'].apply(resolve_as_arr)
file3['rmlC35'] = file3['rmlC35'].apply(resolve_as_arr)
file3['rmlC3'] = file3['rmlC3'].apply(resolve_as_arr)
file3['rmlD'] = file3['rmlD'].apply(resolve_as_arr)

file3
import itertools
a = [1, 2, 3]
b = [4, 5, 6]
list(itertools.product(a,b))
arr = [2,1,3,5,7]
arr.sort()
print(arr)

diff = []
for i in range(1, len(arr)):
    diff.append(arr[i] - arr[i-1])
max(diff)
def findRange(arr_of_nums):
    arr_of_nums = list(arr_of_nums)
    arr_of_nums = [a for a in arr_of_nums if a != -9999]
    arr_of_nums.sort()
    
    diff = []
    for i in range(1, len(arr_of_nums)):
        diff.append(arr_of_nums[i] - arr_of_nums[i-1])
    return max(diff) 

def findMinRangeVal(arr_of_arr_of_nums):
    return min(arr_of_arr_of_nums, key=findRange)

def findMinRange(arr_of_arr_of_nums):
    print(findMinRangeVal(arr_of_arr_of_nums))
    return findRange(min(arr_of_arr_of_nums, key=findRange))
        
a = (10, 20, 30)
b = (15, 4, -9999)
d = (2, 3, 11)
c = list(itertools.product(a,b,d))


print("Range =>")
print(a)
print(findRange(a))
print("-------")
print(b)
print(findRange(b))
print("-------")
print(d)
print(findRange(d))
print("-------")

print("Combinations =>")
print(c)
print("Min Combination Range =>" )
print( findMinRange(c))

file3
def findMinRangeInRow(row):
    col_names = ['rmlA','rmlB','rmlC35','rmlC3','rmlD']
    for col_name in col_names:
        row[col_name] = map(int, row[col_name])
    combinations = list(itertools.product(row['rmlA'], row['rmlB'], row['rmlC35'], row['rmlC3'], row['rmlD']))
    return findMinRange(combinations)

def isNeighbour(row):
    return row['minRange']<=10

file3['minRange'] = file3.apply(findMinRangeInRow, axis=1)
file3['output_2'] = file3.apply(isNeighbour, axis=1)
file3
file3.to_csv('output_2.csv')