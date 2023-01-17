#load the library and check its version

import numpy as np

np.__version__
#Using List

kaggle_list = [1,2,3,5,25,50]

np.array(kaggle_list)

# array([ 1,  2,  3,  5, 25, 50])



print("np.array(kaggle_list) - ",np.array(kaggle_list))



#using in-built functions like arnage, random, zeros, ones, linspace



#using arange(any_number) - which generates number from 0 to 10 range

np.arange(10) 

# [0 1 2 3 4 5 6 7 8 9]



## using arange(start,end,step)  NOTE: end value will not be included 

np.arange(10,50,5)

# [10 15 20 25 30 35 40 45]



#using random which generates randomly given n number

np.random.rand(5)

#[0.17274775 0.82368477 0.11113462 0.89180089 0.070426  ]



#To generate random integers where size = number of elements to return 

np.random.randint(5)

#[9 3 5 5 9 4]



#creates array with zero's

np.zeros(10)

# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]



#creates array with one's

np.ones(10)

# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]





#creates array with  even space between the given range of values

np.linspace(0, 10, 5)

# [ 0.   2.5  5.   7.5 10. ]



print("np.arange(10) - ",np.arange(10))

print("np.arange(10,50,5) - ",np.arange(10,50,5))

print("np.random.rand(5) - ",np.random.rand(5))

print("np.random.randint(10,size=6) - ",np.random.randint(10,size=6))

print("np.zeros(10) - ",np.zeros(10))

print("np.ones(10) - ",np.ones(10))

print("np.linspace(0, 10, 5) - ",np.linspace(0, 10, 5))


np.random.randint(10, size=(3,4)) 



#dtype - denotes datatype

np.zeros((3,3), dtype='int')

# array([[0, 0, 0],

#        [0, 0, 0],

#        [0, 0, 0]])



np.ones((3,3), dtype=float)

# array([[1., 1., 1.],

#        [1., 1., 1.],

#        [1., 1., 1.]])



#create an identity matrix

np.eye(3)

# array([[ 1.,  0.,  0.],

#       [ 0.,  1.,  0.],

#       [ 0.,  0.,  1.]])



#creating a matrix with a predefined value

np.full((3,5),1.23)

# array([[ 1.23,  1.23,  1.23,  1.23,  1.23],

#       [ 1.23,  1.23,  1.23,  1.23,  1.23],

#       [ 1.23,  1.23,  1.23,  1.23,  1.23]])







print("np.random.randint(10, size=(3,4)) - \n",np.random.randint(10, size=(3,4)))

print("np.zeros((3,3), dtype='int') - \n",np.zeros((3,3), dtype='int'))

print("np.ones((3,3), dtype=float) - \n",np.ones((3,3), dtype=float))

print("np.eye(3) - \n",np.eye(3))

print("np.full((3,5),1.23) - \n",np.full((3,5),1.23))

Kaggle = np.arange(1,100,5)





#To know the shape of the array

Kaggle.shape

# (20,) 



#To know the shape of the array

new_kaggle = Kaggle.reshape((4,5))





# To know the n dimentional array

new_kaggle.ndim

# (4,5)







# To know the total elements in array

new_kaggle.size

# 20



#To calculate min max and mean

new_kaggle.min(),new_kaggle.max(),new_kaggle.mean()

# 1 96 48.5



print("New kaggle ndim:", new_kaggle.ndim,"\nNew Kaggle shape:", new_kaggle.shape,"\nNew Kaggle size: ", new_kaggle.size,"\nNew kaggle min:", new_kaggle.min(),"\nNew Kaggle max:", new_kaggle.max(),"\nNew Kaggle mean: ", new_kaggle.mean())



import pandas as pd



#Reprseting with list

kaggle_list =[10,20,30]  

pd.Series(data=kaggle_list, index=['a','b','c'])

# a    10

# b    20

# c    30

# dtype: int64





#Represting with dictonary

kaggle_dict = {'one':1,'two':2}

pd.Series(data=kaggle_dict)

# one    1

# two    2

# dtype: int64



print(pd.Series(data=kaggle_list, index=['a','b','c']))

print(pd.Series(data=kaggle_dict))
kaggle_array = np.random.rand(5,4)

df = pd.DataFrame(kaggle_array,[1,2,3,4,5],['w','x','y','z'])



#info -complete information about the data set |  describe -computes summary of statistics integer / double variables

print("DataFrame is {} \nDataFrame info is \n{} \nDataFrame Describe is \n{}".format(df,df.info(),df.describe()))
#Let's sort the data frame by colomn 

#NOTE: inplace = True will make changes to the data

df.sort_values(by=['w'],ascending=True,inplace=False)
# Retrus number of rows and number of colomns

df.shape[0],df.shape[1]

# (5, 4)





# Adding a colomn to DataFrame

df['new_col'] = ['one',2,'3',4.4,5]





# Adding a row to DataFrame

# df.loc[index]=value

df.loc[df.shape[0]] = ['one','two',3,4.4,7]









#replace exsiting value with new values

df.replace(2, "second",inplace=True)

df

# df.drop('col_name',axis='colomns')

df.drop('y',axis='columns',inplace=True)

df
#Copy dataframe

df2 = df.copy()

df2
#conditions

df2[df2['z']>1]
#list all columns where A is greater than C

df2.query('w >x')