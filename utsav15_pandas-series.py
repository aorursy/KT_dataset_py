import numpy as np
import pandas as pd
my_list = [10,20,30]
labels = ['a','b','c']
pd.Series(data=my_list)
pd.Series(data=my_list,index=labels)
pd.Series(my_list,labels)
arr = np.array([10,20,30])
pd.Series(arr)
pd.Series(arr,labels)
d = {'a':10,'b':20,'c':30}
pd.Series(d)
pd.Series(data=labels)
# Even functions (although unlikely that you will use this)
pd.Series([sum,print,len])
ser1 = pd.Series([1,2,3,4],index = ['USA', 'Germany','USSR', 'Japan'])                                   
ser1
ser2 = pd.Series([1,2,5,4],index = ['USA', 'Germany','Italy', 'Japan'])                                   
ser2
ser1['USA']
ser1 + ser2