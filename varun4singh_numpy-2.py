import numpy as np
x=np.arange(1,10,dtype=int)
print(x)
import numpy as np
x=np.arange(1,10,dtype=int).reshape(3,3)
print(x)
a=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a)
a.shape #returns the dimensions of the array 
array1=np.array([[1,2,3],[4,5,6],[7,8,9]])
array2=np.arange(11,20).reshape(3,3)
array1
array2
np.add(array1,array2) #adding array1 and array2
np.multiply(array1,array2) #multiplying array1 and array2
np.divide(array1,array2) #dividing array1 and array2
np.remainder(array1,array2) #remainder of array1 and array2
np.subtract(array1,array2) #subtracting array1 and array2
a[:2,:] 
a[1:3,2:3]
np.transpose(a)
x=np.arange(2,5).reshape(3,1)
print(x)
np.append(a,x,axis=1)
x=np.arange(2,5).reshape(1,3)
np.append(a,x,axis=0)
x=np.insert(a,2,[0,0,0],axis=0)
print(x)
np.delete(x,2,axis=0)