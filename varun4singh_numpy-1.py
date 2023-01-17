import numpy as np
x=np.array([1,4,6,3])
print(type(x))# return the type of array i.e numpy.ndarray, nd means n-dimensional
print(x)
x=np.array([1,2,5,'v','q',5,6])
print(type(x))
print(x)
#here stop value 9 is not included because endpoint=False 
x=np.linspace(start=3,stop=9,endpoint=False,retstep=False)
print(x)
#here stop value 9 is included because endpoint=True and increment value is also returned as: 0.12244897959183673
x=np.linspace(start=3,stop=9,endpoint=True,retstep=True)
print(x)
x=np.arange(5,25,5)
print(x)#x will never includes stop values
x=np.zeros((2,3))
print(x)
x=np.ones((5,3),int)
print(x)
x=np.random.rand(2,4)
print(x)
x=np.logspace(4,50,10,endpoint=True,base=4.0,dtype=int)
print(x)
import timeit #determines the time of program
x=range(1000)
%time sum(x)
x=range(1000)
%timeit sum(x)
import sys
x=np.arange(1,1000)
a=sys.getsizeof(1)#returns size of an element in bytes
b=sys.getsizeof(1)*len(x)# returns whole size of list/array
print('a:',str(a) + ' bytes')
print('b:',str(b) + ' bytes')
x=np.arange(1,1000)
x.itemsize
x.itemsize*x.size