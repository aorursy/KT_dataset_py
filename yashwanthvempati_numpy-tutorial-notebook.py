#Basic array creation

import numpy as np

a = np.array([2,3,4])

print(a)

print(a.dtype)



b = np.array([1.2, 3.5, 5.1])

print(b.dtype)
#Basic array creation

import numpy as np

b = np.array([(1.5,2,3), (4,5,6)])

print(b)
#Creating a complex array

c = np.array( [ [1,2], [3,4] ], dtype=complex )

print (c)
#Zeros, Ones and Empty functions

print (np.zeros( (3,4) ))

print (np.ones( (2,3,4), dtype=np.int16 ))

print (np.empty( (2,3) ))
#Sequence of Numbbers

print(np.arange( 10, 30, 5 ))

print(np.arange( 0, 2, 0.3 ))
#Printing Arrays

a = np.arange(6)

print(a)



b = np.arange(12).reshape(4,3)

print(b)



c = np.arange(24).reshape(2,3,4)

print(c)
#Basic Operations

a = np.array( [20,30,40,50] )

b = np.arange( 4 )

print (b)

c = a-b

print (c)

print(b**2)

print(10*np.sin(a))

print(a<35)
#Matrix multiplication

A = np.array( [[1,1],

            [0,1]] )

B = np.array( [[2,0],

               [3,4]] )

print(A*B)



print(A @ B )



print(A.dot(B))
#Finding the array type



a = np.ones(3, dtype=np.int32)

b = np.linspace(0,3.14,3)

print(b.dtype.name)



c = a+b

print(c)

print(c.dtype.name)

d = np.exp(c*1j)

print (d)

print(d.dtype.name)
#Unary operations

a = np.random.random((2,3))

print(a)

print(a.sum())

print(a.min())

print(a.max())
#Universal functions

B = np.arange(3)

print(B)

print(np.exp(B))

print(np.sqrt(B))

C = np.array([2., -1., 4.])

print(C)
#Indexing, Slicing and iterating

a = np.arange(10)**3

print(a)

print(a[2])

print(a[2:5])

a[:6:2] = -1000

print(a)

print(a[ : :-1])

for i in a:

    print(i**(1/3.))
#Shape Manipulation

a = np.floor(10*np.random.random((3,4)))

print(a)

print(a.shape)

print(a.ravel())

print(a.reshape(6,2))

print(a.T)

print(a.T.shape)

print(a.shape)
#Stacking together different arrays 

a = np.floor(10*np.random.random((2,2)))

print(a)



b = np.floor(10*np.random.random((2,2)))

print(b)



print(np.vstack((a,b)))



print(np.hstack((a,b)))

#Column stack



from numpy import newaxis

a = np.floor(10*np.random.random((2,2)))

b = np.floor(10*np.random.random((2,2)))

print(np.column_stack((a,b)))



a = np.array([4.,2.])

b = np.array([3.,8.])

print(np.column_stack((a,b)))

print(np.hstack((a,b)))

print(a[:,newaxis])

print(np.column_stack((a[:,newaxis],b[:,newaxis])))

print(np.hstack((a[:,newaxis],b[:,newaxis])))

#Splitting one array into several smaller ones

a = np.floor(10*np.random.random((2,12)))

print(a)

print(np.hsplit(a,3))

print(np.hsplit(a,(3,4)))
#Copies and Views

a = np.arange(12)

b=a

print(b is a)

b.shape = 3,4 

print(a.shape)



c = a.view()

print(c is a)

print(c.base is a)

print(c.flags.owndata)

c.shape = 2,6

print(a.shape)

c[0,4] = 1234

print(a)



d = a.copy() 

print(d is a)

print(d.base is a)

d[0,0] = 9999

print(a)