import numpy as np
import numpy as np

ary = np.array([1,2,3,5,8])

ary = ary + 1

print (ary[1])
import numpy as np

a = np.array([1,2,3,5,8])
b = np.array([0,3,4,2,1])
c = a + b
c = c*a

print (c[2])
import numpy as np

a = np.array([1,2,3,5,8])
print (a.ndim)
import numpy as np

a = np.array([[1,2,3],[0,1,4]])
print (a.size)
import numpy as np

a = np.array([[1,2,3],[0,1,4]])
b = np.zeros((2,3), dtype=np.int16)
c = np.ones((2,3), dtype=np.int16)
d = a + b + c
print (d[1,2] )
import numpy as np

a = np.array([1,2,3,4,5])
b = np.arange(0,10,2)
c = a + b
print (c[4])
import numpy as np

a = np.zeros(6)
b = np.arange(0,10,2)
c = a + b
print (c[4])
print(a.shape, b.shape)
import numpy as np

a = np.array([[0, 1, 0], [1, 0, 1]])
a += 3
b = a + 3
print (a[1,2] + b[1,2])
import numpy as np

a = np.array([[0, 1, 2], [3, 4, 5]])
b = a.sum(axis=1)
print (b)
import numpy as np

a = np.array([[0, 1, 2], [3, 4, 5]])
b = a.ravel()
print (b[0,0])
print(b)
print(b.shape)