import numpy as np
a = np.zeros(3)

a
type(a)
a = np.zeros(3)

type(a[0])
a = np.zeros(3, dtype=int)

type(a[0])
z = np.zeros(10)
z.shape
z.shape = (10, 1)

z
z = np.zeros(4)

z.shape = (2, 2)

z
z = np.empty(3)

z
z = np.linspace(2, 4, 5)  # From 2 to 4, with 5 elements
z = np.identity(2)

z
z = np.array([10, 20])                 # ndarray from Python list

z
type(z)
z = np.array((10, 20), dtype=float)    # Here 'float' is equivalent to 'np.float64'

z
z = np.array([[1, 2], [3, 4]])         # 2D array from a list of lists

z
na = np.linspace(10, 20, 2)

na is np.asarray(na)   # Does not copy NumPy arrays
na is np.array(na)  
z = np.linspace(1, 2, 5)

z
z[0]
z[0:2]  # Two elements, starting at element 0
z[-1]
z = np.array([[1, 2], [3, 4]])

z
z[0, 0]
z[0, 1]
z[0, :]
z[:, 1]
z = np.linspace(2, 4, 5)

z
indices = np.array((0, 2, 3))

z[indices]
z
d = np.array([0, 1, 1, 0, 0], dtype=bool)

d
z[d]
z = np.empty(3)

z
z[:] = 42

z
z = np.array([1, 2, 3])

np.sin(z)
n = len(z)

y = np.empty(n)

for i in range(n):

    y[i] = np.sin(z[i])
z
(1 / np.sqrt(2 * np.pi)) * np.exp(- 0.5 * z**2)
def f(x):

    return 1 if x > 0 else 0
x = np.random.randn(4)

x
np.where(x > 0, 1, 0)  # Insert 1 if x > 0 true, otherwise 0
f = np.vectorize(f)

f(x)                # Passing the same vector x as in the previous example
z = np.array([2, 3])

y = np.array([2, 3])

z == y
y[0] = 5

z == y

z != y