import numpy as np # importing numpy
arr_1 = np.array([1, 2, 3, 4])  # crating array
arr_1
numbers = [1, 2, 3, 4]
arr_2 = np.array(numbers)
arr_2
array_of_zeros = np.zeros((3, 4)) # all zeros in array
array_of_zeros
array_of_ones = np.ones((3, 4))  # all ones in array
array_of_ones
array_of_ones_int = np.ones((3, 4), dtype=np.int16)    # array of type integer (row, columns)
array_of_ones_int
array_empty = np.empty((2, 3))  # Empty array
array_empty
array_eye = np.eye(3)  # only diagonal elements are one
array_eye
array_of_events = np.arange(1, 10, 1)  # like for loop it create array 
array_of_events
array_of_floats = np.arange(0, 5, 0.5)
array_of_floats
array_2d = np.array([(2, 4, 6), (3, 5, 7)]) # create two dimensional array
array_2d
array_2d.shape # find the shape of the array
np.arange(5)
array_nd = np.arange(6).reshape(3, 2)
array_nd
array_nd
array_ones = np.ones_like(array_nd) # create same dimension of array but all values will be 1
array_ones
a = np.arange(6)
print(a)
b = np.arange(12).reshape(4, 3)
print(b)
c = np.arange(24).reshape(2, 3, 4)
print(c)
print(np.arange(10000))
print(np.arange(1000).reshape(100, 10))
np.set_printoptions(threshold = np.nan)
# set_printoptions follow in all the print statement of the notebook
# threshold indicates no summaries
print(np.arange(100).reshape(10, 10))
a = np.array([10, 10, 10])
b = np.array([5, 5, 5])
a + b   # add element wise 
a - b # subtract elemetn wise
a * b # mul element wise
a / b # Division element wise
a % 3 # modulo operator
a < 35 # condition operator
a > 25
A = np.array([[1, 1], [0, 1]])
B = np.array([[2, 0], [3, 4]])

print('A:\n', A)
print('B:\n', B)
A * B
A.dot(B)  # matrix multiplication
# [row, col] = sum(all elements in row * all elements in col)
np.dot(A, B)
a *= 3 # element wise multiplication
a
b += a
b
ages = np.array([1, 2, 3, 4, 5])
ages.sum()
ages.min()
ages.max()
numbers = np.arange(12).reshape(3, 4)
numbers
numbers.sum(axis = 0)  # sums up the columns
numbers.sum(axis = 1) # sums up the rows
numbers.min(axis = 1) # find minimum in the row
angles = np.array([0, 30, 45, 60, 90])
angles_radians = angles * (np.pi/180)
angles_radians
print("Sin of angles in the array: ")
print(np.sin(angles_radians))
angles_radians = np.radians(angles)
angles_radians
print("Cosine of angles in the array: ")
print(np.cos(angles_radians))
print("Tangent of angles in the array: ")
print(np.tan(angles_radians))
sin = np.sin(angles * np.pi/180)
print('Compute sine inverse of angles. Returned Values are in radians.')

inv = np.arcsin(sin)
print(inv)
print('Check result by converting to degrees: ')
print(np.degrees(inv))
scores = np.array([1, 2 , 3, 4, 5])
print(np.mean(scores))  # mean
print(np.median(scores)) # median
salaries = np.genfromtxt('../input/salary.csv',
                        delimiter = ',')  # read the file 
salaries
salaries.shape
mean = np.mean(salaries)
median = np.median(salaries)
stddev = np.std(salaries)
variance = np.var(salaries)
print("Mean = %i" %mean)
print("Median = %i" %median)
print("Standard Deviation = %i" %stddev)
print("Variance = %i" %variance)
a = np.arange(11) ** 2   # square all the elements
a
a[2]
a[-2]
a[2:7]
a[2:-2] # means 2 element to third last element
a[2:] # all element from given index
a[:7]
a[:11:2] # begining: end: step size
a[::-1] # reverse the array means omiting the start and end than starting backwards
students = np.array([['Krunal', 'Mihir', 'Kunal', 'Sai', 'Ankit'],
                   [1, 2, 3, 4, 5]])
students
students[0]
students[1]
students[0, 1]
students[0:2, 1:4]  # [rows, columns]
students[:, 1:2]  # all rows and given column
students[:, 1:3] 
students[-1, :]
students[0, ...]
students[..., 1]
a = np.arange(11) ** 2
a
for i in a:
    print(i**(1/2))
for i in students:
    print('i = ', i)
# Row major flattening - elements in a row appear together
for element in students.flatten():
    print(element)
for element in students.flatten(order='F'):
    print(element)
x = np.arange(12).reshape(3, 4)
x
for i in np.nditer(x): # iterate in row major form, one 
    print(i)
for i in np.nditer(x, order = 'F'): # iterate in col major form, one 
    print(i)
for i in np.nditer(x, order = 'F', flags = ['external_loop']):
    print(i)
# for arr in np.nditer(x):
#     arr[...] = arr * arr
# ---------------------------------------------------------------------------
# ValueError                                Traceback (most recent call last)
# <ipython-input-110-2dde0a500dd6> in <module>()
#       1 for arr in np.nditer(x):
# ----> 2     arr[...] = arr * arr

# ValueError: assignment destination is read-only
for arr in np.nditer(x, op_flags = ['readwrite']):
    arr[...] = arr * arr
x
a = np.array([('a', 'b', 'c'),
            ('d', 'e', 'f')])
a
a.shape
a.ravel() # 2D to 1D
a.T # Transpose
a.T.ravel()  # flatten the array in row wise
a.reshape(3, 2)
np.arange(15).reshape(3, 5)
np.arange(15).reshape(5, 3)
name = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
name
name.reshape(-1, 3) # arrange rows automatically according to the columns
name.reshape(3, -1)
x = np.arange(9)
x
np.split(x, 3) # split in equal parts
print('Split the array at positions indicated in 1-D array')
np.split(x, [4, 7])
y = np.array([('a', 'b', 'c', 'x'),
            ('d', 'e', 'f', 'y')])
y
a, b = np.hsplit(y, 2) # split horizontally
a
b
av, bv = np.vsplit(y, 2) # Split vertically
av
bv
from scipy import ndimage
from scipy import misc
f = misc.face()  # Get a color image of a raccon face
f.shape
type(f)
import matplotlib.pyplot as plt
plt.imshow(f)
a = f[384:, 512:, :]
plt.imshow(a)
plt.show()
a, b = np.split(f, 2)  # spliting in two parts by default in row
plt.imshow(a)
plt.show()
plt.imshow(b)
plt.show()
x, y = np.split(f, 2, axis = 1) # Spliting in two parts in column
plt.imshow(x)
plt.show()
plt.imshow(y)
plt.show()
plt.imshow(np.concatenate((a, b))) # concatenating in row
plt.show()
plt.imshow(np.concatenate((x, y), axis = 1)) # concatenating in column
plt.show()
fruits = np.array(['Apple', 'Mango', 'Orange'])
a = fruits.view()
b = fruits.view()
print(a)
print(b)
print('ids for the arrays are different.')
print('id for fruits is : ')
print(id(fruits))
print('id for baskets is :')
print(id(a))
print(id(b))
a is fruits
a.base is fruits  # Describes that 'a' is derived from the fruits
a[0] = 'Strawberry'
a
fruits
fruits = np.array(['Apple', 'Orange', 'Mango'])
a = fruits.copy()
a
a is fruits
a.base is fruits # Describes that 'a' is not derived from the fruits
a[0] = 'Strawberry'
a
fruits