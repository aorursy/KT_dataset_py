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
big_cities = pd.read_csv('../input/travelsanta/cities.csv')

big_cities.head()
small_cities = big_cities[:round((len(big_cities)*.1))]

small_cities.tail()
def isPrime(n):#time complexity: O(nlogn)

    prime = [True for i in range(n + 1)] 

    prime[0] = False # 0 and 1 is not prime

    prime[1] = False

    p = 2

    while (p * p <= n):  # keep the loop only log n

        # If prime[p] is not changed, then it is a prime

        if prime[p]:

            # Update all multiples of p

            for i in range(p * p, n + 1, p):

                prime[i] = False

        p += 1

    return prime

prime_cities = isPrime(max(small_cities.CityId))
small_cities["prime"] = prime_cities

small_cities.tail()
import ctypes

class DynamicArray:

  """A dynamic array class akin to a simplified Python list."""



  def __init__(self):

    """Create an empty array."""

    self._n = 0                                    # count actual elements

    self._capacity = 1                             # default array capacity

    self._A = self._make_array(self._capacity)     # low-level array



  def __len__(self):

    """Return number of elements stored in the array."""

    return self._n



  def __getitem__(self, k):

    """Return element at index k."""

    if not 0 <= k < self._n:

      raise IndexError('invalid index')

    return self._A[k]                              # retrieve from array



  def append(self, obj):

    """Add object to end of the array."""

    if self._n == self._capacity:                  # not enough room

      self._resize(2 * self._capacity)             # so double capacity

    self._A[self._n] = obj

    self._n += 1



  def _resize(self, c):                            # nonpublic utitity

    """Resize internal array to capacity c."""

    B = self._make_array(c)                        # new (bigger) array

    for k in range(self._n):                       # for each existing value

      B[k] = self._A[k]

    self._A = B                                    # use the bigger array

    self._capacity = c



  def _make_array(self, c):                        # nonpublic utitity

    """Return new array with capacity c."""

    return (c * ctypes.py_object)()               # see ctypes documentation



  def insert(self, k, value):

    """Insert value at index k, shifting subsequent values rightward."""

    # (for simplicity, we assume 0 <= k <= n in this verion)

    if self._n == self._capacity:                  # not enough room

      self._resize(2 * self._capacity)             # so double capacity

    for j in range(self._n, k, -1):                # shift rightmost first

      self._A[j] = self._A[j - 1]

    self._A[k] = value                             # store newest element

    self._n += 1



  def remove(self, value):

    """Remove first occurrence of value (or raise ValueError)."""

    # note: we do not consider shrinking the dynamic array in this version

    for k in range(self._n):

      if self._A[k] == value:              # found a match!

        for j in range(k, self._n - 1):    # shift others to fill gap

          self._A[j] = self._A[j + 1]

        self._A[self._n - 1] = None        # help garbage collection

        self._n -= 1                       # we have one less item

        return                             # exit immediately

    raise ValueError('value not found')    # only reached if no match

array_cities = DynamicArray()

array_cities._resize(len(small_cities)) # filled with none, changed later => help to increase speed

for i in range(len(small_cities)):

    array_cities.append(list(small_cities.values[i]))

array_cities.append(list(small_cities.values[0])) # create path by adding default start point to the end
array_cities[19773] #checking if runs correctly

for i in array_cities: #checking if runs correctly

    print(i)
def distance(c1, c2):

     return np.sqrt((c1[1]-c2[1])**2 + (c1[2]-c2[2])**2 ) # eucludian distance formula
def total_distance(data):#time complexity: O(n)

    step_num = 0 # init count of steps

    total = 0    # init total

    for i in range (len(data)-1): # loop all cities in path

        total += distance(data[i], data[i+1]) # calculate 2 cities distance, sum it with total distance

        if step_num % 10 == 0 and data[i+1][3] == "False": # prime path condition

            total += distance(data[i+1], data[i]) * 0.1

        step_num += 1 # keep increase step to be reasonable with the condition

    return total

total_distance(array_cities)
def insertion_sorting(data):#time complexity: O(n**2)

    #Sort list of comparable elements into nondecreasing order.

    for k in range(1, len(data)):  

        cur = data[k]   # current element to be inserted                   

        j = k           # find correct index j for current                 

        while j > 0 and data[j-1] > cur:    # element must be after current(if meet the condition)

            data[j] = data[j-1] #swap places

            j -= 1

        data[j] = cur    # cur is now in the right place

    return data
def insertion_sortingX(data):#time complexity: O(n**2)

    for k in range(1, len(data)):         

        cur = data[k]                      

        j = k                            

        while j > 0 and data[j-1][1] > cur[1]:  #change x axis conditions here  

            data[j] = data[j-1]

            j -= 1

        data[j] = cur

    return data
sorted_x = [None]*(len(small_cities)+1)# filled with none, changed later => help to increase speed

for i in range(1,len(small_cities)):

    sorted_x[i] = list(small_cities.values[i])

sorted_x[0] = list(small_cities.values[0])

sorted_x[-1] = list(small_cities.values[0])

total_distance(insertion_sortingX(sorted_x))

# create path by adding default start point to the end and start calculating total distance
def insertion_sortingY(data): #time complexity: O(n**2)

    for k in range(1, len(data)):         

        cur = data[k]                      

        j = k                            

        while j > 0 and data[j-1][2] > cur[2]:    

            data[j] = data[j-1]

            j -= 1

        data[j] = cur

    return data
sorted_y = [None]*(len(small_cities)+1)# filled with none, changed later => help to increase speed

for i in range(1,len(small_cities)):

    sorted_y[i] = list(small_cities.values[i])

sorted_y[0] = list(small_cities.values[0])

sorted_y[-1] = list(small_cities.values[0])

total_distance(insertion_sortingY(sorted_y))

# create path by adding default start point to the end and start calculating total distance