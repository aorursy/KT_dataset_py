# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        #print(os.path.join(dirname, filename))

        pass

        

print("Finished Importing Libraries")



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
v1 = np.random.rand(1000000, 1)

v2 = np.random.rand(1000000, 1)
# Scaling Vector - For loop

start = time.process_time()

v1_scaled = np.zeros((1000000, 1))



for i in range(len(v1)):

    v1_scaled[i] = 2 * v1[i]



end = time.process_time()

    

print("Scaling vector Answer = " + str(v1_scaled))

print("Time taken = " + str(1000*(end - start)) + " ms")  
#Scaling Vector - Vectorized

start = time.process_time()

v1_scaled = np.zeros((1000000, 1))



v1_scaled = 2 * v1



end = time.process_time()

    

print("Scaling vector Answer = " + str(v1_scaled))

print("Time taken = " + str(1000*(end - start)) + " ms")  
# Dot product For loop

start = time.process_time()

product = 0



for i in range(len(v1)):

    product += v1[i] * v2[i]



end = time.process_time()



print("Dot product Answer = " + str(product))

print("Time taken = " + str(1000*(end - start)) + " ms")
#Dot product Vectorized

start = time.process_time()

product = 0



product = np.dot(v1.T, v2)



end = time.process_time()



print("Dot product Answer = " + str(product))

print("Time taken = " + str(1000*(end - start)) + " ms")
#Element wise mutliplication For loop

start = time.process_time()



answer = np.zeros((1000000, 1))



for i in range(len(v1)):

    answer[i] = v1[i] * v2[i]

    

end = time.process_time()



print("Element Wise answer = " + str(answer))

print("Time Taken = " + str(1000*(end - start)) + " ms")
#Element wise multiplication Vectorized

start = time.process_time()



answer = np.zeros((1000000, 1))



answer = v1 * v2



end = time.process_time()



print("Element Wise answer = " + str(answer))

print("Time Taken = " + str(1000*(end - start)) + " ms")
#Element wise matrix multiplication For loop



m1 = np.random.rand(1000, 1000)

m2 = np.random.rand(1000, 1000)

answer = np.zeros((1000, 1000))



start = time.process_time()



for i in range(m1.shape[0]):

    for j in range(m1.shape[1]):

        answer[i, j] = m1[i, j] * m2[i, j]

    

end = time.process_time()



print("Element Wise Matrix answer = " + str(answer))

print("Time Taken = " + str(1000*(end - start)) + " ms")
#Element wise matrix multiplication Vectorized

answer = np.zeros((1000, 1000))



start = time.process_time()



answer = np.multiply(m1, m2)



end = time.process_time()



print("Element Wise Matrix answer = " + str(answer))

print("Time Taken = " + str(1000*(end - start)) + " ms")
sizes = [10, 100, 1000, 10000, 100000, 1000000, 10000000]

complexity = pd.DataFrame(columns=['sizes', 'for_loop', 'numpy'])

complexity['sizes'] = sizes
for_loops = []

numpy = []



for size in sizes:

    v1 = np.random.rand(size, 1)

    v2 = np.random.rand(size, 1)

    

    #For loop implementation

    start = time.process_time()

    product = 0



    for i in range(len(v1)):

        product += v1[i] * v2[i]



    end = time.process_time()

    

    for_loops.append(1000*(end-start))

    

    #Vectorized implementation

    

    start = time.process_time()

    product = 0



    product = np.dot(v1.T, v2)



    end = time.process_time()

    numpy.append(1000*(end - start))

    
complexity['for_loops'] = for_loops

complexity['numpy'] = numpy
plt.plot(complexity['sizes'], complexity['for_loops'])

plt.plot(complexity['sizes'], complexity['numpy'])



plt.xscale(value='log')

plt.xlabel("Size of input")

plt.ylabel("Time taken in ms")

plt.legend(['for loop', 'numpy'])

plt.show()