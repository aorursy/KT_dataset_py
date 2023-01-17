# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

data.head()
nums = data[data['Country/Region']=='US'].sum().drop(['Province/State','Country/Region','Lat','Long'])

#nums = data[data['Country/Region']=='US'].sum().drop(['Lat','Long'])
y = data[data['Country/Region']=='US'].sum().drop(['Province/State','Country/Region','Lat','Long']).tolist()

x = range(len(y))
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(15,5))

plt.plot(nums)

plt.xticks(rotation=90)

plt.ylabel('Cases')

plt.title("US Coronavirus Cases")

plt.show()
#Obtain estimate

def get_error(a,b):

    error = 0

    for index in x:

        error += abs((a**(x[index]-b)) - (y[index]))

    return error / len(x)



def element_same_check(history):

    for element in history:

        if round(element) != history[0]:

            return False

    return True



def increasing_check(history):

    if history[-1] > history[0]:

        return True

    else:

        return False



import ast
'''

    if (iteration+1) % 1_000  == 0: # A check every 100 iterations

        if element_same_check(global_mem[-1000:]) == True:

            lr1 *= 1.2

            lr2 *= 1.2

            print('^',end='')

        elif element_same_check(global_mem) == True:

            lr1 *= 0.83333333333 #1 / 1.2.

            lr2 *= 0.83333333333 #0.8333 * 1.2 is approximately 1

            print('.',end='')

        lr1 = round(lr1,4)

        lr2 = round(lr2,4)

        '''
a = 1

lr1 = 0.01

b = 3

lr2 = 0.01



iterations = 10_000



global_mem = []



for iteration in range(iterations):



    b1 = b + lr2

    b2 = b - lr2

    if get_error(a,b1) < get_error(a,b2):

        b = b1

    else:

        b = b2



    a1 = a + lr1

    a2 = a - lr1

    if get_error(a1,b) < get_error(a2,b):

        a = a1

    else:

        a = a2



    global_mem.append(get_error(a,b))

    

    if iteration % 1_000 == 0:

        print('ITERATION {}'.format(iteration))

        print("{'A':"+str(a)+", 'B':"+str(b)+"}")

        print('Error:',global_mem[-1],'\n')
import matplotlib.pyplot as plt

plt.figure(figsize=(20,6))

plt.plot(global_mem)
best_set = {'A':1.1900000000000002, 'B':2.250000000000016}

def function(x):

    return best_set['A']**(x-best_set['B'])
plt.figure(figsize=(20,8))

plt.plot(x,y,label='Real Data')

plt.plot(x,[function(i) for i in x],label='Exponential Model')

plt.legend()