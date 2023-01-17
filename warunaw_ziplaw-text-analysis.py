# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
fsinhala= open('/kaggle/input/sinhala.txt', 'r')

ftamil= open('/kaggle/input/tamil.txt', 'r')



sin = fsinhala.read()

tam=ftamil.read()

def count(elements): 

    if elements[-1] == '.': 

        elements = elements[0:len(elements) - 1] 

   

    if elements in dictionary: 

        dictionary[elements] += 1

   

    else: 

        dictionary.update({elements: 1}) 

        

dictionary = {}

lst = sin.split() 

for elements in lst: 

    count(elements)
sinout = open('sinhala_freq.txt','w')



for k,v in dictionary.items():

    sinout.write(k + " " + str(v) + "\r\n") 

    

   
frq_freq= dict()



for i in dictionary.values():

    if i in frq_freq:

        frq_freq[i] += 1

    else:

        frq_freq[i] = 1
file2 = open("frqoffreqsin.txt", "w")

for k,v in frq_freq.items():

    file2.write(str(k) + " " + str(v) + "\r\n" )
print(frq_freq.values())
sorted_dic = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
lst = []

for k,v in sorted_dic.items():

    lst.append(v)
import matplotlib.pyplot as plt



plt.plot(lst)

plt.ylabel('frequency')

plt.xlabel('rank')

plt.yscale('log')

plt.xscale('log')

plt.show()
dictionary = {}

lst = tam.split() 

for elements in lst: 

    count(elements)
tamout = open('tamil_freq.txt','w')



for k,v in dictionary.items():

    tamout.write(k + " " + str(v) + "\r\n") 

    
frq_freq= dict()



for i in dictionary.values():

    if i in frq_freq:

        frq_freq[i] += 1

    else:

        frq_freq[i] = 1
file3 = open("FrequencyOfFreqTamil.txt", "w")

for k,v in frq_freq.items():

    file3.write(str(k) + " " + str(v) + "\r\n" )
sorted_dic_tamil = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
lst = []

for k,v in sorted_dic_tamil.items():

    lst.append(v)
plt.plot(lst)

plt.ylabel('frequency')

plt.xlabel('rank')

plt.yscale('log')

plt.xscale('log')

plt.show()