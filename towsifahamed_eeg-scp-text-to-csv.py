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
head1=list()

for i in range (0,5377):

    head1.append(i)

#print(head)
train1=pd.DataFrame(columns=head1)

file1 = open("/kaggle/input/eeg-input/Traindata_2.txt","r")

file1.seek(0)  

  

i=1

string=file1.readline()

while string:

  datas=list()

  data=''

  for element in range(0, len(string)): 

    if string[element]==' ' and string[element+1]!=' ' and element+1<len(string):

      while string[element+1]!=' ' and element+2<len(string):

        data+=string[element+1]

        element+=1

      datas.append(data)

      data=''

  train1=train1.append(pd.Series(datas, index=head1), ignore_index=True)

  print(i)

  i+=1

  string=file1.readline()
train1
head2=list()

for i in range (1,5377):

    head2.append(i)

#print(head)
test1=pd.DataFrame(columns=head2)

file1 = open("/kaggle/input/eeg-input1/Testdata1.txt","r")

file1.seek(0)  

  

i=1

string=file1.readline()

while string:

  datas=list()

  data=''

  for element in range(0, len(string)): 

    if string[element]==' ' and string[element+1]!=' ' and element+1<len(string):

      while string[element+1]!=' ' and element+2<len(string):

        data+=string[element+1]

        element+=1

      datas.append(data)

      data=''

  test1=test1.append(pd.Series(datas, index=head2), ignore_index=True)

  print(i)

  i+=1

  string=file1.readline()
test1
head3=list()

for i in range (0,8065):

    head3.append(i)

#print(head)
train2=pd.DataFrame(columns=head3)

file1 = open("/kaggle/input/eeg-input1/Traindata_0.txt","r")

file1.seek(0)  

  

i=1

string=file1.readline()

while string:

  datas=list()

  data=''

  for element in range(0, len(string)): 

    if string[element]==' ' and string[element+1]!=' ' and element+1<len(string):

      while string[element+1]!=' ' and element+2<len(string):

        data+=string[element+1]

        element+=1

      datas.append(data)

      data=''

  train2=train2.append(pd.Series(datas, index=head3), ignore_index=True)

  print(i)

  i+=1

  string=file1.readline()
train2
head4=list()

for i in range (1,8065):

    head4.append(i)

#print(head)
test2=pd.DataFrame(columns=head4)

file1 = open("/kaggle/input/eeg-input1/Testdata2.txt","r")

file1.seek(0)  

  

i=1

string=file1.readline()

while string:

  datas=list()

  data=''

  for element in range(0, len(string)): 

    if string[element]==' ' and string[element+1]!=' ' and element+1<len(string):

      while string[element+1]!=' ' and element+2<len(string):

        data+=string[element+1]

        element+=1

      datas.append(data)

      data=''

  test2=test2.append(pd.Series(datas, index=head4), ignore_index=True)

  print(i)

  i+=1

  string=file1.readline()
test2
train1.to_csv('train1.csv', index=False)

test1.to_csv('test1.csv', index=False)

train2.to_csv('train2.csv', index=False)

test2.to_csv('test2.csv', index=False)