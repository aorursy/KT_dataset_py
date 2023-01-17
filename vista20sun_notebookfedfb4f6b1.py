# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv as csv

import matplotlib.pyplot as plt

import matplotlib.pyplot as pyt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# hypothesis :

# 1.attacks happend in the U.S.A

# 2.attacks increase year by year

# 3.most people that be attacked was man

type_a={};

year={};

country={};

male=0;

fmale=0;

csv_file_object=csv.reader(open('../input/attacks.csv', encoding = "ISO-8859-1"))
next(csv_file_object)

for row in csv_file_object: 

    if row[9]=='M':

        male=male+1

    elif row[9]=='F':

        fmale=fmale+1

    if int(row[2]) in year and int(row[2]) is not 0:

        year[int(row[2])]=year[int(row[2])]+1

    else:

        year[int(row[2])]=1

    if row[3] in type_a:

        type_a[row[3]]= type_a[row[3]]+1

    else:

        type_a[row[3]]=1

    if row[4] in country:

        country[row[4]]=country[row[4]]+1

    else:

        country[row[4]]=1
s_year=sorted(year.items(),key=lambda d:d[1],reverse =True)

s_type=sorted(type_a.items(),key=lambda d:d[1],reverse =True)

s_country=sorted(country.items(),key=lambda d:d[1],reverse =True)
print("male be attacked percent is %s%%" % (male/(male+fmale)*100))

print('\nmost attacked happend in this 5 years:')

for i in range(0,5):

    print(s_year[i])

print('\nmost attacked happend in this 5 countries:')

for i in range(0,5):

    print(s_country[i])

print('\nattacked type:')

for x in s_type:

    print(x)
size=np.arange(1,s_year[0][1],1)

#plt.plot(size,[s_year[x]for x in size],'ro')

plt.plot([s_year[x][0]for x in size],[s_year[x][1]for x in size],'ro')

plt.title("attacks-years relation") 

plt.show();
labels=[x[0]for x in s_type]

sizes=[x[1]for x in s_type]

fig=pyt.figure()

pyt.pie(sizes,labels=labels,autopct='%1.2f%%');

pyt.axis('equal')

pyt.title("attacks-type relation") 

pyt.show()
labels=[x[0]for x in s_country]

sizes=[x[1]for x in s_country]

pyt.pie(sizes,labels=labels,autopct='%1.2f%%');

pyt.axis('equal')

pyt.title("attacks-countries relation")

pyt.show()
labels=['male','fmale']

sizes=[male,fmale]

fig=pyt.figure()

pyt.pie(sizes,labels=labels,autopct='%1.2f%%');

pyt.axis('equal')

pyt.title("attacks-sexual relation")

pyt.legend()

pyt.show()
# hypothesis #1 true

# hypothesis #2 false:

#      attacks happend more heavily after 2000 but not increase year by year

# hypothesis #3 true