# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import csv as csv 

import numpy as np



with open('../input/test.csv', 'r') as test:

    test_reader = csv.reader(test)

    test_header = next(test_reader)

    data = [['PassengerId', 'Survived']]

    for row in test_reader:

        a = row[test_header.index('PassengerId')]

        b = str(int(row[test_header.index('Sex')] == 'female'))

        data.append([a, b])

    

    #print(test_header)

    

    

    #for row in data:

        #print(row)

        

    a=0

    b=0

    for row in test_reader:

        print (a)

        a = a+int((row[test_header.index('Sex')] == 'female')and(row[test_header.index('Survived')] == '1'))

        b = b+1-int((row[test_header.index('Sex')] == 'female')and(row[test_header.index('Survived')] == '1'))

    print (a)



# Open up the csv file in to a Python object

#csv_file_object = csv.reader(open('../csv/train.csv', 'rb')) 

#header = csv_file_object.next()  # The next() command just skips the 

                                 # first line which is a header

#data=[]                          # Create a variable called 'data'.

#for row in csv_file_object:      # Run through each row in the csv file,

#    data.append(row)             # adding each row to the data variable

#data = np.array(data) 	         # Then convert from a list to an array

			         # Be aware that each item is currently

                                 # a string in this format