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
#Finger Exercise 1 
print("this program converts miles to Kilometers")
number_to_kms = int(input("Enter a number in miles:"))

km = number_to_kms /0.62137

print(number_to_kms,"miles is equivalent to",km,"Kilometers")

    
#Finger Exercise 1 
print("this program converts miles to Meters")
number_to_meters = int(input("Enter a number in miles:"))
kms = number_to_kms /0.62137
meters = 1000 * (kms)

print(number_to_meters,"miles is equivalent to",meters,"meters")
#Finger Exercise 2
name = str(input("What is your name:"))
age = int(input("What is your Age:"))

age_in_2047 = age + (2047-2020)

print("Hi",name,"!" "In 2047 you will be",age_in_2047,"!")

