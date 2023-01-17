from learntools.core import binder

binder.bind(globals())

from learntools.data_cleaning.ex4 import *

print("Setup Complete")
# modules we'll use

import pandas as pd

import numpy as np



# helpful character encoding module

import chardet



# set seed for reproducibility

np.random.seed(0)
sample_entry = b'\xa7A\xa6n'

print(sample_entry)

print('data type:', type(sample_entry))
sample_entry = sample_entry.decode("big5-tw")

new_entry = sample_entry.encode()



# Check your answer

q1.check()
# Lines below will give you a hint or solution code

#q1.hint()

#q1.solution()
# TODO: Load in the DataFrame correctly.

police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv")



# Check your answer

q2.check()
# (Optional) Use this code cell for any additional work.

with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv","rb") as raw_data:

    result =  chardet.detect(raw_data.read(10000))

print(result)

police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv" , encoding = 'Windows-1252')



#This is the correct code. I think there is some problem!

#police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv" , encoding = 'ascii')

#as you can see {'encoding': 'ascii', 'confidence': 1.0, 'language': ''}

q2.check()
# Lines below will give you a hint or solution code

q2.hint()

q2.solution()
# TODO: Save the police killings dataset to CSV

police_killings.to_csv("police_killings.csv")



# Check your answer

q3.check()
# Lines below will give you a hint or solution code

q3.hint()

q3.solution()