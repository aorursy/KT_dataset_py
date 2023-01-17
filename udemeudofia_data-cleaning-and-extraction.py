#Import the necessary modules for data cleaning and processing

import numpy as np

import pandas as pd
#Load the excel file into your worksheet

df = pd.ExcelFile('../input/lga-sen-districts-dataset/LGA_SEN_Districts.xlsx')
#Select the worksheet to work on and save it as data

data = df.parse('LGA_SEN_Districts')

#Information about the data

data.info()
#Outputs the first 50 rows of the data

data.head(50)
#Outputs the last 5 rows of the data

data.tail()
#Outputs all the columns in the data

data.columns = ['State', 'Senatorial District', 'Code', 'Composition', 'Collation Center']
#Drops all the rows with null values

data = data.dropna()
data.info()
data.head(20)
data.head(80)
#Filters the rowa and column to get the clean data

data = data[data.State != 'S/N']
data.head(80)
#Get the 'State' column from the 'Senatorial District' column

state_list = []

for cell in data['Senatorial District']:

    sp = cell.split(' ')

    state = sp[0]

    state_list.append(state)

data['State'] = state_list
data
#Reset the index

data = data.reset_index(drop=True)
data
#For Akwa Ibom, Cross River, and FCT, slice using index and name the State properly

data.iloc[3:6,0] = 'AKWA IBOM'

data.iloc[24:27,0] = 'CROSS RIVER'

data.iloc[-1,0] = 'FCT'
data
#Export clean data

data.to_csv('clean_file.csv', header = True, index = False)