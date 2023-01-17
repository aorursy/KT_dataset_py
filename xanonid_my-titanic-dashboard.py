import csv as csv 
import numpy as np
import os
import subprocess

#print(subprocess.check_output("ls -la ../input",shell=True).decode("utf-8"))

# Open up the csv file in to a Python object
csv_file_object = csv.reader(open('../input/train.csv', 'r')) 
header = next(csv_file_object)
print(header)

data=[]
for row in csv_file_object:
    data.append(row)      
data = np.array(data)  
print(data)
			         