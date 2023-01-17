import csv as csv

import numpy as np



csv_file_object = csv.reader(open('../input/train.csv', 'rb')) 	# Load in the csv file

							# Then convert from a list to an array.

    										# Create a variable to hold the data



for row in csv_file_object: 							# Skip through each row in the csv file,

    data.append(row[0:]) 								# adding each row to the data variable

data = np.array(data) 									# Then convert from a list to an array.