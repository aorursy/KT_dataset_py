#Used for parsing through file to create 2d array

import csv

from subprocess import check_output

#Used for the actual graph

import matplotlib.pyplot as plt

%matplotlib inline

#Used to facilitate date calculations

import calendar

import datetime

#Used for summary statistics

import numpy as np



#Creates 2d array where rows are months and columns are day of month

usefuldata = [[0 for j in range(calendar.monthrange(1, i+1)[1])] for i in range(12)]

#Keeps track of the useful columns in data (ignores latitude/longitude etc)

usefulcols = [2,3,4,6,7]

#Parses through entire list of files line by line

for filename in check_output(["ls", "../input"]).decode("utf8").splitlines():

        with open("../input/"+ filename, 'rt') as csvfile:

            reader = csv.reader(csvfile, delimiter=',')

            #Creates array of data in each row

            crime_data = [row for row in reader]

            for row in range(len(crime_data)):

                #Goes through array of data looking specifically for thefts

                if crime_data[row][6]=="THEFT":

                    #If a theft is found then parses the date column to find the month

                    #and day the theft occurred. Ignores year to notice only seasonal

                    #changes within each year

                    date_string = crime_data[row][3]

                    month = int(date_string[:2])

                    day = int(date_string[3:5])

                    if month!=2 or day!=29:

                        #If the date is not a leap day (we chose to ignore this as it

                        #would skew the rest of our data) increment the theft amount

                        #in the array for that given month and day

                        usefuldata[month-1][day-1] = usefuldata[month-1][day-1]+1
#Go through array by month and calculate summary statistics for each month

#(specifically: average, standard deviation, and number of elements)

monthavg = [np.median(usefuldata[i]) for i in range(12)]

monthstdev = [np.std(usefuldata[i]) for i in range(12)]

monthcnt = [len(usefuldata[i]) for i in range(12)]

#Use the above statistics to calculate confidence intervals for each month

confints = [(2*monthstdev[i])/(monthcnt[i]**.5) for i in range(12)]

#Create a new plot, whose size will by 12 by 6

plt.figure(figsize=(12,6))

#The x axis of the plot can simply be the numbers [1,12] (for each month)

xints = [i+1 for i in range(12)]

#Then plot the monthly average over month using plt

line, = plt.plot(xints,monthavg)

#Make the line better looking

plt.setp(line, color='b',linewidth=1.3, linestyle='--', marker='o')

#Use confidence intervals to add width to the line to show error range

plt.fill_between(xints, [monthavg[i]+confints[i] for i in range(12)], 

                 [monthavg[i]-confints[i] for i in range(12)], alpha=0.3, color='gold')

#Label the x axis as months, and replace the numbers in the x axis with month names

plt.xlabel('Month')

plt.xticks(xints, [calendar.month_name[i+1] for i in range(12)])

#Label the y axis and give the graph as a whole a title

plt.ylabel('Median Number of Daily Crimes')

plt.title("Monthly Variation of Crimes in Chicago")