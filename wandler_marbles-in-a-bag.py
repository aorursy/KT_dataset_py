# Library used for standard deviation calculation

import statistics



# Relative weight of different marbles

marbleWeight1 = 1

marbleWeight2 = 2



# Initial number of each marble type in the bag

numberMarbles1 = 1

numberMarbles2 = 10



# Initiate an empty list for the bag

marbleBag = []



# then add the appropriate number of each marble type

for i in range(numberMarbles1):

    marbleBag.append(marbleWeight1)



for i in range(numberMarbles2):

    marbleBag.append(marbleWeight2)
import collections



print("Bag: ",marbleBag)

print("Counts: ",collections.Counter(marbleBag))
# Rounding statistics to 4 decimal places

print("Mean: ",round(statistics.mean(marbleBag),4))

print("Median: ",statistics.median(marbleBag))

print("Mode: ",statistics.mode(marbleBag))

print("Std dev: ",round(statistics.pstdev(marbleBag),4))

# We create a copy of the initial marble bag first

# allowing this section of code to be re-run and keeping the initial bag conditions

marbleBagUpdated = list(marbleBag)



# Record the initial std dev of the weight distribution

stdevBag = [statistics.pstdev(marbleBagUpdated)]



# Define how many marbles to track

numMarblesToAdd = 100



# Then add one at a time and record the new std dev of the weight distribution

for i in range(numMarblesToAdd):

    marbleBagUpdated.append(marbleWeight1)

    stdevBag.append(statistics.pstdev(marbleBagUpdated))

print("Bag: ",marbleBagUpdated)

print("Counts: ",collections.Counter(marbleBagUpdated))
# Libraries used for plotting and finding the position of the min/max standard deviation

import matplotlib.pyplot as plt

import numpy as np



# Create the plot

plt.xlabel('Number of type 1 marbles in the bag')

plt.ylabel('Std dev of weight distribution')

plt.plot(stdevBag)

plt.show()



# Get the min/max statistics

# (note Python lists are indexed from zero, so 0 = 1 marble etc, hence the +1 below)

print("Max Std dev: ",round(max(stdevBag),4))

print("Max Std dev position: ",np.argmax(stdevBag)+1)

print("Min Std dev: ",round(min(stdevBag),4))

print("Min Std dev position: ",np.argmin(stdevBag)+1)