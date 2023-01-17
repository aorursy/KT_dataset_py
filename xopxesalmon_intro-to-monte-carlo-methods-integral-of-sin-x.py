from random import random

from random import uniform

from math import sin

from math import pi



# Number of random points to be generated

N=1000000
count=0

for i in range(N):

    point=(uniform(0,pi), random())

    if point[1] < sin(point[0]):

        count+=1

        

answer=(float(count)/float(N))*pi

print(answer)