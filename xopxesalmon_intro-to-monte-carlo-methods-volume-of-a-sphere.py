# Monte Carlo method. Find the volume of a sphere with radius equal to 1. The volume of a sphere is 4/3𝝅R**3



from random import random
# N is the number of points to be generated. In order to have a good approximation, N has to be very large, at least 1x10**6

N=1000000

count=0



# N random points P in the cartesian space (x,y,z) are generated. All the random points that fall within the volume of the sphere are counted

for i in range(N):

    P=(random(), random(), random())

    if P[0]**2 + P[1]**2 + P[2]**2 < 1:

        count+=1
# the volume of 1/8 of the sphere is the fraction of points inside the sphere times the volume of the unit cube, which is 1.

v=float(count)/float(N)*1



# The volume of the whole sphere is 8 times previous result

volume=8*v

print(volume)



#which is close to 4/3𝝅R**3, where  R=1