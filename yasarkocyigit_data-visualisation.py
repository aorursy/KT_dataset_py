

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np



"""

WHAT IS MATPLOTLIB?



Matplotlib is a plotting library for the Python programming language 

and its numerical mathematics extension NumPy



You can Generate visualizations with pyplot



"""



# lets generate arrays for examples



x = np.arange(1,6)



print(x)



"""



[1 2 3 4 5]



"""



y = np.arange(2,11,2)



print(y)



"""



[ 2  4  6  8 10]



"""



# NOW WE CAN GENERATE VISUALIZATIONS WITH THE x AND y ARRAYS



# we will use plot function here which means "draw"



plt.plot(x,y,"yellow")

plt.show()

#and we will ise "show" function to see graphic



# when you run this code graphic is gonna appear on your screen



# if you working in jupyter notebook you dont have to use "show" function its optional graphic is gonna appear without it.





# IF YOU WANNA GENERATE MORE THAN 1 GRAPHICS ON SAME SCREEN:

# we will use "subplot" function here



plt.subplot(2,2,1)

plt.plot(x,y,"red")

plt.subplot(2,2,2)

plt.plot(y,x,"black")

plt.subplot(2,2,3)

plt.plot(x,y,"blue")

plt.subplot(2,2,4)

plt.plot(x,x**2,"purple")

plt.show()



# there should be 4 graphics on same screen

# as i said before if youre working in jupyter notebook you dont have to use "show" function



# GENERATING FIGURES:



a=np.arange(1,6)

b=np.arange(2,11,2)



fig = plt.figure()



#and we will ad graphics on this figure:

#we can add 1 or more graphics on same figure



axes1=fig.add_axes([0.1,0.1,0.8,0.8])

axes2=fig.add_axes([0.4,0.5,0.4,0.3])



plt.show()
