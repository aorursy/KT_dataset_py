import random

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
fig=plt.figure(figsize=(16,6))



X = np.arange(1,5)



ax1=plt.subplot(1,2,1)

ax1.plot(X,[x+(x*random.random()) for x in X])

ax1.set_title('Sub Plot 1')



ax2=plt.subplot(1,2,2)

ax2.plot(X,[x-(x*random.random()) for x in X])

ax2.set_title('Sub Plot 2')



plt.show()
fig,axes=plt.subplots(2,2,figsize=(16,8))



axes[0][0].plot(X,[x+(x*random.random()) for x in X])

axes[0][0].set_title('Sub Plot 1')



axes[0][1].plot(X,[x*(x*random.random()) for x in X])

axes[0][1].set_title('Sub Plot 2')



axes[1][0].plot(X,[x-(x*random.random()) for x in X])

axes[1][0].set_title('Sub Plot 3')



axes[1][1].plot(X,[x/(x*random.random()) for x in X])

axes[1][1].set_title('Sub Plot 4')



plt.show()
fig=plt.figure(figsize=(10,10))



ax1=plt.subplot2grid((4,4),(0,0),colspan=2)

ax1.plot(X,[x+(x*random.random()) for x in X])

ax1.set_title('Plot 1 : (0,0)')



ax2=plt.subplot2grid((4,4),(0,2),colspan=2)

ax2.plot(X,[x-(x*random.random()) for x in X])

ax2.set_title('Plot 2 : (0,3)')



ax3=plt.subplot2grid((4,4),(1,0),rowspan=3,colspan=3)

ax3.plot(X,[x-(x*random.random()) for x in X])

ax3.set_title('Plot 3 : (1,0)')



ax4=plt.subplot2grid((4,4),(1,3),rowspan=3,colspan=1)

ax4.plot(X,[x+(x*random.random()) for x in X])

ax4.set_title('Plot 4 : (1,3)')



fig.tight_layout()



plt.show()


fig=plt.figure(figsize=(10,6))



temp=[ random.uniform(20,40) for i in range(5)]

city=['City A','City B','City C','City D','City E']

y_pos=list(range(1,6))



graph=plt.bar(y_pos, temp,color='green')



plt.xticks(y_pos,city)

plt.title('City Temperature')

plt.xlabel('Cities')

plt.ylabel('Temperature ($^\circ$C)')



for bar,t in zip(graph,temp):

    plt.text(bar.get_x() + bar.get_width()/2.0,bar.get_height(),'%.2f $^\circ$C'%t,ha='center',va='bottom')



plt.show()
fig=plt.figure(figsize=(8,6))



plt.plot(X,np.exp(X))

plt.title('Annotating Exponential Plot using plt.annotate()')

plt.xlabel('x-axis')

plt.ylabel('y-axis')



plt.annotate('Point 1',xy=(2,7),arrowprops=dict(arrowstyle='->'),xytext=(1.25,10))



plt.annotate('Point 2',xy=(3,20),arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=-.2'),xytext=(2,22))



plt.annotate('Point 2',xy=(3,20),arrowprops=dict(arrowstyle='-|>',connectionstyle='angle,angleA=90,angleB=0'),xytext=(3.5,10))



plt.show()
plt.figure(figsize = (13,10))

plt.subplot(221)

plt.plot(X,[x+(x*random.random()) for x in X])

plt.title("With spines")



plt.subplot(222)

plt.plot(X,[x+(x*random.random()) for x in X])

plt.gca().spines['top'].set_visible(False)

plt.gca().spines['right'].set_visible(False)

plt.title("Without spines")



plt.subplot(223)

plt.plot(X,[x+(x*random.random()) for x in X])

plt.gca().spines['left'].set_color('yellow')

plt.gca().spines['bottom'].set_color('red')

plt.title("Colored axes" , color = 'green')



plt.show()
import math

import random



n = 250

x = [random.triangular() for i in range(n)]

y = [random.gauss(0.5, 0.25) for i in range(n)]

colors = [random.randint(1, 4) for i in range(n)]

areas = [math.pi * random.randint(5, 15)**2 for i in range(n)]



plt.figure(figsize = (13,8))

plt.scatter(x, y, s=areas, c=colors, alpha=0.7)

plt.title('Opacity with alpha', fontsize = 14)

plt.xlabel("X")

plt.ylabel("Y")

plt.show()
plt.figure(figsize = (17,8))

plt.subplot(121)

plt.scatter(x, y, s=areas, c=colors, alpha=0.7, zorder=1) # Background

plt.plot([.2, .9], [0, .9], 'k-', lw=2, zorder=2) # Foreground

plt.title("Line in Foreground", fontsize = 14)

plt.xlabel("X")

plt.ylabel("Y")



plt.subplot(122)

plt.scatter(x, y, s=areas, c=colors, alpha=0.7, zorder=2) # Background

plt.plot([.2, .9], [0, .9], 'k-', lw=2, zorder=1) # Foreground

plt.title("Line in Background", fontsize = 14)

plt.xlabel("X")

plt.ylabel("Y")



plt.show()
plt.figure(figsize = (13,8))



plt.scatter(x, y) 



# Add horizontal and vertical lines

plt.axhline(0.5, linestyle ="--")  #horizontal line

plt.axvline(0.5, linestyle ="--")  #vertical line



plt.show()
from matplotlib.patches import Rectangle
plt.figure(figsize = (13,8))



plt.scatter(x, y) 



# Add horizontal and vertical lines

plt.axhline(0.5, linestyle ="--")  #horizontal line

plt.axvline(0.5, linestyle ="--")  #vertical line



plt.gca().add_patch(Rectangle((.3, .3), .4, .45,  fill = False))

plt.show()
plt.figure(figsize = (13,8))



plt.scatter(x, y) 



# Add horizontal and vertical spans

plt.axhspan(.2, .3, alpha = 0.2, color = 'green')  # Horizontal shading

plt.axvspan(.2, .3, alpha = 0.4, color = 'pink')  # Vertical shading



plt.show()
fig = plt.figure(figsize = (13,8))

ax = fig.add_subplot(1, 1, 1)

image = np.random.poisson(10., (100, 80))

i = ax.imshow(image, interpolation='nearest')

fig.colorbar(i)  # note that colorbar is a method of the figure, not the axes

plt.show()
fig = plt.figure(figsize = (13,8))

ax = fig.add_axes([0.1,0.1,0.6,0.8])

image = np.random.poisson(10., (100, 80))

i = ax.imshow(image, interpolation='nearest')

colorbar_ax = fig.add_axes([0.7, 0.1, 0.05, 0.8])

fig.colorbar(i, cax=colorbar_ax)

plt.show()
fig = plt.figure(figsize = (10,8))

ax = fig.add_axes([0.1,0.1,0.6,0.8])

image = np.random.poisson(10., (100, 80))

i = ax.imshow(image, aspect='auto', interpolation='nearest')

colorbar_ax = fig.add_axes([0.7, 0.1, 0.05, 0.8])

fig.colorbar(i, cax=colorbar_ax)

plt.show()
# creating the dataset 

data = {'C':20, 'C++':15, 'Java':30,'Python':35} 



x_pos_course = list(range(4))

courses = list(data.keys()) 

values = list(data.values()) 

   

fig = plt.figure(figsize = (10, 5)) 

  

# creating the bar plot 

plt.bar(courses, values, color ='green',  

        width = 0.4) 



#Modifiying ticks

plt.xticks([i for i in x_pos_course],courses,fontname='Chilanka',rotation=45,fontsize=14)



plt.xlabel("Courses offered", fontsize = 12) 

plt.ylabel("No. of students enrolled", fontsize = 12) 

plt.title("Students enrolled in different courses", fontsize = 14) 

plt.show() 
x = np.linspace(0, 20, 1000)

y1 = np.sin(x)

y2 = np.cos(x)



plt.figure(figsize = (13,8))

plt.plot(x, y1, "-b", label="sine")

plt.plot(x, y2, "-r", label="cosine")



# modifying legend

plt.legend(loc='upper center',ncol=2,frameon=False) 

plt.ylim(-1.5, 2.0)

plt.show()