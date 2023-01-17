# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import  numpy as np
import pandas as pd
from matplotlib import pyplot as plt
x=[]

y=[]
for i in range(20000):

    x.append(-10+i/20000)

    y.append(15*((np.sin(2*x[i]))*(np.sin(2*x[i])))-((x[i]-2)*(x[i]-2))+160)
x.clear()

y.clear()
x

y
for i in range(20000):

    x.append(-10+i/1000)

    y.append(15*((np.sin(2*x[i]))*(np.sin(2*x[i])))-((x[i]-2)*(x[i]-2))+160)
plt.plot(x,y)
t=10+20.*rand(1,30)
import random
sub=[]
for i in range(30):

    sub.append(-10+random.randrange(0,20))
gbest=20

vmafo=20

w=0.5

c1=2

c2=2
pbest=sub
def fitness(a):

    return(15*((np.sin(2*a))*(np.sin(2*a)))-((a-2)*(a-2))+160)
fitness(0)
velocity=[]
for i in range(30):

    velocity.append(0)
velocity
fitness=[]
fitness(0)
def fitness(a):

    return(15*((np.sin(2*a))*(np.sin(2*a)))-((a-2)*(a-2))+160)
fitness(0)
fit=[]
for i in range(30):

    fit.append(0)
fit
for i in range(30):

    fit[i]=fitness(sub[i])
max(fit)
gbest=sub[fit.index(max(fit))]
gbest
for j in range(100):

    for i in range(30):

        r1=np.random.rand()

        r2=np.random.rand()

        vel=w*velocity[i]+r1*c1*(pbest[i]-sub[i])+r2*c2*(gbest-sub[i])

        if vel>vmax:

            vel=vmax

        subnew=sub[i]+vel

        if subnew<-10:

            subnew=-10

        if subnew>10:

            subnew=10

        if fitness(subnew)>fitness(pbest[i]):

            pbest[i]=subnew

        if fitness(subnew)>fitness(gbest):

            gbest=subnew

        sub[i]=subnew

        velocity[i]=vel

        fit[i]=fitness(sub[i])

plt.scatter(sub,fit)
fitness(2.36)