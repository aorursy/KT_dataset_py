# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#generate random input data to train
observations = 1000

xs=np.random.uniform(low=-10, high=10, size=(observations,1))
zs=np.random.uniform(-10,10,(observations,1))

inputs=np.column_stack((xs,zs))

print(inputs.shape)
#create the targets
#targets= f(x,z)=2*xs-3*zs+bias+noise

noise=np.random.uniform(-1,1,(observations,1))
targets= 2*xs-3*zs+5+noise

print(targets.shape)
#plot the training data
targets= targets.reshape(observations,)
fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')
ax.plot(xs,zs,targets)
ax.set_xlabel('xs')
ax.set_ylabel('zs')
ax.set_zlabel('targets')
ax.view_init(azim=100)
plt.show()
targets=targets.reshape(observations,1)
#initialize variables
#our initial weights & biases will picked randomly from the interval[-0.1,0.1]
init_range=0.1

weights= np.random.uniform(-init_range,init_range,size=(2,1)) #w=2x1
biases= np.random.uniform(-init_range, init_range, size=1)    #b=1x1

print(weights)
print(biases)
#set a learning rate

learning_rate = 0.02 #different learning rates affects the speed of optimization

for i in range(100):
  outputs= np.dot(inputs,weights)+ biases #np.dot(A,B) is a method used for multiplying matrices.
 
  deltas= outputs- targets

  loss= np.sum(deltas**2)/2/observations
  # division by a constant doesn't change the logic of a loss, as it is still lower for higher accuracy

  print(loss) # we print loss on each step as we want to keep on eye whether it is decreasing

  deltas_scaled = deltas/observations

  weights = weights - learning_rate* np.dot(inputs.T,deltas_scaled)
             
  biases = biases - learning_rate* np.sum(deltas_scaled)

  
print(weights, biases) 
# plots last outputs vs targets

plt.plot(outputs,targets)
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()