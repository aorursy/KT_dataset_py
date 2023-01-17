#Import relevent libraries 

import numpy as np 
#GENERATED RANDOM DATA

observations=1000000  #we have set total no. of observations to 1000

xs=np.random.uniform(-10,10,observations)   #random 1000 values between -10 and +10 

zs=np.random.uniform(-10,10,observations)

inputs=np.column_stack((xs,zs)) #Generating a single matrix for x and z

inputs.shape
#Generating noise

noise=np.random.uniform(-1,1,observations)  #We will a small noise to our targets

targets=3*xs+8*zs+8+noise
#Initialize weights and biases

init_range=0.1



weights=np.random.uniform(-init_range,init_range,2)

biases=np.random.uniform(-init_range,init_range,1)



print(weights)

print(biases)
#set a leaning rate(I played around with it and found 0.02 works best for this model)

learning_rate=0.02
#No of epochs to run gradient descient is also a hyperparameter(I also played around with it & found 300 works best for this setup)

for i in range(300):

    outputs=np.dot(inputs,weights)+biases

    

    deltas=outputs-targets

    loss=np.sum(deltas**2)/2/observations

    deltas_scaled = deltas / observations

    

    print(loss)

    

    #gradient descient

    weights=weights-learning_rate*np.dot(inputs.T,deltas_scaled)

    biases=biases-learning_rate*np.sum(deltas_scaled)

    
'''In our equation the values were x=3,z=8 and b=8, the values this model gave us are very close 

to the original ones as you can see'''

print(weights)

print(biases)