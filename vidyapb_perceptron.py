from IPython.display import Image

import os

!ls ../input/
Image("../input/perceptron.png")
Image("../input/mathformula.png")
import numpy as np



#Randomly initialize 2 weights and one weight for bias. We're only taking first two decimal places.

weights = np.around(np.random.uniform(size=2), decimals=2)

biases = np.around(np.random.uniform(size=1), decimals=2)
print(weights)

print(biases)
x_1 = 0.7 # input 1

x_2 = 0.95 # input 2



print('x1 is {} and x2 is {}'.format(x_1, x_2))
z_1 = x_1 * weights[0] + x_2 * weights[1] + biases[0]



print('The linear combination of inputs and their weights is {}'.format(z_1))
a_1 = 1.0 / (1.0 + np.exp(-z_1))


print('The output of the network for x1 = 0.7 and x2 = 0.95 is {}'.format(np.around(a_1, decimals=4)))