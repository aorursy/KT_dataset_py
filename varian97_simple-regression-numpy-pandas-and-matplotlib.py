import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("../input/train.csv")
data = data.dropna()
data.describe()
x = data['x']
y = data['y']

iterations = 1000
learning_rate = 0.0005
tetha0 = 0.0
tetha1 = 0.0

for iteration in range(iterations):
    output = np.dot(tetha1,x) + tetha0
    loss = output - y
    mse = np.sum(loss ** 2) / (2 * len(y))
    
    if(iteration % 100 == 0):
        print("Iterations {}   |   Error = {}".format(iteration, mse))
    
    # update weight
    tetha0 -= learning_rate * (np.sum(loss) / len(y))
    tetha1 -= learning_rate * (np.sum(x * loss) / len(y))
test_data = pd.read_csv("../input/test.csv")
test_data = test_data.dropna()
test_data.describe()
test_x = test_data['x']
test_y = test_data['y']

output = tetha1 * test_x + tetha0
loss = np.absolute(test_y - output)
mse = np.sum(loss ** 2) / (2 * len(y))

print("Error = ", mse)
print("Tetha1 = {}  Tetha0 = {}".format(tetha1, tetha0))

plt.plot(test_x,test_y, 'ro')
plt.plot(test_x, output)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Linear Regression')
plt.show()