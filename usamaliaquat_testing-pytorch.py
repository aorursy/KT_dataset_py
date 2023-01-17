import torch

import numpy as np
t2 = torch.tensor(4)

t2
t1 = torch.tensor([[1,2],[3,4],[4,2]])

t1
t1.dtype
# Matrixs



t3 = torch.tensor([[.5,4],[7,8],[7,8]])

t3
# Createing Three Dimensional Array



t4 = torch.tensor([

    [[11,12,13,55],

    [14,15,16,44]],

    [[17,18,19,45],

    [20,21,22,54]]

])

t4
t3.shape

t4.shape
x = torch.tensor(3.)

w = torch.tensor(4., requires_grad = True)

b = torch.tensor(5., requires_grad = True)

y = w * x + b

y
y.backward()
y
# Display Gradients



print('dy/dx', x.grad)

print('dy/dw', w.grad)

print('dy/db', b.grad)
x = np.array([[1,2],[3,4]])

x
# Convert the numpy Array to a torch tensor

y = torch.from_numpy(x)

y
x.dtype,   y.dtype
# Convert a torch tensor to a numpy array

z = y.numpy()

z
# Input Temp / Rainfall / Humidity

inputs  = np.array([[73,67,43],

                    [91,88,64],

                    [87,134,85],

                    [102,43,37],

                    [69,96,70]], dtype='float32')