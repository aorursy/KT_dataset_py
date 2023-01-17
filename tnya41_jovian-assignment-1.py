import torch
#Example 1

a=torch.tensor([1])

torch.is_tensor(a)
#Example 2

b=torch.tensor([[1,1],[2,4]])

torch.is_tensor(b)
x=torch.ones(2,2,3)

print(x)

#few more operations 

#(R,C)

y= torch.ones(3,2)

print(y)

w=torch.ones(1,2)+3

print(w)
print(torch.numel(x))

#2*2*3

print(torch.numel(y))

print(torch.numel(w))

print(torch.numel(y[1]))

print(torch.numel(x[0][1]))

print(torch.numel(x[0][1][2]))

#torch.numel(x+y)

#to perfrom this operation both x and y should be of same size
x1 = torch.rand(2,3) 

y1 = torch.rand(3,4) 
print(x1)

print(y1)
import numpy as np

x3=np.zeros((2,2))
print(x3)
type(x3)
X=torch.as_tensor(x3)
print(X)
xy=np.arange(1,8,2)

print(xy)

xy1=np.arange(1,10,3)

print(xy1)

#arange([start,] stop[, step,][, dtype]) 

"""start : [optional] start of interval range. By default start = 0

stop  : end of interval range

step  : [optional] step size of interval. By default step size = 1,  

For any output out, this is the distance between two adjacent values, out[i+1] - out[i]. 

dtype : type of output array"""

#refer https://www.geeksforgeeks.org/numpy-arange-python/
A=torch.as_tensor(xy,xy1)

#as_tensor() takes 1 positional argument but 2 were given
X
gen=torch.empty(X)

#doesnt work when you pass tensor as an argument always pass shape of a tensor
input=torch.empty(X.shape)

print(input)
c=np.arange(6)

print(c)
C=torch.empty(c.shape)

print(C)
X
torch.unbind(X)