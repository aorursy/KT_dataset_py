import torch
#Scalar
x = torch.empty(1) 
print(x)


# vector, 1D
x = torch.empty(3) 
print(x)
# matrix, 2D
x = torch.empty(2,3) 
print(x)
# tensor, 3 dimensions
x = torch.empty(2,2,3) 
print(x)
# tensor, 4 dimensions
y = torch.empty(2,2,2,3) 
print(y)
#creating random numbers using torch. we can use torch.rand(size)
x = torch.rand(5, 3)
print(x)

# torch.zeros(size), fill with 0
# torch.ones(size), fill with 1
x = torch.zeros(5, 3)
print(x)

y=torch.ones(5,3)
print(y)
# we can check size by using size keyword
print(x.size())
# check data type
print(x.dtype)
# we can specify datatypes while creating tensors  float32 default
x = torch.zeros(5, 3, dtype=torch.float16)
print(x)

x = torch.tensor([5.5, 3])
print(x.size())
# Operations
y = torch.rand(2, 2)
x = torch.rand(2, 2)

# elementwise addition
z = x + y

# Addition
print(torch.add(x,y))

#Substarction
print(torch.sub(x, y))

#multiplication
print(torch.mul(x,y))

#division

print(torch.div(x,y))
# Slicing
x = torch.rand(5,3)
print(x)
print(x[:, 0]) # all rows, column 0
print(x[1, :]) # row 1, all columns
print(x[1,1]) # element at 1, 1
# Get the actual value if only 1 element in your tensor
print(x[1,1].item())

# Reshape with torch.view()
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# if -1 it pytorch will automatically determine the necessary size
print(x.size(), y.size(), z.size())
# Numpy
# Converting a Torch Tensor to a NumPy array and vice versa is very easy
a = torch.ones(5)
print(a)

# torch to numpy with .numpy()
b = a.numpy()
print(b)
print(type(b))
# Carful: If the Tensor is on the CPU (not the GPU),
# both objects will share the same memory location, so changing one
# will also change the other
a.add_(1)
print(a)
print(b)
# numpy to torch with .from_numpy(x)
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)

# again be careful when modifying
a += 1
print(a)
print(b)

# by default all tensors are created on the CPU,
# but you can also move them to the GPU (only if it's available )
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    # z = z.numpy() # not possible because numpy cannot handle GPU tenors
    # move to CPU again
    z.to("cpu")       # ``.to`` can also change dtype together!
    print(z = z.numpy())