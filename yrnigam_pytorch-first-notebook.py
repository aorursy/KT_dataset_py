# Import torch and other required modules
import torch
a = torch.zeros((3,3))
print(a)
print(a.shape)
# Example 2 - working
b = torch.zeros((10,20),dtype=int)
print(b)
print(b.shape)
# Example 3 - working
c = torch.zeros(5,dtype=int)
print(c)
print(c.shape)
# Example 1 - working
ten1=torch.tensor([[1, 2], [3, 4.]])
ten2=torch.tensor([[5, 6], [7, 8.]])
print(ten1,ten2)
torch.cat((ten1,ten2))
# Example 2 - working
ten3 =torch.tensor((1, 2))
ten4 =torch.tensor((3, 4))
print(ten3,ten4)
torch.cat((ten3,ten4))
# Example 3 - breaking (to illustrate when it breaks)
ten5 =torch.tensor((1,2,3))
ten6 = torch.tensor((1.0,2.0))
torch.cat((ten5,ten6))
# Example 1 - working
t = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]
], dtype=torch.float32)
t.reshape([1,12])
# Example 2 - working
t.reshape([2,12])
# Example 3 - breaking (to illustrate when it breaks)
t.reshape([2,4])

# Example 1 - working
r = torch.squeeze(t)     # Size 2x2
print("squeeze",r)
# Example 2 - working
print(t.reshape([1,12]).squeeze())
# Example 3 - breaking (to illustrate when it breaks)

# Example 1 - working
print(t.reshape([1,12]).squeeze().unsqueeze(dim=0))
# Example 2 - working
print(t.reshape([1,12]).squeeze().unsqueeze(dim=0).shape)
# Example 3 - breaking (to illustrate when it breaks)
jovian.commit()
