import torch

a = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
b = torch.tensor([[1,0,0],[0,2,0],[0,0,3]])
c = torch.tensor([1,2,3])
print(a)
print(b)
print(torch.mm(a,b))
print(a @ b)
print(a * (c.unsqueeze(0)))
