import pandas as pd



from pathlib import Path

from IPython.core.debugger import set_trace

from fastai import datasets

import pickle, gzip, math, torch, matplotlib as mpl

import matplotlib.pyplot as plt

from torch import tensor

from torch import nn

from torch import optim

import torch.nn

import torch.nn.functional as F

from torch.nn import init

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler



# File handling

import os

print(os.listdir("../input"))
training_df = pd.read_csv("../input/digit-recognizer/train.csv")

testing_df = pd.read_csv("../input/digit-recognizer/test.csv")

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
valid_df = training_df[35000:42000]

training_df = training_df[0:34999]
x_train = torch.tensor(training_df.drop(['label'], axis = 1).values).float()

y_train = torch.tensor(training_df.label.values).float()

x_valid = torch.tensor(valid_df.drop(['label'], axis = 1).values).float()

y_valid = torch.tensor(valid_df.label.values).float()
n,c = x_train.shape

x_train, x_train.shape, y_train, y_train.shape, x_valid.shape, y_valid.shape, y_train.min(), y_train.max()
mpl.rcParams['image.cmap'] = 'gray'

img = x_train[0]

img.view(28,28).type()

plt.imshow(img.view((28,28)));
weights = torch.randn(784,10)

bias = torch.zeros(10)

weights, bias
def matmul(a,b):

    ar,ac = a.shape # n_rows * n_cols

    br,bc = b.shape

    assert ac==br

    c = torch.zeros(ar, bc)

    for i in range(ar):

        for j in range(bc):

            for k in range(ac): # or br

                c[i,j] += a[i,k] * b[k,j]

    return c
m1 = x_valid[:5]

m2 = weights

m1.shape,m2.shape
%time t1=matmul(m1, m2)
t1.shape
def matmul(a,b):

    ar,ac = a.shape

    br,bc = b.shape

    assert ac==br

    c = torch.zeros(ar, bc)

    for i in range(ar):

        for j in range(bc):

            # Any trailing ",:" can be removed

            c[i,j] = (a[i,:] * b[:,j]).sum()

    return c
%timeit -n 10 _=matmul(m1, m2)
#Example of broadcasting

a = tensor([1., 2, 3])

m = tensor([[1., 2, 3], [4,5,6], [7,8,9]])

# The '1' is brodcasted, to have the same dimension as the tensor.

print(a+1)

# a is brodcasted, to have the same dimension as the matrix.

print(a+m)

# If you want to broadcast column-wise, you can use 'None'

print(a[:,None]+m)
def matmul(a,b):

    ar,ac = a.shape

    br,bc = b.shape

    assert ac==br

    c = torch.zeros(ar, bc)

    for i in range(ar):

#       c[i,j] = (a[i,:]          * b[:,j]).sum() # previous

        c[i]   = (a[i  ].unsqueeze(-1) * b).sum(dim=0)

    return c
%timeit -n 10 _=matmul(m1, m2)
# c[i,j] += a[i,k] * b[k,j]

# c[i,j] = (a[i,:] * b[:,j]).sum()

def matmul(a,b): return torch.einsum('ik,kj->ij', a, b)
%timeit -n 10 _=matmul(m1, m2)
%timeit -n 10 t2 = m1.matmul(m2)
#This is a normalization function. 'm' represents the mean and 's' the standard deviation

def normalize(x, m, s): return (x-m)/s
train_mean,train_std = x_train.mean(),x_train.std()

train_mean,train_std
#Let's normalize our tensors.

x_train = normalize(x_train, train_mean, train_std)

# NB: We use training, not validation mean for validation set. this is VERY IMPORTANT

x_valid = normalize(x_valid, train_mean, train_std)
train_mean,train_std = x_train.mean(),x_train.std()

train_mean,train_std
#'n','m' Shape of our input matrix

n,m = x_train.shape #42000,784

#'c' Number of activations (i.e. at the end of the model, we want a shape-10 vector representing the probabilities of each number from 0 to 9)

c = y_train.max()+1

# num hidden

nh = 50

n,m,c,nh
# simplified kaiming init / he init. We are dividing the weights by sqrt(m) in order to normalize the matrices

w1 = torch.randn(m,nh)/math.sqrt(m)

b1 = torch.zeros(nh)

w2 = torch.randn(nh,1)/math.sqrt(nh)

b2 = torch.zeros(1)
def lin(x, w, b): return x@w + b

t = lin(x_valid, w1, b1)

t.mean(),t.std()
def relu(x): return x.clamp_min(0.)

t = relu(lin(x_valid, w1, b1))

t.mean(),t.std()
# kaiming init / he init for relu

w1 = torch.randn(m,nh)*math.sqrt(2/m)

t = relu(lin(x_valid, w1, b1))

t.mean(),t.std()
w1 = torch.zeros(m,nh)

init.kaiming_normal_(w1, mode='fan_out')

t = relu(lin(x_valid, w1, b1))

t.mean(),t.std()
# what if...?

def relu(x): return x.clamp_min(0.) - 0.5

w1 = torch.randn(m,nh)*math.sqrt(2./m )

t1 = relu(lin(x_valid, w1, b1))

t1.mean(),t1.std()
def lin(x, w, b): return x@w + b



def relu(x): return x.clamp_min(0.) - 0.5



def model(xb):

    l1 = lin(xb, w1, b1)

    l2 = relu(l1)

    l3 = lin(l2, w2, b2)

    return l3
%timeit -n 10 _=model(x_valid)
model(x_valid).shape
def mse(output, targ): return (output.squeeze(-1) - targ).pow(2).mean()
preds = model(x_train)
preds.shape
#computing loss

mse(preds, y_train)
def mse_grad(inp, targ): 

    # grad of loss with respect to output of previous layer

    inp.g = 2. * (inp.squeeze() - targ).unsqueeze(-1) / inp.shape[0]

    

def relu_grad(inp, out):

    # grad of relu with respect to input activations

    inp.g = (inp>0).float() * out.g



def lin_grad(inp, out, w, b):

    # grad of matmul with respect to input

    inp.g = out.g @ w.t()

    w.g = (inp.unsqueeze(-1) * out.g.unsqueeze(1)).sum(0)

    b.g = out.g.sum(0)

    

def forward_and_backward(inp, targ):

    # forward pass:

    l1 = inp @ w1 + b1

    l2 = relu(l1)

    out = l2 @ w2 + b2

    # we don't actually need the loss in backward! we are storing it so we can print it, but it is not used in our model.

    loss = mse(out, targ)

    

    # backward pass:

    mse_grad(out, targ)

    lin_grad(l2, out, w2, b2)

    relu_grad(l1, l2)

    lin_grad(inp, l1, w1, b1)
class Module():

    def __call__(self, *args):

        self.args = args

        self.out = self.forward(*args)

        return self.out

    

    def forward(self): raise Exception('not implemented')

    def backward(self): self.bwd(self.out, *self.args)

        

class Relu(Module):

    def forward(self, inp): return inp.clamp_min(0.)-0.5

    def bwd(self, out, inp): inp.g = (inp>0).float() * out.g

        

class Lin(Module):

    def __init__(self, w, b): self.w,self.b = w,b

        

    def forward(self, inp): return inp@self.w + self.b

    

    def bwd(self, out, inp):

        inp.g = out.g @ self.w.t()

        self.w.g = torch.einsum("bi,bj->ij", inp, out.g)

        self.b.g = out.g.sum(0)

        

class Mse(Module):

    def forward (self, inp, targ): return (inp.squeeze() - targ).pow(2).mean()

    def bwd(self, out, inp, targ): inp.g = 2*(inp.squeeze()-targ).unsqueeze(-1) / targ.shape[0]

        

class Model():

    def __init__(self):

        self.layers = [Lin(w1,b1), Relu(), Lin(w2,b2)]

        self.loss = Mse()

        

    def __call__(self, x, targ):

        for l in self.layers: x = l(x)

        return self.loss(x, targ)

    

    def backward(self):

        self.loss.backward()

        for l in reversed(self.layers): l.backward()
#Initializing the gradient of the weights and biases

w1.g,b1.g,w2.g,b2.g = [None]*4

#Creating a new empty model

model = Model()
%time loss = model(x_train, y_train)
%time model.backward()
def log_softmax(x): return (x.exp()/(x.exp().sum(-1,keepdim=True))).log()
sm_pred = log_softmax(preds)
#Negative log-likelihood

def nll(input, target): return -input[range(target.shape[0]), target].mean()
# loss = nll(sm_pred, y_train)

# loss
def log_softmax(x): return x - x.exp().sum(-1,keepdim=True).log()
def logsumexp(x):

    m = x.max(-1)[0]

    return m + (x-m[:,None]).exp().sum(-1).log()



def log_softmax(x): return x - x.logsumexp(-1,keepdim=True)
# F.cross_entropy(preds, y_train)
loss_func = F.cross_entropy



#Let's define an accuracy metric. We take the argmax of our output, to find out which of the numbers of the softmax is the highest.

#The index of that is our prediction. Then we check if it's equal with the real value, and we take the mean of it.

def accuracy(out, yb): return (torch.argmax(out, dim=1)==yb).float().mean()



class Model(nn.Module):

    def __init__(self, n_in, nh, n_out):

        super().__init__()

        self.layers = [nn.Linear(n_in,nh), nn.ReLU(), nn.Linear(nh,n_out)]

        

    def __call__(self, x):

        for l in self.layers: x = l(x)

        return x



#Our simple Model

model = Model(m, nh, 10)

# batch size

bs=64                  

# a mini-batch from x

xb = x_train[0:bs]   

# predictions

preds = model(xb)      

preds[0], preds.shape
yb = y_train[0:bs]

yb = yb.long()

print(loss_func(preds, yb))

print(accuracy(preds, yb))
lr = 0.5   # learning rate

epochs = 1 # how many epochs to train for



#We go through the loop for each epoch

for epoch in range(epochs):

    # We divide the loop into small batches (batch size has no impact on the results of the model)

    for i in range((n-1)//bs + 1):

        start_i = i*bs

        end_i = start_i+bs

        xb = x_train[start_i:end_i]

        yb = y_train[start_i:end_i]

        yb = yb.long()

        # We do a forward pass, then compute the loss between the results of the model and the actuel values

        loss = loss_func(model(xb), yb)

        

        #We do a backward pass (computing the gradient of the loss with respect to every parameters of the model)

        loss.backward()

        with torch.no_grad():

            #For each layer, we update weight and bias values, according to their gradient and learning rate

            for l in model.layers:

                if hasattr(l, 'weight'):

                    l.weight -= l.weight.grad * lr

                    l.bias   -= l.bias.grad   * lr

                    l.weight.grad.zero_()

                    l.bias  .grad.zero_()
loss_func(model(xb), yb), accuracy(model(xb), yb)
def get_model():

    model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))

    return model, optim.SGD(model.parameters(), lr=lr)



model,opt = get_model()

print(model)

loss_func(model(xb), yb)
for epoch in range(epochs):

    for i in range((n-1)//bs + 1):

        start_i = i*bs

        end_i = start_i+bs

        xb = x_train[start_i:end_i]

        yb = y_train[start_i:end_i]

        yb = yb.long()

        pred = model(xb)

        loss = loss_func(pred, yb)



        loss.backward()

        opt.step()

        opt.zero_grad()
loss,acc = loss_func(model(xb), yb), accuracy(model(xb), yb)

loss,acc
class Dataset():

    def __init__(self, x, y): self.x,self.y = x,y

    def __len__(self): return len(self.x)

    def __getitem__(self, i): return self.x[i],self.y[i]



class Sampler():

    def __init__(self, ds, bs, shuffle=False):

        self.n,self.bs,self.shuffle = len(ds),bs,shuffle

        

    def __iter__(self):

        self.idxs = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)

        for i in range(0, self.n, self.bs): yield self.idxs[i:i+self.bs]

    

def collate(b):

    xs,ys = zip(*b)

    return torch.stack(xs),torch.stack(ys)



class DataLoader():

    def __init__(self, ds, sampler, collate_fn=collate):

        self.ds,self.sampler,self.collate_fn = ds,sampler,collate_fn

        

    def __iter__(self):

        for s in self.sampler: yield self.collate_fn([self.ds[i] for i in s])

            

#The fit() function is now clean and very understandable

def fit():

    for epoch in range(epochs):

        for xb,yb in train_dl:

            pred = model(xb)

            loss = loss_func(pred, yb.long())

            loss.backward()

            opt.step()

            opt.zero_grad()
train_ds,valid_ds = Dataset(x_train, y_train),Dataset(x_valid, y_valid)

train_samp = Sampler(train_ds, bs, shuffle=True)

valid_samp = Sampler(valid_ds, bs, shuffle=False)

train_dl = DataLoader(train_ds, sampler=train_samp, collate_fn=collate)

valid_dl = DataLoader(valid_ds, sampler=valid_samp, collate_fn=collate)

xb,yb = next(iter(train_dl))

#Note that every time we run this cell, a different image will be displayed, as "shuffle" is set to True. For our validation set, it should be set to False.

plt.imshow(xb[1].view(28,28))
model,opt = get_model()

fit()

loss,acc = loss_func(model(xb), yb.long()), accuracy(model(xb), yb.long())

loss,acc
train_dl = torch.utils.data.DataLoader(train_ds, bs, sampler=RandomSampler(train_ds), collate_fn=collate)

valid_dl = torch.utils.data.DataLoader(valid_ds, bs, sampler=SequentialSampler(valid_ds), collate_fn=collate)



#PyTorch's defaults work fine for most things

#train_dl = DataLoader(train_ds, bs, shuffle=True, drop_last=True)

#valid_dl = DataLoader(valid_ds, bs, shuffle=False)



model,opt = get_model()

fit()

loss,acc = loss_func(model(xb), yb.long()), accuracy(model(xb), yb.long())

loss,acc
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):

    for epoch in range(epochs):

        # Handle batchnorm / dropout

        model.train()

#         print(model.training)

        for xb,yb in train_dl:

            loss = loss_func(model(xb), yb.long())

            loss.backward()

            opt.step()

            opt.zero_grad()



        model.eval()

#         print(model.training)

        with torch.no_grad():

            tot_loss,tot_acc = 0.,0.

            for xb,yb in valid_dl:

                pred = model(xb)

                tot_loss += loss_func(pred, yb.long())

                tot_acc  += accuracy (pred,yb.long())

        nv = len(valid_dl)

        print(epoch, tot_loss/nv, tot_acc/nv)

    return tot_loss/nv, tot_acc/nv
def get_dls(train_ds, valid_ds, bs, **kwargs):

    return (torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),

            torch.utils.data.DataLoader(valid_ds, batch_size=bs*2, **kwargs))
train_dl,valid_dl = get_dls(train_ds, valid_ds, bs)

model,opt = get_model()

loss,acc = fit(5, model, loss_func, opt, train_dl, valid_dl)
class DataBunch():

    def __init__(self, train_dl, valid_dl, c=None):

        self.train_dl,self.valid_dl,self.c = train_dl,valid_dl,c

        

    @property

    def train_ds(self): return self.train_dl.dataset

        

    @property

    def valid_ds(self): return self.valid_dl.dataset

    

def get_model(data, lr=0.5, nh=50):

    m = data.train_ds.x.shape[1]

    model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,data.c))

    return model, optim.SGD(model.parameters(), lr=lr)



class Learner():

    def __init__(self, model, opt, loss_func, data):

        self.model,self.opt,self.loss_func,self.data = model,opt,loss_func,data

        

def fit(epochs, learn):

    for epoch in range(epochs):

        learn.model.train()

        for xb,yb in learn.data.train_dl:

            loss = learn.loss_func(learn.model(xb), yb)

            loss.backward()

            learn.opt.step()

            learn.opt.zero_grad()



        learn.model.eval()

        with torch.no_grad():

            tot_loss,tot_acc = 0.,0.

            for xb,yb in learn.data.valid_dl:

                pred = learn.model(xb)

                tot_loss += learn.loss_func(pred, yb)

                tot_acc  += accuracy (pred,yb)

        nv = len(learn.data.valid_dl)

        print(epoch, tot_loss/nv, tot_acc/nv)

    return tot_loss/nv, tot_acc/nv
data = DataBunch(*get_dls(train_ds, valid_ds, bs), c)

learn = Learner(*get_model(data), loss_func, data)

loss,acc = fit(1, learn)