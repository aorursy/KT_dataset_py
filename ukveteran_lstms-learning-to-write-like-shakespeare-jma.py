import numpy as np



class Tensor (object):

    

    def __init__(self,data,

                 autograd=False,

                 creators=None,

                 creation_op=None,

                 id=None):

        

        self.data = np.array(data)

        self.autograd = autograd

        self.grad = None



        if(id is None):

            self.id = np.random.randint(0,1000000000)

        else:

            self.id = id

        

        self.creators = creators

        self.creation_op = creation_op

        self.children = {}

        

        if(creators is not None):

            for c in creators:

                if(self.id not in c.children):

                    c.children[self.id] = 1

                else:

                    c.children[self.id] += 1



    def all_children_grads_accounted_for(self):

        for id,cnt in self.children.items():

            if(cnt != 0):

                return False

        return True 

        

    def backward(self,grad=None, grad_origin=None):

        if(self.autograd):

 

            if(grad is None):

                grad = Tensor(np.ones_like(self.data))



            if(grad_origin is not None):

                if(self.children[grad_origin.id] == 0):

                    return

                    print(self.id)

                    print(self.creation_op)

                    print(len(self.creators))

                    for c in self.creators:

                        print(c.creation_op)

                    raise Exception("cannot backprop more than once")

                else:

                    self.children[grad_origin.id] -= 1



            if(self.grad is None):

                self.grad = grad

            else:

                self.grad += grad

            

            # grads must not have grads of their own

            assert grad.autograd == False

            

            # only continue backpropping if there's something to

            # backprop into and if all gradients (from children)

            # are accounted for override waiting for children if

            # "backprop" was called on this variable directly

            if(self.creators is not None and 

               (self.all_children_grads_accounted_for() or 

                grad_origin is None)):



                if(self.creation_op == "add"):

                    self.creators[0].backward(self.grad, self)

                    self.creators[1].backward(self.grad, self)

                    

                if(self.creation_op == "sub"):

                    self.creators[0].backward(Tensor(self.grad.data), self)

                    self.creators[1].backward(Tensor(self.grad.__neg__().data), self)



                if(self.creation_op == "mul"):

                    new = self.grad * self.creators[1]

                    self.creators[0].backward(new , self)

                    new = self.grad * self.creators[0]

                    self.creators[1].backward(new, self)                    

                    

                if(self.creation_op == "mm"):

                    c0 = self.creators[0]

                    c1 = self.creators[1]

                    new = self.grad.mm(c1.transpose())

                    c0.backward(new)

                    new = self.grad.transpose().mm(c0).transpose()

                    c1.backward(new)

                    

                if(self.creation_op == "transpose"):

                    self.creators[0].backward(self.grad.transpose())



                if("sum" in self.creation_op):

                    dim = int(self.creation_op.split("_")[1])

                    self.creators[0].backward(self.grad.expand(dim,

                                                               self.creators[0].data.shape[dim]))



                if("expand" in self.creation_op):

                    dim = int(self.creation_op.split("_")[1])

                    self.creators[0].backward(self.grad.sum(dim))

                    

                if(self.creation_op == "neg"):

                    self.creators[0].backward(self.grad.__neg__())

                    

                if(self.creation_op == "sigmoid"):

                    ones = Tensor(np.ones_like(self.grad.data))

                    self.creators[0].backward(self.grad * (self * (ones - self)))

                

                if(self.creation_op == "tanh"):

                    ones = Tensor(np.ones_like(self.grad.data))

                    self.creators[0].backward(self.grad * (ones - (self * self)))

                

                if(self.creation_op == "index_select"):

                    new_grad = np.zeros_like(self.creators[0].data)

                    indices_ = self.index_select_indices.data.flatten()

                    grad_ = grad.data.reshape(len(indices_), -1)

                    for i in range(len(indices_)):

                        new_grad[indices_[i]] += grad_[i]

                    self.creators[0].backward(Tensor(new_grad))

                    

                if(self.creation_op == "cross_entropy"):

                    dx = self.softmax_output - self.target_dist

                    self.creators[0].backward(Tensor(dx))

                    

    def __add__(self, other):

        if(self.autograd and other.autograd):

            return Tensor(self.data + other.data,

                          autograd=True,

                          creators=[self,other],

                          creation_op="add")

        return Tensor(self.data + other.data)



    def __neg__(self):

        if(self.autograd):

            return Tensor(self.data * -1,

                          autograd=True,

                          creators=[self],

                          creation_op="neg")

        return Tensor(self.data * -1)

    

    def __sub__(self, other):

        if(self.autograd and other.autograd):

            return Tensor(self.data - other.data,

                          autograd=True,

                          creators=[self,other],

                          creation_op="sub")

        return Tensor(self.data - other.data)

    

    def __mul__(self, other):

        if(self.autograd and other.autograd):

            return Tensor(self.data * other.data,

                          autograd=True,

                          creators=[self,other],

                          creation_op="mul")

        return Tensor(self.data * other.data)    



    def sum(self, dim):

        if(self.autograd):

            return Tensor(self.data.sum(dim),

                          autograd=True,

                          creators=[self],

                          creation_op="sum_"+str(dim))

        return Tensor(self.data.sum(dim))

    

    def expand(self, dim,copies):



        trans_cmd = list(range(0,len(self.data.shape)))

        trans_cmd.insert(dim,len(self.data.shape))

        new_data = self.data.repeat(copies).reshape(list(self.data.shape) + [copies]).transpose(trans_cmd)

        

        if(self.autograd):

            return Tensor(new_data,

                          autograd=True,

                          creators=[self],

                          creation_op="expand_"+str(dim))

        return Tensor(new_data)

    

    def transpose(self):

        if(self.autograd):

            return Tensor(self.data.transpose(),

                          autograd=True,

                          creators=[self],

                          creation_op="transpose")

        

        return Tensor(self.data.transpose())

    

    def mm(self, x):

        if(self.autograd):

            return Tensor(self.data.dot(x.data),

                          autograd=True,

                          creators=[self,x],

                          creation_op="mm")

        return Tensor(self.data.dot(x.data))

    

    def sigmoid(self):

        if(self.autograd):

            return Tensor(1 / (1 + np.exp(-self.data)),

                          autograd=True,

                          creators=[self],

                          creation_op="sigmoid")

        return Tensor(1 / (1 + np.exp(-self.data)))



    def tanh(self):

        if(self.autograd):

            return Tensor(np.tanh(self.data),

                          autograd=True,

                          creators=[self],

                          creation_op="tanh")

        return Tensor(np.tanh(self.data))

    

    def index_select(self, indices):



        if(self.autograd):

            new = Tensor(self.data[indices.data],

                         autograd=True,

                         creators=[self],

                         creation_op="index_select")

            new.index_select_indices = indices

            return new

        return Tensor(self.data[indices.data])

    

    def softmax(self):

        temp = np.exp(self.data)

        softmax_output = temp / np.sum(temp,

                                       axis=len(self.data.shape)-1,

                                       keepdims=True)

        return softmax_output

    

    def cross_entropy(self, target_indices):



        temp = np.exp(self.data)

        softmax_output = temp / np.sum(temp,

                                       axis=len(self.data.shape)-1,

                                       keepdims=True)

        

        t = target_indices.data.flatten()

        p = softmax_output.reshape(len(t),-1)

        target_dist = np.eye(p.shape[1])[t]

        loss = -(np.log(p) * (target_dist)).sum(1).mean()

    

        if(self.autograd):

            out = Tensor(loss,

                         autograd=True,

                         creators=[self],

                         creation_op="cross_entropy")

            out.softmax_output = softmax_output

            out.target_dist = target_dist

            return out



        return Tensor(loss)

        

    

    def __repr__(self):

        return str(self.data.__repr__())

    

    def __str__(self):

        return str(self.data.__str__())  



class Layer(object):

    

    def __init__(self):

        self.parameters = list()

        

    def get_parameters(self):

        return self.parameters



    

class SGD(object):

    

    def __init__(self, parameters, alpha=0.1):

        self.parameters = parameters

        self.alpha = alpha

    

    def zero(self):

        for p in self.parameters:

            p.grad.data *= 0

        

    def step(self, zero=True):

        

        for p in self.parameters:

            

            p.data -= p.grad.data * self.alpha

            

            if(zero):

                p.grad.data *= 0





class Linear(Layer):



    def __init__(self, n_inputs, n_outputs, bias=True):

        super().__init__()

        

        self.use_bias = bias

        

        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0/(n_inputs))

        self.weight = Tensor(W, autograd=True)

        if(self.use_bias):

            self.bias = Tensor(np.zeros(n_outputs), autograd=True)

        

        self.parameters.append(self.weight)

        

        if(self.use_bias):        

            self.parameters.append(self.bias)



    def forward(self, input):

        if(self.use_bias):

            return input.mm(self.weight)+self.bias.expand(0,len(input.data))

        return input.mm(self.weight)





class Sequential(Layer):

    

    def __init__(self, layers=list()):

        super().__init__()

        

        self.layers = layers

    

    def add(self, layer):

        self.layers.append(layer)

        

    def forward(self, input):

        for layer in self.layers:

            input = layer.forward(input)

        return input

    

    def get_parameters(self):

        params = list()

        for l in self.layers:

            params += l.get_parameters()

        return params





class Embedding(Layer):

    

    def __init__(self, vocab_size, dim):

        super().__init__()

        

        self.vocab_size = vocab_size

        self.dim = dim

        

        # this random initialiation style is just a convention from word2vec

        self.weight = Tensor((np.random.rand(vocab_size, dim) - 0.5) / dim, autograd=True)

        

        self.parameters.append(self.weight)

    

    def forward(self, input):

        return self.weight.index_select(input)





class Tanh(Layer):

    def __init__(self):

        super().__init__()

    

    def forward(self, input):

        return input.tanh()





class Sigmoid(Layer):

    def __init__(self):

        super().__init__()

    

    def forward(self, input):

        return input.sigmoid()

    



class CrossEntropyLoss(object):

    

    def __init__(self):

        super().__init__()

    

    def forward(self, input, target):

        return input.cross_entropy(target)



    

class RNNCell(Layer):

    

    def __init__(self, n_inputs, n_hidden, n_output, activation='sigmoid'):

        super().__init__()



        self.n_inputs = n_inputs

        self.n_hidden = n_hidden

        self.n_output = n_output

        

        if(activation == 'sigmoid'):

            self.activation = Sigmoid()

        elif(activation == 'tanh'):

            self.activation == Tanh()

        else:

            raise Exception("Non-linearity not found")



        self.w_ih = Linear(n_inputs, n_hidden)

        self.w_hh = Linear(n_hidden, n_hidden)

        self.w_ho = Linear(n_hidden, n_output)

        

        self.parameters += self.w_ih.get_parameters()

        self.parameters += self.w_hh.get_parameters()

        self.parameters += self.w_ho.get_parameters()        

    

    def forward(self, input, hidden):

        from_prev_hidden = self.w_hh.forward(hidden)

        combined = self.w_ih.forward(input) + from_prev_hidden

        new_hidden = self.activation.forward(combined)

        output = self.w_ho.forward(new_hidden)

        return output, new_hidden

    

    def init_hidden(self, batch_size=1):

        return Tensor(np.zeros((batch_size,self.n_hidden)), autograd=True)

    

class LSTMCell(Layer):

    

    def __init__(self, n_inputs, n_hidden, n_output):

        super().__init__()



        self.n_inputs = n_inputs

        self.n_hidden = n_hidden

        self.n_output = n_output



        self.xf = Linear(n_inputs, n_hidden)

        self.xi = Linear(n_inputs, n_hidden)

        self.xo = Linear(n_inputs, n_hidden)        

        self.xc = Linear(n_inputs, n_hidden)        

        

        self.hf = Linear(n_hidden, n_hidden, bias=False)

        self.hi = Linear(n_hidden, n_hidden, bias=False)

        self.ho = Linear(n_hidden, n_hidden, bias=False)

        self.hc = Linear(n_hidden, n_hidden, bias=False)        

        

        self.w_ho = Linear(n_hidden, n_output, bias=False)

        

        self.parameters += self.xf.get_parameters()

        self.parameters += self.xi.get_parameters()

        self.parameters += self.xo.get_parameters()

        self.parameters += self.xc.get_parameters()



        self.parameters += self.hf.get_parameters()

        self.parameters += self.hi.get_parameters()        

        self.parameters += self.ho.get_parameters()        

        self.parameters += self.hc.get_parameters()                

        

        self.parameters += self.w_ho.get_parameters()        

    

    def forward(self, input, hidden):

        

        prev_hidden = hidden[0]        

        prev_cell = hidden[1]

        

        f = (self.xf.forward(input) + self.hf.forward(prev_hidden)).sigmoid()

        i = (self.xi.forward(input) + self.hi.forward(prev_hidden)).sigmoid()

        o = (self.xo.forward(input) + self.ho.forward(prev_hidden)).sigmoid()        

        g = (self.xc.forward(input) + self.hc.forward(prev_hidden)).tanh()        

        c = (f * prev_cell) + (i * g)



        h = o * c.tanh()

        

        output = self.w_ho.forward(h)

        return output, (h, c)

    

    def init_hidden(self, batch_size=1):

        init_hidden = Tensor(np.zeros((batch_size,self.n_hidden)), autograd=True)

        init_cell = Tensor(np.zeros((batch_size,self.n_hidden)), autograd=True)

        init_hidden.data[:,0] += 1

        init_cell.data[:,0] += 1

        return (init_hidden, init_cell)
import sys,random,math

from collections import Counter

import numpy as np

import sys



np.random.seed(0)

# dataset from http://karpathy.github.io/2015/05/21/rnn-effectiveness/

f = open('../input/shakespear.txt','r')

raw = f.read()

f.close()



vocab = list(set(raw))

word2index = {}

for i,word in enumerate(vocab):

    word2index[word]=i

indices = np.array(list(map(lambda x:word2index[x], raw)))



embed = Embedding(vocab_size=len(vocab),dim=512)

model = RNNCell(n_inputs=512, n_hidden=512, n_output=len(vocab))



criterion = CrossEntropyLoss()

optim = SGD(parameters=model.get_parameters() + embed.get_parameters(), alpha=0.05)



batch_size = 32

bptt = 16

n_batches = int((indices.shape[0] / (batch_size)))



trimmed_indices = indices[:n_batches*batch_size]

batched_indices = trimmed_indices.reshape(batch_size, n_batches).transpose()



input_batched_indices = batched_indices[0:-1]

target_batched_indices = batched_indices[1:]



n_bptt = int(((n_batches-1) / bptt))

input_batches = input_batched_indices[:n_bptt*bptt].reshape(n_bptt,bptt,batch_size)

target_batches = target_batched_indices[:n_bptt*bptt].reshape(n_bptt, bptt, batch_size)
raw[0:5]
indices[0:5]
batched_indices[0:5]
input_batches[0][0:5]
target_batches[0][0:5]
def train(iterations=400):

    for iter in range(iterations):

        total_loss = 0

        n_loss = 0



        hidden = model.init_hidden(batch_size=batch_size)

        for batch_i in range(len(input_batches)):



            hidden = Tensor(hidden.data, autograd=True)

            loss = None

            losses = list()

            for t in range(bptt):

                input = Tensor(input_batches[batch_i][t], autograd=True)

                rnn_input = embed.forward(input=input)

                output, hidden = model.forward(input=rnn_input, hidden=hidden)



                target = Tensor(target_batches[batch_i][t], autograd=True)    

                batch_loss = criterion.forward(output, target)

                losses.append(batch_loss)

                if(t == 0):

                    loss = batch_loss

                else:

                    loss = loss + batch_loss

            for loss in losses:

                ""

            loss.backward()

            optim.step()

            total_loss += loss.data

            log = "\r Iter:" + str(iter)

            log += " - Batch "+str(batch_i+1)+"/"+str(len(input_batches))

            log += " - Loss:" + str(np.exp(total_loss / (batch_i+1)))

            if(batch_i == 0):

                log += " - " + generate_sample(n=70, init_char='\n').replace("\n"," ")

            if(batch_i % 10 == 0 or batch_i-1 == len(input_batches)):

                sys.stdout.write(log)

        optim.alpha *= 0.99

        print()
train(100)
def generate_sample(n=30, init_char=' '):

    s = ""

    hidden = model.init_hidden(batch_size=1)

    input = Tensor(np.array([word2index[init_char]]))

    for i in range(n):

        rnn_input = embed.forward(input)

        output, hidden = model.forward(input=rnn_input, hidden=hidden)

        output.data *= 10

        temp_dist = output.softmax()

        temp_dist /= temp_dist.sum()



        m = (temp_dist > np.random.rand()).argmax()

#         m = output.data.argmax()

        c = vocab[m]

        input = Tensor(np.array([m]))

        s += c

    return s

print(generate_sample(n=2000, init_char='\n'))