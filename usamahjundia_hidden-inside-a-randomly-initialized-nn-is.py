# import numpy as np

import cupy as np

import matplotlib.pyplot as plt
np.random.seed(69420)
def glorot_init(shape):

    fan_out,fan_in = shape

    limit = np.sqrt(6/(fan_in+fan_out))

    params = np.random.uniform(low=limit*-1,high=limit,size=shape)

    return params
def kaiming_uniform(shape):

    fan_out, fan_in = shape

    params = np.random.normal(loc=0,scale=np.sqrt(2/fan_in),size=shape)

    return params
def relu(a,prime=False):

    if not prime:

        mask = a > 0

        return a * mask

    else:

        mask = a > 0

        return mask.astype(np.float32)
def softmax(a,prime=False):

    a = np.exp(a)

    a = a / np.sum(a,axis=0)

    return a
def targets_to_onehot(targets,numclass):

    onehot = np.zeros((numclass,len(targets)))

    onehot[targets,np.arange(len(targets))] = 1

    return onehot
def crossentropy_with_logits(logits_out, y,prime=False):

    num_classes, num_samples = logits_out.shape

    y_onehot = targets_to_onehot(y,num_classes)

    assert logits_out.shape == y_onehot.shape

    softmaxed = softmax(logits_out)

    if prime:

        softmaxed[y,np.arange(num_samples)] -=1

        return softmaxed

    else:

        return np.sum(np.nan_to_num(y_onehot * np.log(softmaxed) + (1-y_onehot) * np.log((1-softmaxed))))
def SGDOptimizer(gradients,parameters,learningrate):

    parameters = [p - learningrate * g for p,g in zip(parameters,gradients)]

    return parameters
def MomentumSGD(lr,m):

    vl = None

    def optimizer(grads,params):

        nonlocal vl

        if vl is None:

            vl = [np.zeros(p.shape) for p in params]

        vl = [m * v + lr * g for v,g in zip(vl,grads)]

        params = [p-v for p,v in zip(params,vl)]

        return params    

    return optimizer
class FCNN(object):

    

    def __init__(self, layers_size,initializer_weight,initializer_bias,activation,loss_function,optimizer_w,optimizer_b):

        self.layers_size = layers_size

        self.weights = [initializer_weight((k,j)) for j,k in zip(layers_size[:-1],layers_size[1:])]

        self.biases = [initializer_bias((k,1)) for k in layers_size[1:]]

        self.activation = activation

        self.loss_function = loss_function

        self.optimizer_w = optimizer_w

        self.optimizer_b = optimizer_b

    

    def feedforward(self,X):

        a = X

        zs = []

        acs = [a]

        for w,b in zip(self.weights,self.biases):

            a = np.dot(w,a) + b

#             print(a.shape)

            zs.append(a)

            a = self.activation(a)

            acs.append(a)

        return a, zs, acs

    

    def feedforward_softmax(self,X):

        final_a, _, _ = self.feedforward(X)

        final_a = np.exp(final_a)

#         print(final_a.shape)

        final_a = final_a / np.sum(final_a,axis=0)

        return final_a

    

    def backpropagation(self,X,y):

        input_dim, batch_size = X.shape

        final_activation, zs, acs = self.feedforward(X)

        delta_w = [np.zeros(w.shape) for w in self.weights]

        delta_b = [np.zeros(b.shape) for b in self.biases]

        error = self.loss_function(final_activation,y,prime=True) * self.activation(zs[-1],True)

        delta_w[-1] += np.einsum('ik,jk->ijk',error,acs[-2]).mean(axis=2)

        delta_b[-1] += np.expand_dims(error,1).mean(axis=2)

        for i in range(2,len(self.layers_size)):

            error = np.einsum('ik,ij->jk',error,self.weights[-i+1]) * self.activation(zs[-i],prime=True)

            delta_w[-i] += np.einsum('ik,jk->ijk',error,acs[-i-1]).mean(axis=2)

            delta_b[-i] += np.expand_dims(error,1).mean(axis=2) 

        return delta_w, delta_b

    

    def fit(self,X,y,batch_size,epochs):

        # expecting X and y to be already numpy array with the num_samples as 

        # 1st dim

        history = np.zeros(epochs + 1)

        history -= 1

        index = 0

        for epoch in range(epochs):

            indices = np.arange(X.shape[0],dtype=np.int32)

            np.random.shuffle(indices)

            X = X.copy()[indices,:]

            y = y.copy()[indices]

            i = 0

            batches_X = [X[k:k+batch_size] for k in range(0,X.shape[0],batch_size)]

            batches_y = [y[k:k+batch_size] for k in range(0,y.shape[0],batch_size)]

            losses = 0

            if index == 0:

                for Xb, yb in zip(batches_X,batches_y):

                    logits,_,_ = self.feedforward(Xb.T)

                    history[0] += self.loss_function(logits,yb.T) / (Xb.shape[0] * len(batches_X))

            for Xb, yb in zip(batches_X,batches_y):

                delta_w, delta_b = self.backpropagation(Xb.T, yb.T)

#                 print(delta_w)

                self.weights = self.optimizer_w(delta_w,self.weights)

                self.biases = self.optimizer_b(delta_b,self.biases)

                logits,_,_ = self.feedforward(Xb.T)

                losses += self.loss_function(logits,yb.T) / (Xb.shape[0] * len(batches_X))

#                 print((Xb.shape[0] * len(batches_X)))

            history[index] = losses

            index += 1

            print(f"Epoch {epoch} done. Loss : {losses}")

        return history

    

    def predict(self,X):

        softmaxed = self.feedforward_softmax(X.T)

        preds = np.argmax(softmaxed,axis=0)

        return preds

    

    def score(self, X, y):

        preds = self.predict(X)

        assert preds.shape == y.shape

        return np.mean(preds == y) * 100
import numpy as pp
mnist_data = pp.loadtxt('../input/digit-recognizer/train.csv',delimiter=',', skiprows=1)
mnist_data = np.array(mnist_data)
mnist_label = mnist_data[:,0]

mnist_pixels = mnist_data[:,1:] / 255
smol_batch_pixels = mnist_pixels[:32,:]

smol_batch_label = mnist_label[:32].astype(np.int32)
weight_optimizer = MomentumSGD(0.001,0.9)

bias_optimizer = MomentumSGD(0.001,0.9)

nn_overfit = FCNN([784,100,100,10],kaiming_uniform,kaiming_uniform,relu,crossentropy_with_logits,weight_optimizer,bias_optimizer)
hist = nn_overfit.fit(smol_batch_pixels,smol_batch_label,32,1000)
samples = mnist_pixels[:120,:]

samples_y =  mnist_label[:120]
nn_overfit.score(smol_batch_pixels,smol_batch_label)
plt.figure()

plt.plot(-1 * np.asnumpy(hist[hist != -1]))
weight_optimizer = MomentumSGD(0.001,0.9)

bias_optimizer = MomentumSGD(0.001,0.9)

nn = FCNN([784,100,100,10],kaiming_uniform,kaiming_uniform,relu,crossentropy_with_logits,weight_optimizer,bias_optimizer)
indices = np.arange(len(mnist_label),dtype=np.int32)

np.random.shuffle(indices)
cutoff = int(len(indices) * 0.9)
indices_train = indices[:cutoff]

indices_test = indices[cutoff:]
train_pix, train_lab = mnist_pixels[indices_train], mnist_label[indices_train].astype(np.int32)

test_pix, test_lab = mnist_pixels[indices_test], mnist_label[indices_test].astype(np.int32)
fit_history = nn.fit(train_pix,train_lab,32,30)
plt.figure()

plt.plot(np.asnumpy(fit_history[fit_history != -1]) * -1)
print(f"Performance on training data : {nn.score(train_pix,train_lab)} %")
print(f"Performance on test data : {nn.score(test_pix,test_lab)} %")
class ScoreFCNN(object):

    

    def __init__(self, layers_size,topKPercentage,initializer_weight,initializer_score,activation,loss_function,optimizer):

#         assert len(layers_size) == (len(topKPercentage)+1)

        self.layers_size = layers_size

        self.weights = [initializer_weight((k,j)) for j,k in zip(layers_size[:-1],layers_size[1:])]

        self.scores = [initializer_score(w.shape) for w in self.weights]

        self.topK = [int(j*k*topKPercentage) for j,k in zip(layers_size[:-1],layers_size[1:])]

        assert len(self.weights) == len(self.scores)

        self.activation = activation

        self.loss_function = loss_function

        self.optimizer = optimizer

    

    def feedforward(self,X):

        a = X

        zs = []

        acs = [a]

        for w,s,k in zip(self.weights,self.scores,self.topK):

            ss = s.copy()

            ww = w.copy()

#             assert not np.shares_memory(ss,s)

#             assert not np.shares_memory(ww,w)

#             ss = np.abs(ss)

            ss_flat = ss.ravel()

            sort_idx = ss_flat.argsort()

            ss_flat[sort_idx[-k:]] = 1

            ss_flat[sort_idx[:-k]] = 0

            assert ss.shape == w.shape

            ww = ww * ss

#             print(ww)

#             print(ww.shape)

            a = np.dot(ww,a)

            zs.append(a)

            a = self.activation(a)

            acs.append(a)

        return a, zs, acs

    

    def feedforward_softmax(self,X):

        final_a, _, _ = self.feedforward(X)

        final_a = np.exp(final_a)

        final_a = final_a / np.sum(final_a,axis=0)

        return final_a

    

    def backpropagation(self,X,y):

        input_dim, batch_size = X.shape

        final_activation, zs, acs = self.feedforward(X)

        delta_s = [np.zeros(s.shape) for s in self.scores]

        error = self.loss_function(final_activation,y,prime=True) * self.activation(zs[-1],True)

        delta_s[-1] += np.einsum('ik,jk->ijk',error,acs[-2]).mean(axis=2)

        for i in range(2,len(self.layers_size)):

            error = np.einsum('ik,ij->jk',error,self.weights[-i+1]) * self.activation(zs[-i],prime=True)

            delta_s[-i] += np.einsum('ik,jk->ijk',error,acs[-i-1]).mean(axis=2) * self.weights[-i]

        return delta_s

    

    def fit(self,X,y,batch_size,epochs):

        # expecting X and y to be already numpy array with the num_samples as 

        # 1st dim

        history = np.zeros(epochs + 1)

        history -= 1

        index = 1

        for epoch in range(epochs):

            indices = np.arange(X.shape[0],dtype=np.int32)

            np.random.shuffle(indices)

            X = X.copy()[indices,:]

            y = y.copy()[indices]

            i = 0

            batches_X = [X[k:k+batch_size] for k in range(0,X.shape[0],batch_size)]

            batches_y = [y[k:k+batch_size] for k in range(0,y.shape[0],batch_size)]

            losses = 0

            if index == 0:

                for Xb, yb in zip(batches_X,batches_y):

                    logits,_,_ = self.feedforward(Xb.T)

                    history[0] += self.loss_function(logits,yb.T) / (Xb.shape[0] * len(batches_X))

            for Xb, yb in zip(batches_X,batches_y):

                delta_s = self.backpropagation(Xb.T, yb.T)

#                 print(delta_w)

                self.scores = self.optimizer(delta_s,self.scores)

                logits,_,_ = self.feedforward(Xb.T)

                losses += self.loss_function(logits,yb.T) / (Xb.shape[0] * len(batches_X))

#                 print((Xb.shape[0] * len(batches_X)))

            history[index] = losses

            index += 1

            print(f"Epoch {epoch} done. Loss : {losses}")

        return history

    

    def predict(self,X):

        softmaxed = self.feedforward_softmax(X.T)

        preds = np.argmax(softmaxed,axis=0)

        return preds

    

    def score(self, X, y):

        preds = self.predict(X)

        assert preds.shape == y.shape

        return np.mean(preds == y) * 100
optimizer = MomentumSGD(0.1,0.9)

scorenn = ScoreFCNN([784,200,200,10],0.5,kaiming_uniform,kaiming_uniform,relu,crossentropy_with_logits,optimizer)
hist = scorenn.fit(smol_batch_pixels,smol_batch_label,32,1000)
plt.figure()

plt.plot(np.asnumpy(hist[1:]) * -1)
scorenn.score(smol_batch_pixels,smol_batch_label)
newoptimizer = MomentumSGD(0.1,0.9)

newscorenn = ScoreFCNN([784,250,250,10],0.5,kaiming_uniform,kaiming_uniform,relu,crossentropy_with_logits,newoptimizer)
history = newscorenn.fit(train_pix,train_lab,32,30)
plt.figure()

plt.plot(np.asnumpy(history[1:]) * -1)
print(f"Performance on training data : {newscorenn.score(train_pix,train_lab)} %")
print(f"Performance on test data : {newscorenn.score(test_pix,test_lab) } %")