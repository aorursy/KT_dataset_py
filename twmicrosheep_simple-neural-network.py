# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from itertools import zip_longest

from typing import List, Tuple, Callable, Optional



# Vector Operations

import numpy as np



# For Generating Datasets

from sklearn.datasets import make_circles, make_moons, make_blobs, make_gaussian_quantiles

from sklearn.model_selection import train_test_split



# Progress Bar

from tqdm.auto import tqdm



# Plotting Diagrams

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



# Plotting Animations

from matplotlib.animation import FuncAnimation

from IPython.display import HTML
# Increase figure size

plt.rcParams["figure.figsize"] = (12, 8)
ALL_DATA_TYPE = ["checkboard", "circle", "mul_circle", "moon", "blob", "spiral"]



def gen_data(name, per_class_data_cnt=500, class_cnt=3, show_data=False):

    n_samples = per_class_data_cnt * class_cnt

    if name == "circle":

        class_cnt = 2

        X, Y = make_circles(n_samples=n_samples, noise=0.2, factor=0.3)

    elif name == "mul_circle":

        X, Y = make_gaussian_quantiles(n_samples=n_samples, n_features=2, n_classes=class_cnt)

    elif name == "moon":

        class_cnt = 2

        X, Y = make_moons(n_samples=n_samples, noise=0.1)

    elif name == "blob":

        X, Y = make_blobs(n_samples=n_samples, n_features=2, centers=class_cnt)

    elif name == "checkboard":

        class_cnt = 2

        X = np.zeros((n_samples, 2))

        Y = np.zeros(n_samples)

        offsets = [((1, 1), 0), ((1, -1), 1), ((-1, 1), 1), ((-1, -1), 0)]

        for bid, (offset, y) in enumerate(offsets):

            idx = range(per_class_data_cnt*bid//2, per_class_data_cnt*(bid+1)//2)

            X[idx] = (np.random.rand(per_class_data_cnt//2, 2) + 0.05) * np.array(offset)

            Y[idx] = y

    elif name == "spiral":

        X = np.zeros((n_samples, 2))

        Y = np.zeros(n_samples)

        for cid in range(class_cnt):

            r = np.linspace(0.0, 1, per_class_data_cnt) # radius

            t = np.linspace(cid*4, (cid+1)*4, per_class_data_cnt) + np.random.randn(per_class_data_cnt)*0.3 # theta

            idx = range(per_class_data_cnt*cid, per_class_data_cnt*(cid+1))

            X[idx] = np.c_[r*np.sin(t), r*np.cos(t)]

            Y[idx] = cid

    else:

        raise ValueError("Unknown Data Name!")



    if show_data:

        plt.scatter(X[:, 0], X[:, 1], c=Y, s=20)

        plt.show()

        

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.25)



    # Change index to one hot (Assumes index 0 ~ class_cnt)

    # [1, 2, 0, 1] -> [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]

    def one_hot(x, class_cnt):

        oh = np.zeros((len(x), class_cnt))

        oh[np.arange(len(x)), [int(i) for i in x]] = 1

        return oh.reshape(-1, class_cnt, 1)



    training_data = list(zip(X_train.reshape(-1, 2, 1), one_hot(Y_train, class_cnt)))

    validation_data = list(zip(X_valid.reshape(-1, 2, 1), one_hot(Y_valid, class_cnt)))

    testing_data = list(zip(X_test.reshape(-1, 2, 1), one_hot(Y_test, class_cnt)))

    

    return class_cnt, training_data, validation_data, testing_data
class_cnt, training_data, validation_data, testing_data = gen_data("spiral", show_data=True)
# Activation Functions

class AF():

    @staticmethod

    def linear(x, d=False):

        return 1 if d else x

        

    @staticmethod

    def sigmoid(x, d=False):

        if d:

            return AF.sigmoid(x) * (1 - AF.sigmoid(x))

        else:

            return 1.0 / (1.0 + np.exp(-x))



    @staticmethod

    def tanh(x, d=False):

        if d:

            return 1 - AF.sigmoid(x)*AF.sigmoid(x)

        else:

            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

        

        

    @staticmethod

    def relu(x, d=False):

        if d:

            return 1.0 * (x > 0.0)

        else:

            return x * (x > 0.0)



    @staticmethod

    def leaky_relu(x, d=False):

        if d:

            return 0.99 * (x > 0) + 0.01

        else:

            return 0.99 * x * (x > 0) + 0.01 * x



    @staticmethod

    def selu(x, d=False):

        LAMBDA = 1.0507009873554804934193349852946

        ALPHA = 1.6732632423543772848170429916717

        if d:

            return np.vectorize(lambda xi: LAMBDA if xi > 0.0 else LAMBDA*ALPHA*np.exp(xi))(x)

        else:

            return np.vectorize(lambda xi: LAMBDA*xi if xi > 0.0 else LAMBDA*(ALPHA*np.exp(xi)-1.0))(x)



    @staticmethod

    def softmax(x, d=False):

        EPS = 1e-6

        if d:

            print("Shall not need!")

            return -1

        else:

            exps = np.exp(x - x.max())

            return exps / (exps.sum()+EPS)
# Loss Functions

class LF():

    @staticmethod

    def softmax_crossentropy(activation_output, y, d=False):

        if d:

            return activation_output - y

        else:

            #return -np.multiply(y, np.log(activation_output)).sum()

            return -(np.multiply(y, np.log(activation_output)) + 

                     np.multiply(np.ones(y.shape) - y, np.log(np.ones(activation_output.shape)-activation_output))).sum()
class SimpleNN():



    def __init__(self, structure: List[int], af: Callable):

        # Save the model structure

        self.layer_structure = structure

        self.num_layers = len(structure)

        self.af = af



        # Initialize the weight and bias

        # For weights:

        #   - Standard Normal: np.random.randn

        #   - Normalized Standard Normal: np.random.randn / np.sqrt(<previous_layer_neuron_cnt>)

        #   - Gaussian Normal: np.random.normal

        # For bias:

        #   - All zeros: np.zeros

        #   - Standard Normal: np.random.randn

        #   - Gaussian Normal: np.random.normal

        self.weights = [np.random.randn(nl, pl) / np.sqrt(pl) for pl, nl in zip(structure[:-1], structure[1:])]

        self.biases = [np.zeros((l, 1)) for l in structure[1:]]



    def empty_structure(self):

        # Return empty structure for weight and bias

        return [np.zeros(w.shape) for w in self.weights], [np.zeros(b.shape) for b in self.biases]



    def activation_func(self, l: int, *args, **kargs):

        # Use softmax on last layer

        return AF.softmax(*args, **kargs) if l == self.num_layers-1 else self.af(*args, **kargs)



    def loss_func(self, *args, **kargs):

        return LF.softmax_crossentropy(*args, **kargs)



    def forward(self, x, return_activations=False):

        activations, before_activations = [x], []



        # Run through each layer

        for l, w, b in zip(range(1, self.num_layers), self.weights, self.biases):

            z = np.dot(w, activations[-1]) + b

            before_activations.append(z)

            az = self.activation_func(l, z)

            activations.append(az)



        # Return value history for backprop usage

        return (activations, before_activations) if return_activations else activations[-1]



    def backprop(self, x, y):

        # Init empty data structure to store delta

        delta_w, delta_b = self.empty_structure()

        

        # Feed forward pass

        activations, before_activations = self.forward(x, return_activations=True)



        # Backward pass

        for l in range(1, self.num_layers):

            if l == 1:

                # Last layer calcualte dC/dz from cost function

                delta = self.loss_func(activations[-1], y, d=True)

            else:

                # Calculate dC/dz backwards layer by layer

                zd = self.activation_func(self.num_layers-l, before_activations[-l], d=True)

                pc = np.dot(self.weights[-l+1].transpose(), delta)

                delta = pc * zd

            # Gradients

            delta_w[-l] = np.dot(delta, activations[-l-1].transpose())

            delta_b[-l] = delta



        return delta_w, delta_b

        

    def update_one_batch_optimizer(self, mini_batch, optimizer):

        # Init empty data structure to store delta

        batch_delta_w, batch_delta_b = self.empty_structure()

    

        # Run through the batch of data

        for x, ohy in mini_batch:

            delta_w, delta_b = self.backprop(x, ohy)

            batch_delta_w = [bd+d for bd, d in zip(batch_delta_w, delta_w)]

            batch_delta_b = [bd+d for bd, d in zip(batch_delta_b, delta_b)]

        

        # Average the delta among the same batch

        batch_delta_w = [bd/len(mini_batch) for bd in batch_delta_w]

        batch_delta_b = [bd/len(mini_batch) for bd in batch_delta_b]

        

        # Change the weights and biases by using the optimizer

        self.weights, self.biases = optimizer.step(self.weights, self.biases, batch_delta_w, batch_delta_b)

 

    def loss_one(self, x, y):

        return self.loss_func(self.forward(x), y)

    

    def loss(self, evaluate_data):

        return sum([self.loss_one(x, y) for x, y in evaluate_data]) / len(evaluate_data)



    def evaluate_one(self, x):

        return np.argmax(self.forward(x))



    def evaluate(self, evaluate_data):

        evaluation_result = [(x, np.argmax(y), self.evaluate_one(x)) for x, y in evaluate_data]

        accuracy = sum([y==pred for x, y, pred in evaluation_result]) / len(evaluate_data)

        return accuracy, evaluation_result
class Optimizer():

    def __init__(self, empty_model_structure):

        # Save the empty model structure

        self.empty_model = model.empty_structure()



    def step(weights, biases, batch_delta_w, batch_delta_b):

        # Update Weight and Bias accordingly to the average of deltas of the whole batch

        weights = [w+bd for w, bd in zip(weights, batch_delta_w)]

        biases = [b+bd for b, bd in zip(biases, batch_delta_b)]

        return weights, biases



class SGD(Optimizer):

    def __init__(self, empty_model_structure,

                 lr: float = 0.01):

        # Call Optimizer's init

        super(SGD, self).__init__(empty_model_structure)

        # Save Parameters

        self.lr = lr



    def step(self, weights, biases, batch_delta_w, batch_delta_b):

        # Multiply delta by lr

        ret_batch_delta_w = [-self.lr*bd for bd in batch_delta_w]

        ret_batch_delta_b = [-self.lr*bd for bd in batch_delta_b]



        # Update Weight and Bias accordingly

        weights = [w+bd for w, bd in zip(weights, ret_batch_delta_w)]

        biases = [b+bd for b, bd in zip(biases, ret_batch_delta_b)]

        return weights, biases



class SGDMomentum(Optimizer):

    def __init__(self, empty_model_structure,

                 lr: float = 0.01, momentum: float = 0.9):

        # Call Optimizer's init

        super(SGDMomentum, self).__init__(empty_model_structure)

        # Save Parameters

        self.lr = lr

        self.momentum = momentum

        # Initial momentum storage

        self.momentum_w, self.momentum_b = self.empty_model



    def step(self, weights, biases, batch_delta_w, batch_delta_b):

        # Multiply delta by lr and add previous batch momentum

        ret_batch_delta_w = [-self.lr*bd+bmw*self.momentum for bmw, bd in zip(self.momentum_w, batch_delta_w)]

        ret_batch_delta_b = [-self.lr*bd+bmb*self.momentum for bmb, bd in zip(self.momentum_b, batch_delta_b)]



        # Save current delta for future momentum usage

        self.momentum_w, self.momentum_b = ret_batch_delta_w, ret_batch_delta_b



        # Update Weight and Bias accordingly

        weights = [w+bd for w, bd in zip(weights, ret_batch_delta_w)]

        biases = [b+bd for b, bd in zip(biases, ret_batch_delta_b)]

        return weights, biases



class SGDNesterovMomentum(Optimizer):

    def __init__(self, empty_model_structure,

                 lr: float = 0.01, momentum: float = 0.9):

        # Call Optimizer's init

        super(SGDNesterovMomentum, self).__init__(empty_model_structure)

        # Save Parameters

        self.lr = lr

        self.momentum = momentum

        # Initial momentum storage

        self.momentum_w, self.momentum_b = self.empty_model



    def step(self, weights, biases, batch_delta_w, batch_delta_b):

        # Save current momentum first

        prev_momentum_w, prev_momentum_b = self.momentum_w, self.momentum_b



        # Multiply delta by lr and add previous batch momentum

        ret_batch_delta_w = [-self.lr*bd+bmw*self.momentum for bmw, bd in zip(self.momentum_w, batch_delta_w)]

        ret_batch_delta_b = [-self.lr*bd+bmb*self.momentum for bmb, bd in zip(self.momentum_b, batch_delta_b)]



        # Save current delta for future momentum usage

        self.momentum_w, self.momentum_b = ret_batch_delta_w, ret_batch_delta_b



        # Modify the delta to accomplish look ahead

        ret_batch_delta_w = [-self.momentum*pmw+(1+self.momentum)*cmw for pmw, cmw in zip(prev_momentum_w, self.momentum_w)]

        ret_batch_delta_b = [-self.momentum*pmb+(1+self.momentum)*cmb for pmb, cmb in zip(prev_momentum_b, self.momentum_b)]

        

        # Update Weight and Bias accordingly

        weights = [w+bd for w, bd in zip(weights, ret_batch_delta_w)]

        biases = [b+bd for b, bd in zip(biases, ret_batch_delta_b)]

        return weights, biases



class Adagrad(Optimizer):

    EPS = 1e-6

    def __init__(self, empty_model_structure,

                 lr: float = 0.01):

        # Call Optimizer's init

        super(Adagrad, self).__init__(empty_model_structure)

        # Save Parameters

        self.lr = lr

        # Initial leraning rate storage

        self.lr_w, self.lr_b = self.empty_model

        

    def step(self, weights, biases, batch_delta_w, batch_delta_b):

        # Modify learning rate for each parameter according to gradient

        self.lr_w = [blrw + bd*bd for blrw, bd in zip(self.lr_w, batch_delta_w)]

        self.lr_b = [blrb + bd*bd for blrb, bd in zip(self.lr_b, batch_delta_b)]



        # Multiply by lr with respect to learning rate of each parameter

        ret_batch_delta_w = [-self.lr*bd/(np.sqrt(blrw)+self.EPS) for blrw, bd in zip(self.lr_w, batch_delta_w)]

        ret_batch_delta_b = [-self.lr*bd/(np.sqrt(blrb)+self.EPS) for blrb, bd in zip(self.lr_b, batch_delta_b)]



        # Update Weight and Bias accordingly

        weights = [w+bd for w, bd in zip(weights, ret_batch_delta_w)]

        biases = [b+bd for b, bd in zip(biases, ret_batch_delta_b)]

        return weights, biases

    

class RMSprop(Optimizer):

    EPS = 1e-6

    def __init__(self, empty_model_structure,

                 lr: float = 0.01, decay_rate: float = 0.99):

        # Call Optimizer's init

        super(RMSprop, self).__init__(empty_model_structure)

        # Save Parameters

        self.lr = lr

        self.decay_rate = decay_rate

        # Initial leraning rate storage

        self.lr_w, self.lr_b = self.empty_model



    def step(self, weights, biases, batch_delta_w, batch_delta_b):

        # Modify learning rate for each parameter according to gradient

        self.lr_w = [blrw*self.decay_rate + bd*bd*(1-self.decay_rate) for blrw, bd in zip(self.lr_w, batch_delta_w)]

        self.lr_b = [blrb*self.decay_rate + bd*bd*(1-self.decay_rate) for blrb, bd in zip(self.lr_b, batch_delta_b)]



        # Multiply by lr with respect to learning rate of each parameter

        ret_batch_delta_w = [-self.lr*bd/(np.sqrt(blrw)+self.EPS) for blrw, bd in zip(self.lr_w, batch_delta_w)]

        ret_batch_delta_b = [-self.lr*bd/(np.sqrt(blrb)+self.EPS) for blrb, bd in zip(self.lr_b, batch_delta_b)]



        # Update Weight and Bias accordingly

        weights = [w+bd for w, bd in zip(weights, ret_batch_delta_w)]

        biases = [b+bd for b, bd in zip(biases, ret_batch_delta_b)]

        return weights, biases

    

class Adam(Optimizer):

    EPS = 1e-8

    def __init__(self, empty_model_structure,

                 lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, bias_correction: bool = True):

        # Call Optimizer's init

        super(Adam, self).__init__(empty_model_structure)

        # Save Parameters

        self.lr = lr

        self.beta1 = beta1

        self.beta2 = beta2

        self.bias_correction = bias_correction

        # Initial leraning rate and momentum storage

        self.lr_w, self.lr_b = self.empty_model

        self.momentum_w, self.momentum_b = self.empty_model

        if self.bias_correction:

            # Save the amount of steps ran

            self.optimize_steps = 1



    def step(self, weights, biases, batch_delta_w, batch_delta_b):

        # Modify momentum according to delta with decay

        self.momentum_w = [bmw*self.beta1 + bd*(1-self.beta1) for bmw, bd in zip(self.momentum_w, batch_delta_w)]

        self.momentum_b = [bmb*self.beta1 + bd*(1-self.beta1) for bmb, bd in zip(self.momentum_b, batch_delta_b)]

        # Modify learning rate for each parameter according to gradient

        self.lr_w = [blrw*self.beta2 + bd*bd*(1-self.beta2) for blrw, bd in zip(self.lr_w, batch_delta_w)]

        self.lr_b = [blrb*self.beta2 + bd*bd*(1-self.beta2) for blrb, bd in zip(self.lr_b, batch_delta_b)]



        # Multiply by lr with respect to learning rate of each parameter

        if not self.bias_correction:

            ret_batch_delta_w = [-self.lr*bmw/(np.sqrt(blrw)+self.EPS) for blrw, bmw in zip(self.lr_w, self.momentum_w)]

            ret_batch_delta_b = [-self.lr*bmb/(np.sqrt(blrb)+self.EPS) for blrb, bmb in zip(self.lr_b, self.momentum_b)]

        else:

            ret_batch_delta_w = [-self.lr*(bmw/(1-self.beta1**self.optimize_steps))/(np.sqrt(blrw/(1-self.beta2**self.optimize_steps))+self.EPS) for blrw, bmw in zip(self.lr_w, self.momentum_w)]

            ret_batch_delta_b = [-self.lr*(bmb/(1-self.beta1**self.optimize_steps))/(np.sqrt(blrb/(1-self.beta2**self.optimize_steps))+self.EPS) for blrb, bmb in zip(self.lr_b, self.momentum_b)]

            self.optimize_steps += 1



        # Update Weight and Bias accordingly

        weights = [w+bd for w, bd in zip(weights, ret_batch_delta_w)]

        biases = [b+bd for b, bd in zip(biases, ret_batch_delta_b)]

        return weights, biases
class StopScheduler():

    def __init__(self):

        pass

    

    def step(self, validation_loss):

        # Never Stop

        return False



class EarlyStopScheduler(StopScheduler):

    def __init__(self, patience=10, threshold=1e-4):

        # Save Parameters

        self.patience = patience

        self.threshold = threshold

        # Save past loss

        self.past_loss = []



    def step(self, validation_loss):

        # Append loss to past loss

        self.past_loss.append(validation_loss)

        # Consider to early stop when past loss accumulates larger than patience

        if len(self.past_loss) > self.patience:

            # Early stop if no loss improved more than threshold% in the last patience steps

            early_stop = True

            for pl in self.past_loss[1:]:

                if pl < self.past_loss[0]*(1 + self.threshold):

                    early_stop = False

                    break

            if early_stop:

                # Leave only the best loss

                self.past_loss = [self.past_loss[0]]

                return True

            else:

                # Drop the oldest loss

                self.past_loss = self.past_loss[1:]

                return False
class LRScheduler():

    def __init__(self):

        pass

    

    def step(self, current_lr, validation_loss):

        # Return the same learning rate

        return current_lr



class ReduceLROnPlateau(LRScheduler):

    def __init__(self, decay_factor=0.5, patience=5, min_lr=0.0001, threshold=1e-4):

        # Save Parameters

        self.decay_factor = decay_factor

        self.patience = patience

        self.min_lr = min_lr

        self.threshold = threshold

        # Save past loss

        self.past_loss = []



    def step(self, current_lr, validation_loss):

        # Append loss to past loss

        self.past_loss.append(validation_loss)

        # Consider to reduce lr when past loss accumulates larger than patience

        if len(self.past_loss) > self.patience:

            # Decay if no loss improved more than threshold% in the last patience steps

            decay = True

            for pl in self.past_loss[1:]:

                if pl < self.past_loss[0]*(1 + self.threshold):

                    decay = False

                    break

            if decay:

                # Decay learning rate by decay_factor, and fixed at min_lr

                new_lr = max(current_lr*self.decay_factor, self.min_lr)

                print(f"Learning Rate Reduced: {current_lr} -> {new_lr}")

                # Leave only the best loss

                self.past_loss = [self.past_loss[0]]

            else:

                # learning rate not changed

                new_lr = current_lr

                # Drop the oldest loss

                self.past_loss = self.past_loss[1:]

            return new_lr

        else:

            return current_lr
# Group iterable into size of n

# [1, 2, 3, 4, 5] -> n=2 -> [[1, 2], [3, 4], [5]]

def grouper(iterable, n):

    zip_none = zip_longest(*([iter(iterable)] * n))

    return [list(filter(lambda x: x is not None, zn)) for zn in zip_none]
def train_model(model: SimpleNN,

                optimizer: Optimizer,

                lr_scheduler: LRScheduler,

                earlystop_scheduler: StopScheduler,

                training_data, validation_data,

                epochs: int = 100,

                mini_batch_size: int = 10,

                log_period: int = 10,

                draw_process: bool = False):



    # Record loss and accuracy

    loss_his, acc_his  = [], []

    # If draw_process, store evaluation history and boundary table

    if draw_process:

        print("Draw process will take time to eval models and save boundary!")

        print("Use larger log periods to save time!")

        eval_his = []



    # Run through the epochs

    progress_bar = tqdm(range(epochs))

    for epoch in progress_bar:

        # Create batches using grouper function and run through the batches

        for mini_batch in grouper(training_data, mini_batch_size):

            model.update_one_batch_optimizer(mini_batch, optimizer)



        # Calculate loss and accuracy every counter epochs

        if epoch%log_period == 0:

            loss_his.append((model.loss(training_data), model.loss(validation_data)))

            te, ve = model.evaluate(training_data), model.evaluate(validation_data)

            acc_his.append((te[0], ve[0]))

            if draw_process:

                eval_his.append((calculate_model_boundary(model, te[1]), calculate_model_boundary(model, ve[1])))



            # Update Progress Bar Description

            desc = f"Train Loss: {loss_his[-1][0]:.3f}, Accuracy: {acc_his[-1][0]:.3f}"

            progress_bar.set_description(desc)

            

            # Check if Early Stopping is needed using validation data

            if earlystop_scheduler.step(loss_his[-1][1]):

                print("Early Stopped!")

                return (loss_his, acc_his, eval_his) if draw_process else (loss_his, acc_his)

                

            # Update learning rate according to validation loss 

            optimizer.lr = lr_scheduler.step(optimizer.lr, loss_his[-1][1])



    return (loss_his, acc_his, eval_his) if draw_process else (loss_his, acc_his)
def calculate_model_boundary(model: SimpleNN, evaluation_result):

    # Separate evaluation_result into buckets

    correct, error = [[], [], []], [[], [], []]

    for X, Y, pred in evaluation_result:

        bucket = correct if Y == pred else error

        bucket[0].append(X[0].item())

        bucket[1].append(X[1].item())

        bucket[2].append(Y)

    # Calculate model boundary limits

    x_min, x_max = min(correct[0]+error[0]), max(correct[0]+error[0])

    y_min, y_max = min(correct[1]+error[1]), max(correct[1]+error[1])

    step = max(x_max-x_min, y_max-y_min)/100

    xx, yy = np.meshgrid(np.arange(x_min-step*10, x_max+step*10, step), np.arange(y_min-step*10, y_max+step*10, step))

    # Evaluate mesh grid points on model

    z = np.array([model.evaluate_one(x.reshape(-1, 1)) for x in np.c_[xx.ravel(), yy.ravel()]]).reshape(xx.shape)

    return (correct, error, xx, yy, z)



def draw_model_boundary(model_boundary, title: str = "Metric", prev_ax = None):

    correct, error, xx, yy, z = model_boundary

    # Init Plot if no ax

    if prev_ax is None:

        fig, ax = plt.subplots()

    else:

        ax = prev_ax

    cmap=plt.cm.Spectral

    # Plot boundary

    ax.contourf(xx, yy, z, alpha=0.4, cmap=cmap)

    # Plot Correct and Error Points from evaluation_result

    ax.scatter(correct[0], correct[1], c=correct[2], s=30, marker='o', label='correct', cmap=cmap)

    ax.scatter(error[0], error[1], c=error[2], s=30, marker='x', label='wrong', cmap=cmap)

    # Set the limit on the chart

    ax.set_xlim([xx.min(), xx.max()])

    ax.set_ylim([yy.min(), yy.max()])

    # Show legends

    ax.legend()

    if prev_ax is None:

        plt.show()



def draw_metrics_history(metrics: List[Tuple[float, ...]], metric_names: Tuple[str, ...],

                         current_step: int, title: str = "Metric", prev_ax = None,

                         y_lim: Optional[List[float]] = None, tolog=False, ma_step=2):

    # Init Plot if no ax

    if prev_ax is None:

        fig, ax = plt.subplots()

    else:

        ax = prev_ax

    # Plot each of the lines

    assert len(metrics[0]) == len(metric_names), "Each metric shall have a name!"

    # Get data of each metric

    metric_datas = np.log(np.array(metrics).T) if tolog else np.array(metrics).T

    for metric_name, metric_data in zip(metric_names, metric_datas):

        # Calculate moving average for better plot

        mov_avg_metric = np.concatenate([metric_data[:ma_step-1], np.convolve(metric_data, np.ones(ma_step), 'valid') / ma_step])

        ax.plot(mov_avg_metric[:current_step+1], label=metric_name)

    # Set the limit on the chart

    ax.set_xlim([0, len(metrics)-1])

    if y_lim is not None:

        ax.set_ylim(y_lim)

    else:

        ax.set_ylim([metric_datas.min()-0.5, metric_datas.max()+0.5])

    # Set title

    ax.set_title(title)

    ax.legend()

    if prev_ax is None:

        plt.show()



def draw_process(model, loss_his, acc_his, eval_his, train=False, interval=200):

    # Get the history of train or valid

    evaluate_data_id = 0 if train else 1

    evaluate_data = [his[evaluate_data_id] for his in eval_his]

    print("This will take some time to plot!")

    print(f"Total plots: {len(evaluate_data)}")

    # Prepare the animation

    fig = plt.figure(constrained_layout=True)

    gs = gridspec.GridSpec(ncols=2, nrows=5, figure=fig)

    ax1 = fig.add_subplot(gs[:-1, :])

    ax2 = fig.add_subplot(gs[-1, 0])

    ax3 = fig.add_subplot(gs[-1, 1])

    def update(i):

        # Draw Model Boundary

        ax1.clear()

        ax1_title = f'Train Data Evaluation History: {i:04}' if train else f'Validation Data Evaluation History: {i:04}'

        draw_model_boundary(evaluate_data[i], ax1_title, prev_ax=ax1)

        # Draw Loss Change

        ax2.clear()

        ax2_title = "Loss History (Y is log of loss)"

        draw_metrics_history(loss_his, ("Train", "Valid"), i, ax2_title, prev_ax=ax2, ma_step=2, tolog=True)

        # Draw Accuracy Change

        ax3.clear()

        ax3_title = "Accuracy History"

        draw_metrics_history(acc_his, ("Train", "Valid"), i, ax3_title, prev_ax=ax3, ma_step=2, y_lim=[0, 1])

    anim = FuncAnimation(fig, update, frames=range(0, len(evaluate_data)), interval=interval, blit=False)

    return HTML(anim.to_jshtml())
# Generate the dataset

class_cnt, training_data, validation_data, testing_data = gen_data("spiral")

print(f"Class Count: {class_cnt}")



# Setup the model structure

network = [2] + [4, 4] + [class_cnt]  # Input Dimension + Layers + Output Class Count

model = SimpleNN(network, AF.relu)



# Setup the optimzer and schedulers



# optimizer = SGD(model.empty_structure(), lr=0.01)

# optimizer = SGDMomentum(model.empty_structure(), lr=0.01, momentum=0.9)

# optimizer = SGDNesterovMomentum(model.empty_structure(), lr=0.01, momentum=0.9)

# optimizer = Adagrad(model.empty_structure(), lr=0.01)

# optimizer = RMSprop(model.empty_structure(), lr=0.01, decay_rate=0.99)

optimizer = Adam(model.empty_structure(), lr=0.01, beta1=0.9, beta2=0.999, bias_correction=True)



# lr_scheduler = LRScheduler()

lr_scheduler = ReduceLROnPlateau(decay_factor=0.5, patience=3, min_lr=0.0001)



# stop_scheduler = StopScheduler()

stop_scheduler = EarlyStopScheduler(patience=5, threshold=1e-4)



# Run the training process

loss_his, acc_his, eval_his = train_model(model, optimizer, lr_scheduler, stop_scheduler,

                                          training_data, validation_data,

                                          epochs=300, mini_batch_size=10, log_period=10, draw_process=True)
draw_process(model, loss_his, acc_his, eval_his)
accuracy, evaluation_result = model.evaluate(testing_data)

print("Accuracy:", accuracy)

draw_model_boundary(calculate_model_boundary(model, evaluation_result))