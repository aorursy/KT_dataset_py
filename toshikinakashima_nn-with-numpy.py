import numpy as np 

        

class Layer:

    def __init__(self, neurons_size, weights_size):

        self.neurons = np.empty(neurons_size)

        if weights_size != 0: self.weights = np.random.randn(*weights_size)

        if weights_size != 0: self.weights_div = np.empty(neurons_size)



class Input(Layer):

     def __init__(self, size):

        self.size = size

        super().__init__(self.size, 0)



class Dense(Layer):

    def __init__(self, size, input_shape):

        self.size = size

        self.input_shape = input_shape

        super().__init__(self.size, (input_shape, self.size))

    def forward(self, inputs):

        self.neurons = inputs @ self.weights

    def backward(self, inputs, pre_neurons):

        self.weights_div = np.dot(pre_neurons.T, inputs)

        return inputs @ self.weights.T

        

class ReLU(Layer):

    def __init__(self, size):

        self.size = size

        super().__init__(self.size, 0)

    def forward(self, inputs):

        self.neurons = np.where(inputs>0, inputs, 0)

    def backward(self, inputs, pre_neurons):

        return np.where(pre_neurons>0, inputs, 0)   



class Sigmoid(Layer):

    def __init__(self, size):

        self.size = size

        super().__init__(self.size, 0)

    def forward(self, inputs):

        self.neurons = 1 / (1 + np.exp(-inputs))

    def backward(self, inputs, pre_neurons):

        f = 1 / (1 + np.exp(-pre_neurons))

        return (1 - f)*f*inputs



class Softmax_with_cetr(Layer):

    def __init__(self, size):

        self.size = size

        super().__init__(self.size, 0)

    def forward(self, inputs):

        self.neurons = np.exp(inputs) / (np.sum(np.exp(inputs), axis=1).reshape(-1, 1))

    def backward(self, inputs, pre_neurons):

        return inputs
import numpy as np 



class Model:

    def __init__(self, layers_list):

        self.layers_list = layers_list

        pass



    def forward(self, inputs):

        self.layers_list[0].neurons = inputs

        pre_neurons = inputs

        for layer in self.layers_list[1:]:

            layer.forward(pre_neurons)

            pre_neurons = layer.neurons



    def backward(self, labels):

        #error = 0.5 * np.sum((self.layers_list[-1].neurons - labels)**2)

        error = -np.sum(labels*np.log(self.layers_list[-1].neurons+1e-8))

        pred = self.layers_list[-1].neurons.argmax(axis=1)

        ans = labels.argmax(axis=1)

        hit = np.count_nonzero(pred==ans)

        input_div = self.layers_list[-1].neurons - labels

        rev_layers_list = self.layers_list[::-1]

        for idx, layer in enumerate(rev_layers_list[:-1]):

            pre_neurons = rev_layers_list[idx+1].neurons

            input_div = layer.backward(input_div, pre_neurons)

        return error, hit



    def fit(self, learn_late):

        for layer in self.layers_list:

            if hasattr(layer, 'weights'):

                layer.weights -= layer.weights_div * learn_late

                

    def train(self, all_inputs, all_labels, learn_late, epoch, batch_size):

        error_list=[]

        acc_list=[]

        #image_no = labels.shape[0]

        batch_no = all_labels.shape[0] // batch_size

        image_no = batch_size*batch_no

        for e in range(1, 1+epoch):

            p = np.random.permutation(len(all_inputs))

            all_inputs = all_inputs[p]

            all_labels = all_labels[p]

            inputs_list = all_inputs[:image_no,:].reshape([batch_no,batch_size,-1])

            labels_list = all_labels[:image_no,:].reshape([batch_no,batch_size,-1])

            print("epoch:", e, "/", epoch, end=" ")

            epoch_error = 0

            epoch_hit = 0

            for (inputs, labels) in zip(inputs_list, labels_list):

                self.forward(inputs)

                error, hit = self.backward(labels)

                self.fit(learn_late / batch_size)

                epoch_error += error

                epoch_hit += hit

            error_list.append(epoch_error / image_no)

            acc_list.append(epoch_hit / image_no)

            print("loss:", error_list[-1], end=" ")

            print("acc:", acc_list[-1])





        return error_list

    

    def predict(self, inputs):

        image_no = inputs.shape[0]

        inputs_list = inputs.reshape([image_no,1,-1])

        pred_list = np.empty(image_no)

        for no, inputs in enumerate(inputs_list):

            self.forward(inputs)

            pro = self.layers_list[-1].neurons

            pred_list[no] = pro.argmax()

        return pred_list
import numpy as np

import pandas as pd

import csv

import matplotlib.pyplot as plt



np.random.seed(seed=32)

np.set_printoptions(threshold=10)



train= pd.read_csv("../input/digit-recognizer/train.csv")

test= pd.read_csv("../input/digit-recognizer/test.csv")

train_data = train.values

test_data = test.values



labels= train_data[:,0]

y_train = np.eye(10)[labels]

x_train = train_data[:,1:] / 255

x_test = test_data / 255



image_no = y_train.shape[0]

image_dim = x_train.shape[1]



inputs = Input(image_dim)

dence1 = Dense(10, image_dim)

relu1 = ReLU(10)

dence2 = Dense(10, 10)

softmax1 = Softmax_with_cetr(10)



model = Model([inputs, dence1, relu1, dence2, softmax1])

error = model.train(x_train, y_train, 0.2, 100, 32)



plt.plot(range(100), error)

plt.show()



pred_test = model.predict(x_test)



if(1==1):

    with open("predicted_data.csv", "w") as f:

        writer = csv.writer(f, lineterminator='\n')

        writer.writerow(["ImageId", "Label"])

        for pid, survived in enumerate(pred_test.astype('int')):

            writer.writerow([pid+1, survived])



print("finished")