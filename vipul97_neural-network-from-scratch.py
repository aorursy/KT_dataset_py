import pandas as pd



creditcard = pd.read_csv("../input/creditcardfraud/creditcard.csv")
X_train = creditcard.drop("Class", axis=1).values[:int(0.8 * len(creditcard))]

y_train = creditcard["Class"].values[:int(0.8 * len(creditcard))]

X_test = creditcard.drop("Class", axis=1).values[int(0.8 * len(creditcard)):]

y_test = creditcard["Class"].values[int(0.8 * len(creditcard)):]
import numpy as np





class StandardScaler:

    def __init__(self):

        self.u = 0

        self.s = 0

    

    def fit(self, X):

        self.u = np.mean(X, axis=0)

        self.s = np.std(X, axis=0)

    

    def transform(self, X):

        return (X - self.u) / self.s
scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
import numpy as np





class NeuralNetwork:

    def __init__(self, layer_dims=np.array([1]), learning_rate=1.0, num_iterations=1000):

        self.layer_dims = layer_dims

        self.learning_rate = learning_rate

        self.num_iterations = num_iterations



    def feedforward(self, X):

        def sigmoid(z):

            return 1 / (1 + np.exp(-z)), z



        def relu(z):

            return np.maximum(0, z), z



        def linear_forward(a, W, b):

            return np.dot(W, a) + b, (a, W, b)



        def linear_activation_forward(a_prev, W, b, activation):

            z, linear_cache = linear_forward(a_prev, W, b)



            if activation == "sigmoid":

                a, activation_cache = sigmoid(z)



                return a, (linear_cache, activation_cache)

            elif activation == "relu":

                a, activation_cache = relu(z)



                return a, (linear_cache, activation_cache)



        caches = []

        L = len(self.parameters) // 2



        a = X

        for l in range(1, L):

            a_prev = a

            a, cache = linear_activation_forward(a_prev, self.parameters['W' + str(l)], self.parameters['b' + str(l)],

                                                 "relu")

            caches.append(cache)



        aL, cache = linear_activation_forward(a, self.parameters['W' + str(L)], self.parameters['b' + str(L)],

                                              "sigmoid")

        caches.append(cache)



        return aL, caches



    def fit(self, X, y):

        def initialize_parameters():

            parameters = {}

            L = len(self.layer_dims)



            for l in range(1, L):

                parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * 0.01

                parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))



            return parameters



        def compute_cost(aL):

            return - np.sum(np.multiply(np.log(aL), self.y) + np.multiply(np.log(1 - aL), 1 - self.y)) / self.m



        def backprop(aL, caches):

            def sigmoid_backward(da, cache):

                s = 1 / (1 + np.exp(-cache))



                return da * s * (1 - s)



            def relu_backward(da, cache):

                dz = np.array(da, copy=True)

                dz[cache <= 0] = 0



                return dz



            def linear_backward(dz, cache):

                a_prev, W, b = cache



                return np.dot(W.T, dz), 1 / self.m * (np.dot(dz, a_prev.T)), 1 / self.m * (

                    np.sum(dz, axis=1, keepdims=True))



            def linear_activation_backward(da, cache, activation):

                linear_cache, activation_cache = cache



                if activation == "relu":

                    return linear_backward(relu_backward(da, activation_cache), linear_cache)

                elif activation == "sigmoid":

                    return linear_backward(sigmoid_backward(da, activation_cache), linear_cache)



            grads = {}

            L = len(caches)

            daL = - (np.divide(self.y, aL) - np.divide(1 - self.y, 1 - aL))



            current_cache = caches[L - 1]

            grads["da" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(daL,

                                                                                                              current_cache,

                                                                                                              "sigmoid")

            for l in reversed(range(L - 1)):

                current_cache = caches[l]

                grads["da" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = linear_activation_backward(

                    grads["da" + str(l + 1)], current_cache, "relu")



            return grads



        def update_parameters(grads):

            L = len(self.parameters) // 2



            for l in range(L):

                self.parameters["W" + str(l + 1)] -= self.learning_rate * grads["dW" + str(l + 1)]

                self.parameters["b" + str(l + 1)] -= self.learning_rate * grads["db" + str(l + 1)]



        self.X = X.T

        self.m = self.X.shape[1]

        self.y = y.reshape(1, self.m)

        self.parameters = initialize_parameters()

        self.costs = []



        for i in range(self.num_iterations):

            aL, caches = self.feedforward(self.X)

            cost = compute_cost(aL)

            grads = backprop(aL, caches)

            update_parameters(grads)

            self.costs.append(cost)



    def predict(self, X):

        aL, caches = self.feedforward(X.T)



        return aL > 0.5
nn_clf = NeuralNetwork(layer_dims=np.array([30, 1]), learning_rate=1, num_iterations=100)

nn_clf.fit(X_train, y_train)

y_pred = nn_clf.predict(X_test)
accuracy_score = np.mean(y_pred == y_test)

accuracy_score
import matplotlib.pyplot as plt



plt.plot(nn_clf.costs)

plt.title('Gradient Descent')

plt.xlabel('Number of iterations')

plt.ylabel('Cost J')

plt.show()