# Importando pacotes
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib

# Configurando variável de tamanho de gráficos
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (6.0, 4.0)
# Gerar o conjunto de dados
np.random.seed(0) #Garante que todos nós tenhamos o mesmo resultado
X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
# Treina um classificador com os parâmetros padrão
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X, y)
# Função que vai nos ajudar a separar as regiões de cada classe no gráfico, conforme os parâmetros do classificador
# Não se preocupe em entender os códigos do matplotlib 
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

# Imprime a fronteira de decisão do classificador
plot_decision_boundary(lambda x: clf.predict(x))
plt.title("Logistic Regression")
num_examples = len(X) # tamanho do conjunto de treino - número de exemplos
nn_input_dim = 2 # dimensão da camada de entrada
nn_output_dim = 2 # dimensão da camada de saída

# Parâmetros do Gradiente Descendente.
epsilon = 0.01 # Taxa de aprendizado
reg_lambda = 0.01 # Regularização
# Avalia o custo médio total da nossa RN em relação aos dados.
def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss
# Função que usa os parâmetros da nossa RN para prever 0 ou 1.
# Note que a última linha apenas troca as probabilidades por um valor inteiro 0 ou 1.
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)
# Essa função aprende os parâmetros w1,w2,b1 e b2 da nossa rede. 
# A cada passada os parâmetros vão se ajustando para diminuir o erro
# - nn_hdim: número de nós da camada escondida
# - num_passes: Número de iterações do Gradiente Descendente
# - print_loss: Se verdadeiro imprime o custo a cada 1000 iterações

def build_model(nn_hdim, num_passes=20000, print_loss=False):
    
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = {}
    
    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
        
        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
              print("Custo após a iteração %i: %f" %(i, calculate_loss(model)))
    
    return model
# Constroi o modelo
model = build_model(3, print_loss=True)

# Imprime a fronteira de classificação
plot_decision_boundary(lambda x: predict(model, x))
plt.title("Fronteira de classificação para 3 neurônios")
plt.figure(figsize=(16, 32))
hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i+1)
    plt.title('Tamanho da camada escondida %d' % nn_hdim)
    model = build_model(nn_hdim)
    plot_decision_boundary(lambda x: predict(model, x))
plt.show()
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import np_utils
#Converte o y para o formato do keras (1 coluna para cada classe)
np.random.seed(222) #Garante que todos nós tenhamos o mesmo resultado
X, y = sklearn.datasets.make_moons(200, noise=0.30)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
y2 = np_utils.to_categorical(y)
model = Sequential()
model.add(Dense(3, input_dim=X.shape[1], activation='tanh', name="hidden"))
model.add(Dense(2, init='uniform', activation='softmax', name="output"))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)
model.summary()
model.fit(X, y2, nb_epoch=1000, shuffle=True, verbose=0)
plot_decision_boundary(lambda x: model.predict_classes(x))
