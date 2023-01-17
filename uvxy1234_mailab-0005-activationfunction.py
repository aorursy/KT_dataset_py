import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

%pylab inline

pylab.rcParams['figure.figsize'] = (10, 6) # adjust the size of figures

def plot_all(x, function, df, mode_f, mode_df):

    _, ax = plt.subplots(1, 2)

    ax[0].title.set_text(function.__name__)

    ax[1].title.set_text(df.__name__)

    function = function(x)

    df = df(x)

    #plot f

    if mode_f == 'center':

        ax[0].spines['left'].set_position('center')

        ax[0].spines['bottom'].set_position('center')

        ax[0].spines['right'].set_position('center')

        ax[0].spines['top'].set_position('center')

        ax[0].plot(x, function)

        

    elif mode_f == 'bottom':

        ax[0].spines['left'].set_position('center')

        ax[0].spines['right'].set_position('center')

        ax[0].spines['top'].set_color('none')

        ax[0].xaxis.set_ticks_position('bottom')

        ax[0].plot(x, function)

        

    

    #plot df

    if mode_df == 'center':

        ax[1].spines['left'].set_position('center')

        ax[1].spines['bottom'].set_position('center')

        ax[1].spines['right'].set_position('center')

        ax[1].spines['top'].set_position('center')

        ax[1].plot(x, df)

        

    elif mode_df == 'bottom':

        ax[1].spines['left'].set_position('center')

        ax[1].spines['right'].set_position('center')

        ax[1].spines['top'].set_color('none')

        ax[1].xaxis.set_ticks_position('bottom')

        ax[1].plot(x, df)

    
### Sigmoid

def sigmoid(x):

    return 1 / (1 + np.exp(-x))



def d_sigmoid(x):

    return sigmoid(x) * (1 - sigmoid(x))



if __name__ == '__main__':

    #define x domain

    x = np.linspace(-10., 10., 50)

    #plot figure

    plot_all(x, sigmoid, d_sigmoid, 'center', 'bottom')
### Hyperbolic Tangent (tanh)

def tanh(x):

    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))



def d_tanh(x):

    return 1 - np.square(tanh(x))



if __name__ == '__main__':

    #define x domain

    x = np.linspace(-10., 10., 50)

    #plot figure

    plot_all(x, tanh, d_tanh, 'center', 'bottom')
### Rectified Linear Unit (ReLU)

def ReLU(x):

    return x * (x > 0)  #another expression: np.where(x > 0, x, 0)



def d_ReLU(x):

    return 1 * (x > 0)  #another expression: np.where(x > 0, 1, 0)



if __name__ == '__main__':

    #define x domain

    x = np.linspace(-10., 10., 50)

    #plot figure

    plot_all(x, ReLU, d_ReLU, 'bottom', 'bottom')
### Leaky ReLU

def Leaky_ReLU(x):

    return 0.01 * x * (x < 0) + x * (x > 0)  



def d_Leaky_ReLU(x):

    return 0.01 * (x < 0) + 1 * (x > 0)  



if __name__ == '__main__':

    #define x domain

    x = np.linspace(-10., 10., 50)

    #plot figure

    plot_all(x, Leaky_ReLU, d_Leaky_ReLU, 'bottom', 'bottom')
### Exponential ReLU (ELU)

def ELU(x, alpha=1):

    return alpha * (np.exp(x) - 1) * (x < 0) + x * (x > 0) 



def d_ELU(x, alpha=1):

    return alpha * (np.exp(x)) * (x < 0) + 1 * (x > 0)  



if __name__ == '__main__':

    #define x domain

    x = np.linspace(-10., 10., 50)

    #plot figure

    plot_all(x, ELU, d_ELU, 'bottom', 'bottom')