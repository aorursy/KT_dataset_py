import numpy as np

import matplotlib.pyplot as plt 

from sklearn.datasets import make_classification

%matplotlib inline

plt.style.use('default')
###########################################

def sigmoid(z):

    return 1/(1+np.exp(-z))

###########################################



# plot 

x=np.linspace(-8,8,1000)

y=sigmoid(x)

fig, ax = plt.subplots(figsize=(8,5))

ax.spines['left'].set_position(('data', 0.0))

ax.spines['bottom'].set_position(('data', 0.0))

ax.spines['right'].set_color('none')

ax.spines['top'].set_color('none')

plt.plot(x,y,linestyle='-')

plt.show()
import numpy as np

import matplotlib.pyplot as plt 

from sklearn.datasets import make_classification





def sigmoid(z):

    return 1/(1+np.exp(-z))





def train(X, y, num_iterations = 2000, learning_rate = 0.5, record = False):





    # Initialisation

    nbr_params , nbr_examples = X.shape

    w, b = np.full((nbr_params,1),0.01) , 0

    record = {"ws": [],"bs":[],"costs": []}





    # Loop over num_iterations

    for i in range(num_iterations+1):

        

        # Forward propagation

        A = sigmoid(np.dot(w.T,X)+b)                                    





        # Compute cost

        cost = np.sum(-y*np.log(A)-(1-y)*np.log(1-A))/nbr_examples                      

    

        # Backward propagation

        # Compute derivatives and update the weights and bias (Gradient Descent Step)

        dw = np.dot(X,(A-y).T)/nbr_examples

        db = np.sum(A-y)/nbr_examples

            

        w = w - learning_rate * dw

        b = b - learning_rate * db

        

        # Recording

        if record and i % 100 == 0:

            record["costs"].append(cost)

            record["ws"].append(w)

            record["bs"].append(b)





    return w,b,record





def predict(w, b, X):

    

    m = X.shape[1]

    Y_prediction = np.zeros((1,m))

    w = w.reshape(X.shape[0], 1)

    

    A = sigmoid(np.dot(w.T,X)+b)

    

    for i in range(A.shape[1]):

        

        if A[0,i] <= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1

    

    return Y_prediction



# Initialisation - generate random 2D samples

X,y0 = make_classification(n_samples=1000,n_features=2,n_informative=2,n_redundant=0)

y = y0.reshape(1,y0.shape[0])

X=X.T





# Compute W and b

w,b,record=train(X,y,num_iterations=2000,learning_rate=0.0001,record = True)



x1=np.linspace(-4,4,1000)

ws=record["ws"]

bs=record["bs"]

costs = record["costs"]





# Print graph

# plt.figure(figsize=(12,8))

# plt.scatter(X[0,:],X[1,:],c=y0.T,marker='o',s=25,edgecolor='k')

# for i in range(len(costs)):

#     y1=-(x1*ws[i][0,0]/ws[i][1,0]+bs[i]/ws[i][1,0])

#     plt.plot(x1,y1,linestyle='-')

       

# plt.show()

from matplotlib.animation import FuncAnimation

from IPython.display import HTML



fig, ax = plt.subplots(figsize=(12,8))

xdata, ydata = [], []

ln, = plt.plot([], [], 'r', animated=True)

f = list(range(len(costs)))

x1=np.linspace(-4,4,1000)



def init():

    ax.scatter(X[0,:],X[1,:],c=y0.T,marker='o',s=25,edgecolor='k')

    ln.set_data(xdata,ydata)

    return ln,



def update(i):

    y1=-(x1*ws[i][0,0]/ws[i][1,0]+bs[i]/ws[i][1,0])

    ln.set_data(x1, y1)

    return ln,

plt.close()



anim = FuncAnimation(fig, update, frames=f,

                    init_func=init, blit=True, interval = 100,repeat=True)



HTML(anim.to_jshtml())
