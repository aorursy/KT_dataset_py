# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import matplotlib.pyplot as plt

from matplotlib import animation, rc

from IPython.display import display, Math, HTML

import seaborn as sns



%matplotlib inline

figsize=(10,5)

def error(x,y, w0,w1):

    z = w0+w1*x

    return np.mean((y-z)**2)



def plot(x,y,w0=None, w1=None, title=""):

    plt.figure(figsize=figsize)

    plt.scatter(x, y,marker='o');

    plt.xlabel("X");

    plt.ylabel("Y");



    plt.grid();

    if w0!=None and w1!=None:

        z = w1*x+w0

        plt.plot(x, z,'g')

        title += f" MSE:{error(x,y,w0,w1):0.3f}"

        plt.vlines(x,y,z,colors='r',linestyles='dotted')

    plt.title(title)
x = np.linspace(1,10,20)

m = np.random.uniform(0.2,0.25, len(x))

c = np.random.uniform(-0.3,-0.1, len(x))

y = m*x + c



plot(x,y)
plot(x, y, 0.0,0.1)
w0, w1 = np.zeros(2)

n = len(y)

alpha = 0.01

tgt_mse = 0.01

hist = {"dJdw0":[],"dJdw1":[],"w0":[], "w1":[],"mse":[]}

for i in range(1000):

    z = w0+w1*x

    mse = error(x, y, w0, w1)

    dJdw0 = np.sum((z-y)/n)

    dJdw1 = np.sum(((z-y)*x)/n)

    w0_ = w0-alpha*dJdw0 

    w1_ = w1-alpha*dJdw1

    mse_ = error(x,y, w0_, w1_) 



    hist["dJdw0"].append(dJdw0)

    hist["dJdw1"].append(dJdw1)

    hist["w0"].append(w0)

    hist["w1"].append(w1)

    hist["mse"].append(mse)



    if mse<=tgt_mse:

        print(f"Converged after {i+1} steps")

        break

    else:

        w0,w1=(w0_,w1_)

        

plot(x,y, w0, w1)


rc('animation', html='html5')

c0 = sns.color_palette().as_hex()[0]

c1 = sns.color_palette().as_hex()[1]

c2 = sns.color_palette().as_hex()[2]



fig, ax = plt.subplots(figsize=figsize)

xmin,xmax,ymin,ymax = x.min(),x.max(),y.min(),y.max()



text = ax.set_title([])

ln, = plt.plot([], [],c=c2)

lc = plt.vlines([],[],[],colors=c1,linestyles='dotted')

# initialization function: plot the background of each frame

def init():   

    ax.set_xlim((xmin, xmax))

    ax.set_ylim((ymin, ymax))



    plt.scatter(x, y,marker='o',c=c0);

    plt.xlabel("X");

    plt.ylabel("Y");

    plt.grid();



    ln.set_data([],[])

    plt.close()

    return (ln,)



# animation function. This is called sequentially

def animate(i):

    w0 = hist['w0'][i]

    w1 = hist['w1'][i]

    mse = hist['mse'][i]

    #print(w0,w1)

    z = w1*x+w0

    ln.set_data(x,z)

    lc.set_segments([[[i[0],i[1]],[i[0],i[2]]] for i in zip(x,y,z)])

    text.set_text(f"Univariate Linear Regression\nIteration:{i}, MSE={mse:0.4f}, $\\alpha$={alpha:0.3f}")

    return (ln,)





# call the animator. blit=True means only re-draw the parts that 

# have changed.

anim = animation.FuncAnimation(fig, animate, 

                               init_func=init,

                               frames=20, 

                               interval=300, 

                               blit=True, 

                               repeat=True

                              )



#writer = animation.PillowWriter()  

#anim.save('lin_reg.gif', writer=writer)

anim
plt.figure(figsize=figsize)

plt.plot(hist["dJdw0"][:50], label=r"$\partial J/\partial w_0$")

plt.plot(hist["dJdw1"][:50], label=r"$\partial J/\partial w_1$")

plt.xlabel("Iteration")

plt.grid();

plt.legend();
plt.figure(figsize=figsize)

plt.plot(hist["w0"][:50], label=r"$w_0$")

plt.plot(hist["w1"][:50], label=r"$w_1$")

plt.xlabel("Iteration")

plt.grid()

plt.legend();
plt.figure(figsize=figsize)

plt.plot(hist["mse"][:50], label=r"$MSE$");

plt.xlabel("Iteration")

plt.grid()

plt.legend();
rsq = 1- hist["mse"]/ np.sum((y-y.mean())**2)

plt.figure(figsize=figsize)

plt.plot(rsq[:50], label=r"$R^2$");

plt.xlabel("Iteration")

plt.grid()

plt.legend();
xb = np.hstack((np.ones((x.shape[0],1)), x.reshape(x.shape[0],1)))

w = np.zeros(1)

z = np.linalg.inv(np.dot(xb.T, xb))

w = np.dot(z, np.dot(xb.T,y))

print(f"Analytical solution: w0={w[0]:0.4f}; w1={w[1]:0.4f}")

print(f"Numerical solution: w0={hist['w0'][-1]:0.4f}; w1={hist['w1'][-1]:0.4f}")