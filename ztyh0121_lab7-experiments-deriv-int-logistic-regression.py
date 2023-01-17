for i in range(10):
    print(i)
print(i)
for i in range(10):
    print(i)
for i in range(1,121):
    if 121%i==0:
        print(i)
def f(x):
    return x**3+5
for x in range(1,5):
    d=10**(-x)
    print((f(3+d)-f(3))/d)
def df(x):
    d=10**(-5)
    return (f(x+d)-f(x))/d
df(2)
from sympy import *
x = Symbol('x')
y = x**3 + 5
y.diff(x)
y.diff(x,2)
y.diff(x,2).evalf(subs={x: 1})
y.diff(x,x,x,x)
y.diff(x,4)
y=exp(-x)
y.diff(x)
y=sin(-x)
y.diff(x)
y=sin(-x)/x
y.diff(x)
y=sin(-x)/exp(x**3)
y.diff(x)
y=sin(-x)*x**2/log(x**3)
y.diff(x)
diff(y,x)
y = x**3 + 5
integrate(y,x)
type((0,1,2))
integrate(y,(x,0,1))
u = Symbol('u')
v = Symbol('v')
z=u*v+u+v
diff(z,u)
diff(z,v)
integrate(z,u)
21/4
np.linspace(0,1,1000)
import numpy as np
for n in range(2,20,2):
    t=np.linspace(0,1,n*1000)
    t=t[1:]
    d=1/1000/n
    print(sum(d*f(t)))
def intf(tl,tu):
    n=100000
    t=np.linspace(tl,tu,n)
    t=t[1:]
    d=(tu-tl)/n
    print(sum(d*f(t))) 
intf(3,5)
y
integrate(y,(x,3,5))
y = x**3 + log(x)
integrate(y,x)
integrate(y,(x,5,10))
y = x**3 + log(x)/sin(x**2)
integrate(y,x)
integrate(y,(x,1,2))
import pandas as pd
import matplotlib.pyplot  as plt
from scipy import optimize
plt.style.use('ggplot')
def make_data(N, draw_plot=True, is_confused=False, confuse_bin=50):
    np.random.seed(1)
    feature = np.random.randn(N, 2)
    df = pd.DataFrame(feature, columns=['x', 'y'])
    df['c'] = df.apply(lambda row : 1 if (5*row.x + 3*row.y - 1)>0 else 0,  axis=1)
    if is_confused:
        def get_model_confused(data):
            c = 1-data.c if (data.name % confuse_bin) == 0 else data.c 
            return c
        df['c'] = df.apply(get_model_confused, axis=1)
    if draw_plot:
        plt.scatter(x=df.x, y=df.y, c=df.c, alpha=0.6,s=10,cmap='winter')
        plt.xlim([df.x.min() -0.1, df.x.max() +0.1])
        plt.ylim([df.y.min() -0.1, df.y.max() +0.1])
    return df
df = make_data(1000)
df.c.head()
df.x.head()
df.y.head()
df_data = make_data(1000, is_confused=True, confuse_bin=10)
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))
def get_prob(x, y, weight_vector):
    feature_vector =  np.array([x, y, 1])
    z = np.inner(feature_vector, weight_vector)
    return sigmoid(z)
def define_likelihood(weight_vector, args):
    likelihood = 0
    df_data = args
    for x, y, c in zip(df_data.x, df_data.y, df_data.c):
        prob = get_prob(x, y, weight_vector) # this is the p
        i_likelihood = np.log(prob) if c==1 else np.log(1.0-prob)
        likelihood = likelihood - i_likelihood
    return likelihood
def estimate_weight(df_data, initial_param):
    parameter = optimize.minimize(define_likelihood,
                                  initial_param, 
                                  args=df_data,
                                  method='Nelder-Mead')
    return parameter.x/np.linalg.norm(parameter.x)
def draw_split_line(weight_vector):
    a,b,c = weight_vector
    x = np.array(range(-10,10,1))
    y = (a * x + c)/-b
    plt.plot(x,y, alpha=1)    
weight_vector = np.random.rand(3)
weight_vector
import warnings
warnings.filterwarnings("ignore")
df_data = make_data(1000, is_confused=True, confuse_bin=10)
weight_vector = np.random.rand(3)
draw_split_line(weight_vector)
weight_vector = estimate_weight(df_data, weight_vector)
draw_split_line(weight_vector)
count = 0
while (count < 9):
   print('The count is:', count)
   count = count + 1
print("Good bye!")