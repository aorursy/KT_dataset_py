import numpy as np

import matplotlib.pyplot as plt





def draw(x1,x2):

    ln=plt.plot(x1,x2)



def sigmoid(score):

    return 1/(1+np.exp(-score))



def calculate_error(line_parameters, points , y):

    n=points.shape[0]

    p= sigmoid(points*line_parameters)

    cross_entropy=-(1/n)*(np.log(p).T*y + np.log(1-p).T*(1-y))

    return cross_entropy





def gradient_descent(line_parameters, points, y , alpha):

    n=points.shape[0]

    for i in range(2000):

      p=sigmoid(points*line_parameters)

      gradient= points.T*(p-y)*(alpha/n)

      line_parameters = line_parameters - gradient

        

      w1=line_parameters.item(0)

      w2=line_parameters.item(1)

      b=line_parameters.item(2)

        

      x1=np.array([points[:,0].min(), points[:,0].max()])

      x2= -b/w2 + (x1*(-w1/w2))

    draw(x1,x2) 



      

n_pts=100

np.random.seed(0)

bias= np.ones(n_pts)

top_region=np.array([np.random.normal(10,2,n_pts), np.random.normal(12,2,n_pts), bias]).T

bottom_region= np.array([np.random.normal(5,2, n_pts), np.random.normal(6,2, n_pts), bias]).T

all_points=np.vstack((top_region, bottom_region))

     

line_parameters = np.matrix([np.zeros(3)]).T

y=np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1)

     

_, ax= plt.subplots(figsize=(4,4))

ax.scatter(top_region[:,0], top_region[:,1], color='r')

ax.scatter(bottom_region[:,0], bottom_region[:,1], color='b')

gradient_descent(line_parameters, all_points, y , 0.06)

plt.show()
import numpy as np

import scipy.linalg

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt



# some 3-dim points

mean = np.array([0.0,0.0,0.0])

cov = np.array([[1.0,-0.5,0.8], [-0.5,1.1,0.0], [0.8,0.0,1.0]])

data = np.random.multivariate_normal(mean, cov, 50)



# regular grid covering the domain of the data

X,Y = np.meshgrid(np.arange(-3.0, 3.0, 0.5), np.arange(-3.0, 3.0, 0.5))

XX = X.flatten()

YY = Y.flatten()



order = 1    # 1: linear, 2: quadratic

if order == 1:

    # best-fit linear plane

    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]

    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients

    

    # evaluate it on grid

    Z = C[0]*X + C[1]*Y + C[2]

    

    # or expressed using matrix/vector product

    #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)



elif order == 2:

    # best-fit quadratic curve

    A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]

    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])

    

    # evaluate it on a grid

    Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)



# plot points and fitted surface

fig = plt.figure()

ax = fig.gca(projection='3d')

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)

ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50)

plt.xlabel('X')

plt.ylabel('Y')

ax.set_zlabel('Z')

ax.axis('equal')

ax.axis('tight')

plt.show()