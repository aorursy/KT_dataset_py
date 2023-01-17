import numpy as np



def clip(value, lower, upper):

    if value < lower:

        return lower

    if value > upper:

        return upper

    return value



def default_ker(x, z):

    return x.dot(z.T)



def svm_smo(x, y, ker, C, max_iter, epsilon=1e-5):

    # initialization

    n, _ = x.shape

    alpha = np.zeros((n,))

        

    K = np.zeros((n, n))

    for i in range(n):

        for j in range(n):

            K[i, j] = ker(x[i], x[j])

    

    iter = 0

    while iter <= max_iter:

        

        for i in range(n):

            # randomly choose an index j, where j is not equal to i

            j = np.random.randint(low=0, high=n-1)

            while (i==j): j = np.random.randint(low=0, high=n-1)

            

            # update alpha_i

            eta = K[j, j] + K[i, i] - 2.0 * K[i, j]

            if np.abs(eta) < epsilon: continue # avoid numerical problem

            

            e_i = (K[:, i] * alpha * y).sum() - y[i]

            e_j = (K[:, j] * alpha * y).sum() - y[j]

            alpha_i = alpha[i] - y[i] * (e_i - e_j) / eta

            

            # clip alpha_i

            lower, upper = 0, C

            zeta = alpha[i] * y[i] + alpha[j] * y[j]

            if y[i] == y[j]:

                lower = max(lower, zeta / y[j] - C)

                upper = min(upper, zeta / y[j])

            else:

                lower = max(lower, -zeta / y[j])

                upper = min(upper, C - zeta / y[j])

                

            alpha_i = clip(alpha_i, lower, upper)

            alpha_j = (zeta - y[i] * alpha_i) / y[j]

            

            alpha[i], alpha[j] = alpha_i, alpha_j

        

        iter += 1

    

    # calculate b

    b = 0

    for i in range(n):

        if epsilon < alpha[i] < C - epsilon:

            b = y[i] - (y * alpha * K[:, i]).sum()

    

    def f(X): # predict the point X based on alpha and b

        results = []

        for k in range(X.shape[0]):

            result = b

            for i in range(n):

                result += y[i] * alpha[i] * ker(x[i], X[k])

            results.append(result)

        return np.array(results)

    

    return f, alpha, b
def data_visualization(x, y):

    import matplotlib.pyplot as plt

    category = {'+1': [], '-1': []}

    for point, label in zip(x, y):

        if label == 1.0: category['+1'].append(point)

        else: category['-1'].append(point)

    fig = plt.figure()

    ax = fig.add_subplot(111)



    for label, pts in category.items():

        pts = np.array(pts)

        ax.scatter(pts[:, 0], pts[:, 1], label=label)

    plt.show() 
import numpy as np



# random a dataset on 2D plane

def simple_synthetic_data(n, n0=5, n1=5): # n: number of points, n0 & n1: number of points on boundary

    # random a line on the plane

    w = np.random.rand(2) 

    w = w / np.sqrt(w.dot(w))

    

    # random n points 

    x = np.random.rand(n, 2) * 2 - 1

    d = (np.random.rand(n) + 1) * np.random.choice([-1,1],n,replace=True) # random distance from point to the decision line, d in [-2,-1] or [1,2]. d=-1 or d=1 indicate the boundary in svm

    d[:n0] = -1

    d[n0:n0+n1] = 1

    

    # shift x[i] to make the distance between x[i] and the decision become d[i]

    x = x - x.dot(w).reshape(-1,1) * w.reshape(1,2) + d.reshape(-1,1) * w.reshape(1,2)

    

    # create labels

    y = np.zeros(n)

    y[d < 0] = -1

    y[d >= 0] = 1

    return x, y



x, y = simple_synthetic_data(200)

data_visualization(x, y)
def spiral_data():

    data = np.loadtxt('/kaggle/input/svm-demo/spiral.txt')

    x = data[:,:2]

    y = data[:,2]

    return x, y



x, y = spiral_data()

data_visualization(x, y)
# load the synthetic data

x, y = simple_synthetic_data(100, n0=5, n1=5)



# run svm classifier

ker = default_ker

model, alpha, bias = svm_smo(x, y, ker, 1e10, 1000)



# visualize the result

import matplotlib.pyplot as plt

category = {'+1': [], '-1': []}

for point, label in zip(x, y):

    if label == 1.0: category['+1'].append(point)

    else: category['-1'].append(point)

fig = plt.figure()

ax = fig.add_subplot(111)



# plot points

for label, pts in category.items():

    pts = np.array(pts)

    ax.scatter(pts[:, 0], pts[:, 1], label=label)



# calculate weight

weight = 0

for i in range(alpha.shape[0]):

    weight += alpha[i] * y[i] * x[i]



# plot the model: wx+b

x1 = np.min(x[:, 0])

y1 = (-bias - weight[0] * x1) / weight[1]

x2 = np.max(x[:, 0])

y2 = (-bias - weight[0] * x2) / weight[1]

ax.plot([x1, x2], [y1, y2])



# plot the support vectors

for i, alpha_i in enumerate(alpha):

    if abs(alpha_i) > 1e-3: 

        ax.scatter([x[i, 0]], [x[i, 1]], s=150, c='none', alpha=0.7,

                   linewidth=1.5, edgecolor='#AB3319')

            

plt.show()
def poly_ker(d): # polynomial

    def ker(x, z): 

        return (x.dot(z.T)) ** d

    return ker



def cos_ker(x, z): # cosine similarity

    return x.dot(z.T) / np.sqrt(x.dot(x.T)) / np.sqrt(z.dot(z.T))

    

def rbf_ker(sigma): # rbf kernel

    def ker(x, z):

        return np.exp(-(x - z).dot((x - z).T) / (2.0 * sigma ** 2))

    return ker
import matplotlib.pyplot as plt

from matplotlib import cm



def plot(ax, model, x, title):

    y = model(x)

    y[y < 0], y[y >= 0] = -1, 1



    category = {'+1': [], '-1': []}

    for point, label in zip(x, y):

        if label == 1.0: category['+1'].append(point)

        else: category['-1'].append(point)

    for label, pts in category.items():

        pts = np.array(pts)

        ax.scatter(pts[:, 0], pts[:, 1], label=label)

    

    # plot boundary

    p = np.meshgrid(np.arange(-1.5, 1.5, 0.025), np.arange(-1.5, 1.5, 0.025))

    x = np.array([p[0].flatten(), p[1].flatten()]).T

    y = model(x)

    y[y < 0], y[y >= 0] = -1, 1

    y = np.reshape(y, p[0].shape)

    ax.contourf(p[0], p[1], y, cmap=plt.cm.coolwarm, alpha=0.4)

    

    # set title

    ax.set_title(title)



fig = plt.figure(figsize=(12,6))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)



x, y = spiral_data()



# plot points

model_default, _, _ = svm_smo(x, y, default_ker, 1e10, 200)

plot(ax1, model_default, x, 'Default SVM')



ker = rbf_ker(0.2)

# ker = poly_ker(5)

# ker = cos_ker

model_ker, _, _ = svm_smo(x, y, ker, 1e10, 200)

plot(ax2, model_ker, x, 'SVM + RBF')



plt.show()
from sklearn import svm

# x, y = simple_synthetic_data(50, 5, 5)

x, y = spiral_data()



model = svm.SVC(kernel='rbf', gamma=50, tol=1e-6)

model.fit(x, y)



fig = plt.figure(figsize=(6,6))

ax = fig.add_subplot(111)

plot(ax, model.predict, x, 'SVM + RBF')

plt.show()