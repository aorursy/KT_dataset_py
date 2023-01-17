import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve



%config InlineBackend.figure_format = 'svg'
y = np.random.normal(0.5,0.3,10000)

y=y[y>0]

z = np.random.normal(1.2,0.3,10000)

z=z[z>0]

z1= 3*z

count, bins, ignored = plt.hist(z, 500, density=True, align='mid')

count, bins, ignored = plt.hist(y, 500, density=True, align='mid')

count, bins, ignored = plt.hist(z1, 500, density=True, align='mid')

plt.xlim((0,6))

plt.xlabel('Интенсивность сигнала')

plt.ylabel('Плотность вероятности')

plt.show()

def plot_roc_auc(y,z):

    Y = np.c_[np.r_[y,z],np.r_[np.zeros_like(y),np.ones_like(z)]]

    np.random.shuffle(Y)

    classes = Y[:,1].astype(np.int)

    fpr,tpr, thresholds = roc_curve(classes,Y[:,0], pos_label=1)

    

    plt.plot(fpr,tpr, linewidth=3, color='blue')

    plt.xlim((-0.001,1))

    plt.ylim((0,1.001))

    plt.ylabel('Чувствительность')

    plt.xlabel('Вероятность ложноположительного результата')

    plt.plot([0], [1.], marker='o', markersize=5, color="red")

    plt.show()
plot_roc_auc(y,z)
plot_roc_auc(y,z1)
np.sum(y<1)/y.shape[0]
np.sum(z>1)/z.shape[0]
np.sum(z1>2)/z1.shape[0]
np.sum(y<2)/y.shape[0]