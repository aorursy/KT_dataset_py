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


# change directory to the dataset where our

# custom scripts are found

os.chdir("../input/doubleq2")

print(os.getcwd())

# read in custom modules 

from Run import *



# reset our working directory

os.chdir("/kaggle/working/")



import matplotlib.pyplot as plt
def myfunction(gamma, omega, noSteps, width):

    

    noRuns = 20

    exp = "onebyone"

    # width of the uniform distribution of the reward function

#     width = width

    p = (4.0 * gamma - 1.0) / (3.0 * gamma)

    World = Tree(p, width)

    return runExperiment(World, gamma, noSteps, noRuns, exp, omega)





def blockwiseBound(gamma, omega, noSteps, width):

    beta = (1-gamma)/4

    Vmax = 2*width/(1-gamma)

    

    # tune tau1 and c to adjust the blocks

    c = 5000

    tau1 = 20000

    

    tau = [0, tau1]

    while tau[-1] < noSteps:

        tau.append(int(np.ceil(tau[-1] + c * (tau[-1])**omega)))

    

    G = np.ones(noSteps) * Vmax

    D = np.ones(noSteps) * Vmax * 2 * gamma / (1-gamma)

    

    for i in range(len(tau)):

        if 0 < i < len(tau)-2:

            G[tau[i]:tau[i+1]] = G[tau[i]:tau[i+1]] * beta**i

            D[tau[i]:tau[i+1]] = D[tau[i]:tau[i+1]] * beta**i

        elif i == len(tau)-2:

            G[tau[i]:] = G[tau[i]:] * beta**i

            D[tau[i]:] = D[tau[i]:] * beta**i

    return [G, D]





noSteps = 1*(10**5)

indices = np.arange(0, noSteps, dtype=int)

width = 40



gammas = [0.7]

omega = 0.85

figureIndex = 0;

for gamma in gammas:

    

    [learning_error, QAQB_error] = myfunction(gamma, omega, noSteps, width)

#     print([len(learning_error), len(QAQB_error), len(learning_error[0]), len(QAQB_error[0])])

    learning_error_mean = np.array([np.mean(learning_error[i]) for i in range(len(learning_error))])

    learning_error_std = np.array([np.std(learning_error[i]) for i in range(len(learning_error))])

    QAQB_error_mean = np.array([np.mean(QAQB_error[i]) for i in range(len(QAQB_error))])

    QAQB_error_std = np.array([np.std(QAQB_error[i]) for i in range(len(QAQB_error))])

    

    [G,D] = blockwiseBound(gamma, omega, noSteps, width)

    

    plt.figure()

    plt.plot(indices, learning_error_mean, linewidth=2, label = r'$\Vert Q^{A}-Q^{*}\Vert$')

    plt.fill_between(indices, learning_error_mean-learning_error_std, learning_error_mean+learning_error_std, alpha = 0.4)

    plt.plot(indices, QAQB_error_mean, linewidth=2, label = r'$\Vert Q^{A}-Q^{B}\Vert$')

    plt.fill_between(indices, QAQB_error_mean-QAQB_error_std, QAQB_error_mean+QAQB_error_std, alpha = 0.4)

    

    plt.plot(indices, G, linestyle="--", linewidth=2, label = r'$G_q$')

    plt.plot(indices, D, linestyle="--", linewidth=2, label = r'$D_k$')

    plt.xscale('log')

    plt.yscale('log')

    

    plt.legend()

    

    plt.savefig('example{}.pdf'.format(figureIndex))

    plt.show()

    figureIndex += 1;
# generate links for downloading figure, 

# can also manually download them from the right column of this page under Data/output/



# This is where the figures are saved

os.chdir("/kaggle/working")

print(os.getcwd())



from IPython.display import FileLink, FileLinks

FileLinks('.')


