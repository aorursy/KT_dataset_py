import numpy as np

import matplotlib.pyplot as plt



# Define the mean and standard deviation

mu, sigma = 0, 0.1 

# Draw 500 samples from this Gaussian distribution

s = np.random.normal(mu, sigma, 500)

# Plot a histogram of the samples

n, bins, patches = plt.hist(s, 50, normed=1, facecolor='blue', alpha=0.5)

# Plot the associated distribution

plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi))*np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')

# Plot the mean of distribution

plt.plot((mu, mu), (0., 5.), '--r')

# Tidy the plot

plt.xlabel('$x$')

plt.ylabel('$P(x)$')

plt.axis((-0.3,0.3,0.,5.))

plt.show()
import seaborn as sns

sns.set(style="white", color_codes=True)



# Define the mean and standard deviation

mu1, sigma1 = 0., 0.1

mu2, sigma2 = 0., 0.2

# Draw 500 samples from each Gaussian distribution

s1 = np.random.normal(mu1, sigma1, 500)

s2 = np.random.normal(mu2, sigma2, 500)



# Scatter plot the randomly drawn variables

g = sns.JointGrid(s1, s2)

g.plot_marginals(sns.distplot, kde=False, color = "g")

g.plot_joint(plt.scatter, color="g", s=40, edgecolor="white")
# Define the mean and standard deviation

mu1, sigma1, cov12 = 0., 0.5, 0.2

mu2, sigma2, cov21 = 0., 0.5, 0.2

# Draw 500 samples from each Gaussian distribution

mu = [mu1,mu2]

covMat = [[sigma1**2, cov12],[cov21,sigma2**2]]

s = np.random.multivariate_normal(mu, covMat, 500)

# Scatter plot the randomly drawn variables

g = sns.JointGrid(s[:,0],s[:,1],xlim=(-2.5,2.5), ylim=(-2.5,2.5))

g.plot_marginals(sns.distplot, kde=False, color = "b")

g.plot_joint(plt.scatter, color="b", s=40, edgecolor="white")

#g.ax_marg_y.set_xlim(0,5)