import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4])

plt.ylabel('some numbers')

plt.show()
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')

plt.axis([0, 6, 0, 20])

plt.show()
import numpy as np



# evenly sampled time at 200ms intervals

t = np.arange(0., 5., 0.2)



# red dashes, blue squares and green triangles

plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

plt.show()
data = {'a': np.arange(50),

        'c': np.random.randint(0, 50, 50),

        'd': np.random.randn(50)}

data['b'] = data['a'] + 10 * np.random.randn(50)

data['d'] = np.abs(data['d']) * 100



plt.scatter('a', 'b', c='c', s='d', data=data)

plt.xlabel('entry a')

plt.ylabel('entry b')

plt.show()
names = ['group_a', 'group_b', 'group_c']

values = [1, 10, 100]



plt.figure(figsize=(9, 3))



plt.subplot(131)

plt.bar(names, values)

plt.subplot(132)

plt.scatter(names, values)

plt.subplot(133)

plt.plot(names, values)

plt.suptitle('Categorical Plotting')

plt.show()
mu, sigma = 100, 15

x = mu + sigma * np.random.randn(10000)



# the histogram of the data

n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)





plt.xlabel('Smarts')

plt.ylabel('Probability')

plt.title('Histogram of IQ')

plt.text(60, .025, r'$\mu=100,\ \sigma=15$')

plt.axis([40, 160, 0, 0.03])

plt.grid(True)

plt.show()
from matplotlib.ticker import NullFormatter  # useful for `logit` scale



# Fixing random state for reproducibility

np.random.seed(19680801)



# make up some data in the interval ]0, 1[

y = np.random.normal(loc=0.5, scale=0.4, size=1000)

y = y[(y > 0) & (y < 1)]

y.sort()

x = np.arange(len(y))



# plot with various axes scales

plt.figure()



# linear

plt.subplot(221)

plt.plot(x, y)

plt.yscale('linear')

plt.title('linear')

plt.grid(True)





# log

plt.subplot(222)

plt.plot(x, y)

plt.yscale('log')

plt.title('log')

plt.grid(True)





# symmetric log

plt.subplot(223)

plt.plot(x, y - y.mean())

plt.yscale('symlog', linthreshy=0.01)

plt.title('symlog')

plt.grid(True)



# logit

plt.subplot(224)

plt.plot(x, y)

plt.yscale('logit')

plt.title('logit')

plt.grid(True)

# Format the minor tick labels of the y-axis into empty strings with

# `NullFormatter`, to avoid cumbering the axis with too many labels.

plt.gca().yaxis.set_minor_formatter(NullFormatter())

# Adjust the subplot layout, because the logit one may take more space

# than usual, due to y-tick labels like "1 - 10^{-3}"

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,

                    wspace=0.35)



plt.show()