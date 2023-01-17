import numpy as np

from numpy import pi, sin, cos

import matplotlib.pyplot as pp



%matplotlib inline
def integrate(func, x, dx):

    return sum(func(x)) * dx





def A0(func, x, dx, L):

    return 2.0 / L * integrate(func, x, dx)





def A(func, k, x, dx, L):

    return 2.0 / L * sum(func(x) * cos(2.0 * pi * k / L * x)) * dx





def B(func, k, x, dx, L):

    return 2.0 / L * sum(func(x) * sin(2.0 * pi * k / L * x)) * dx





def backFT(a0, a, b, L):

    def inner(x):

        a_sum = .0

        b_sum = .0

        for i, v in enumerate(a):

            a_sum += a[i] * cos(2.0 * pi / L * (i + 1) * x)

            b_sum += b[i] * sin(2.0 * pi / L * (i + 1) * x)

        return a0 / 2 + a_sum + b_sum

    return inner
def f(x):

    return sin(0.5 * x) + cos(2.0 * x) - 1.5 * cos(4.0 * x)



f = np.vectorize(f)  
dx = 0.01

x = np.arange(-pi, pi, dx)

L = 2.0 * pi

N = 5



def plot_ft_and_spectrum(func, x, dx, L, N):

    a = []

    b = []

    for i in range(1, N):

        a.append(A(func, i, x, dx, L))

        b.append(B(func, i, x, dx, L))



    back_ft = backFT(A0(func, x, dx, L), a, b, L)



    error = sum((back_ft(x) - func(x)) * (back_ft(x) - func(x))) / len(x)

    print(f"Error: ${error}")



    pp.plot(x, func(x))

    pp.plot(x, back_ft(x))

    pp.grid(True)

    pp.show()



    pp.plot(range(1, N), a, 'rx')

    pp.plot(range(1, N), b, 'bo')

    pp.vlines(range(1, N), [0], a, colors='r')

    pp.vlines(range(1, N), [0], b, colors='b')

    pp.grid(True)

    pp.show()

    

plot_ft_and_spectrum(f, x, dx, L, N)
def step(x):

    return .0 if x < 0 else 1.



step = np.vectorize(step)



dx = 0.01

x = np.arange(-5., 5., dx)

L = 2.0 * 5



plot_ft_and_spectrum(step, x, dx, L, N=20)
def step_up_down(x):

    return step(x + 1) - step(x - 1)



step_up_down = np.vectorize(step_up_down)



plot_ft_and_spectrum(step_up_down, x, dx, L, N=20)