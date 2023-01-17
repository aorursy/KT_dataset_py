import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
N, M = 50, 500
def mandel(X, Y):
    a, b = [0] * 2
    for i in range(N):
        a, b = a**2 - b**2 + X, 2 * a * b + Y
    return a**2 + b**2 < 4
x, y = [np.linspace(-2, 2, M)] * 2
X, Y = np.meshgrid(x, y)
plt.pcolor(X, Y, mandel(X, Y))
plt.show()
N=10000
def mandel2(X, Y):
    a, b = [0] * 2
    for i in range(N):
        a, b = a**2 - b**2 + X, 2 * a * b + Y
    return a**2 + b**2
x, y = [np.linspace(-2, 2, M)] * 2
X, Y = np.meshgrid(x, y)
plt.pcolor(X, Y, mandel2(X, Y),cmap='RdBu')
plt.show()
def logistic(a):
    x = [0.8]
    for i in range(400):
        x.append(a * x[-1] * (1 - x[-1]))
    return x[-100:]

for a in np.linspace(2.0, 4.0, 1000):
    x = logistic(a)
    plt.plot([a]*len(x), x, "c.", markersize=0.1)

plt.show()
from scipy.integrate import odeint, simps
def duffing(var, t, gamma, a, b, F0, omega, delta):
    x_dot = var[1]
    p_dot = -gamma * var[1] + 2 * a * var[0] - 4 * b * var[0]**3 + F0 * np.cos(omega * t + delta)
    return np.array([x_dot, p_dot])
F0, gamma, omega, delta = 10, 0.1, np.pi/3, 1.5*np.pi
a, b = 1/4, 1/2
var, var_lin = [[0, 1]] * 2
t = np.arange(0, 20000, 2*np.pi/omega)
t_lin = np.linspace(0, 100, 10000)
var = odeint(duffing, var, t, args=(gamma, a, b, F0, omega, delta))
var_lin = odeint(duffing, var_lin, t_lin, args=(gamma, a, b, F0, omega, delta))
x, p = var.T[0], var.T[1]
x_lin, p_lin = var_lin.T[0], var_lin.T[1]
plt.plot(x, p, ".", markersize=2)
plt.show()
plt.plot(t_lin, x_lin)
var_lin = odeint(duffing, [0.1, 1], t_lin, args=(gamma, a, b, F0, omega, delta))
x_lin, p_lin = var_lin.T[0], var_lin.T[1]
plt.plot(t_lin, x_lin)
plt.show()
def r(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1- y2)**2)**-3
def manybody(var, t, m1, m2, m3, G):
    x1 = var[2]
    px1 = -G * (m2 * r(var[0], var[1], var[4], var[5]) * (var[0] - var[4]) + m3 * r(var[0], var[1], var[8], var[9]) * (var[0] - var[8]))
    y1 = var[3]
    py1 = -G * (m2 * r(var[0], var[1], var[4], var[5]) * (var[1] - var[5]) + m3 * r(var[0], var[1], var[8], var[9]) * (var[1] - var[9]))
    x2 = var[6]
    px2 = -G * (-m1 * r(var[4], var[5], var[0], var[1]) * (var[0] - var[4]) + m3 * r(var[4], var[5], var[8], var[9]) * (var[4] - var[8]))
    y2 = var[7]
    py2 = -G * (-m1 * r(var[4], var[5], var[0], var[1]) * (var[1] - var[5]) + m3 * r(var[4], var[5], var[8], var[9]) * (var[5] - var[9]))
    x3 = var[10]
    px3 = -G * (-m1 * r(var[8], var[9], var[0], var[1]) * (var[0] - var[8]) - m2 * r(var[8], var[9], var[4], var[5]) * (var[4] - var[8]))
    y3 = var[11]
    py3 = -G * (-m1 * r(var[8], var[9], var[0], var[1]) * (var[1] - var[9]) - m2 * r(var[8], var[9], var[4], var[5]) * (var[5] - var[9]))
    return np.array([x1, y1, px1, py1, x2, y2, px2, py2, x3, y3, px3, py3])
m1, m2, m3 = 3, 4, 5
G = 1
var = np.array([0, 4, 0, 0, -3, 0, 0, 0, 0, 0, 0, 0])
t = np.linspace(0, 70, 3e7)
var = odeint(manybody, var, t, args=(m1, m2, m3, G), full_output=False)
plt.plot(var[:, 0][::1000], var[:, 1][::1000], label="1")
plt.plot(var[:, 4][::1000], var[:, 5][::1000], label="2")
plt.plot(var[:, 8][::1000], var[:, 9][::1000], label="3")
plt.show()