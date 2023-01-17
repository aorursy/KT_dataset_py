import numpy as np

from scipy.integrate import odeint

from scipy.optimize import differential_evolution

from matplotlib import pyplot as plt
def force(omega, b):

    return b * omega ** 2
def quadrotor_model(X, t, m, g, K, I, Jr, l, b, d, O):

    x, y, z, vx, vy, vz, phi, theta, psi, vphi, vtheta, vpsi = X



    K1, K2, K3, K4, K5, K6 = K

    Ix, Iy, Iz = I



    forces = force(O, b)

    Or = O[0] - O[1] + O[2] - O[3]

    u1, u2, u3, u4 = [forces.sum(), -forces[1] + forces[3], -forces[0] + forces[2], d * (-forces[0] + forces[1] + forces[2] + forces[3]) / b]



    dxdt = vx

    dvxdt = (1 / m) * (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) * u1 - ((K1 * vx) / m)



    dydt = vy

    dvydt = (1 / m) * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) * u1 - ((K2 * vy) / m)



    dzdt = vz

    dvzdt = (1 / m) * (np.cos(phi) * np.cos(theta)) * u1 - g - ((K3 * vz) / m)



    dphidt = vphi

    dvphidt = (vtheta * vpsi * ((Iy - Iz) / Ix)) + ((Jr / Ix) * vtheta * Or) + ((l / Ix) * u2) - (((K4 * l) / Ix) * vphi)



    dthetadt = vtheta

    dvthetadt = (vpsi * vphi * ((Iz - Ix) / Iy)) - ((Jr / Iy) * vphi * Or) + ((l / Iy) * u3) - (((K5 * l) / Iy) * vtheta)



    dpsidt = vpsi

    dvpsidt = (vphi * vtheta * ((Ix - Iy) / Iz)) + ((l / Iz) * u4) - ((K6 / Iz) * vpsi)



    return dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt, dphidt, dthetadt, dpsidt, dvphidt, dvthetadt, dvpsidt
m = 2.0 # kg

Ix = Iy = 1.25 # Ns^2/rad

Iz = 2.2 # Ns^2/rad

K1 = K2 = K3 = 0.01 # Ns/m

K4 = K5 = K6 = 0.012 # Ns/m

l = 0.20 # m

Jr = 1 # Ns^2/rad

b = 2 # Ns^2

d = 5 # N ms^2

g = 9.8 # m/s^2
T = 1000

t = np.linspace(0, 40, T)
# Set-point

SP = np.zeros(len(t))

SP[:round(20 / 40 * 1000)] = 5

SP[round(20 / 40 * 1000):round(35 / 40 * 1000)] = 10

SP[round(35 / 40 * 1000):] = 0

# Kp, Ki, Kd

# K0 = [100, 10, 0]
def cost(x, t, T, SP):

    if np.any(x < 0):

        return np.inf



    # O = np.zeros((len(t), 4))

    O = np.array([0, 0, 0, 0], dtype=float)

    y0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # initial state

    e = np.zeros(len(t))

    I = 0.0

    D = 0.0

    Y = np.zeros((T, len(y0)))

    Y[0] = y0



    Kp, Ki, Kd = x



    for i in range(len(t) - 1):

        e[i] = SP[i] - y0[2]

        

        P = Kp * e[i]

        I += Ki * e[i] * (t[i + 1] - t[i])

        if i >= 1:

            D = Kd * ((e[i] - e[i - 1]) / (t[i] - t[i - 1]))



        O += P + I + D



        if np.any(O > 2):

            O[:] = 2

            I -= e[i] * (t[i + 1] - t[i])

        if np.any(O < 0):

            O[:] = 0

            I -= e[i] * (t[i + 1] - t[i])



        y_t = odeint(quadrotor_model, y0, [t[i], t[i + 1]], args=(m, g, (K1, K2, K3, K4, K5, K6), (Ix, Iy, Iz), Jr, l, b, d, O))

        y0 = y_t[1]

        Y[i + 1] = y0

    

    return (e ** 2).sum() / len(e)
res = differential_evolution(cost, ((1, 10000), (1, 10000), (1, 10000)), args=(t, T, SP), disp=True)
res
def test(x):

    O = np.array([0, 0, 0, 0], dtype=float)

    y0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # initial state

    e = np.zeros(len(t))

    I = 0.0

    D = 0.0

    Y = np.zeros((T, len(y0)))

    Y[0] = y0



    Kp, Ki, Kd = x



    for i in range(len(t) - 1):

        e[i] = SP[i] - y0[2]

        

        P = Kp * e[i]

        I += Ki * e[i] * (t[i + 1] - t[i])

        if i >= 1:

            D = Kd * ((e[i] - e[i - 1]) / (t[i] - t[i - 1]))



        O += P + I + D



        if np.any(O > 2):

            O[:] = 2

            I -= e[i] * (t[i + 1] - t[i])

        if np.any(O < 0):

            O[:] = 0

            I -= e[i] * (t[i + 1] - t[i])



        y_t = odeint(quadrotor_model, y0, [t[i], t[i + 1]], args=(m, g, (K1, K2, K3, K4, K5, K6), (Ix, Iy, Iz), Jr, l, b, d, O))

        y0 = y_t[1]

        Y[i + 1] = y0

        

    return Y, e
x, y, z, vx, vy, vz, phi, theta, psi, vphi, vtheta, vpsi = np.arange(12)
Y, e = test(res.x)

plt.figure(figsize=(20, 5))

plt.plot(t, Y[:, z], label="Process Variable")

plt.plot(t, SP, "--", label="Set-point")

plt.legend()

plt.show()