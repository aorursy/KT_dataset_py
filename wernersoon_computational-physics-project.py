import numpy as np

import matplotlib.pyplot as plt



#variables kept constant

g = 9.81 # magnitude of acceleration due to gravity in m/s^2

T = 298.15 #temperature in K 

p = 997.0479 #fluid density of water in kg/m^3

m = 1 #mass of object in kg

r = 1 # radius os sphere in m 



V = (4/3)*np.pi*(r**3)



def Vol(x): #Volume of submerged object for sphere

    return 4/3 * np.pi * (r**3)



# first_a = (p * Vol(-25) * g - m * g)/ m



x_0 = -25

vec = np.array([x_0, 0]) # x = -25, v = 0, a = f_a(-25)





def f_a(d):

    F = m * g # Gravitional force

    B = p * Vol(vec[0]) * g # Buoyancy force

    a = (B - F)/ m # Ok

    return a



def derivs(vec,t):

    x, v = vec

    v_dot = f_a(x)

    x_dot = v

    #x_dot = v_dot * t # The derivative of displacement with respect to time is velocity

    #x = x_dot * t + v_dot*(t**2)/2

    

    # Werner: If x, v, a is in vec, then you need to find x_dot, v_dot, a_dot

    # respectively, not x, x_dot and v_dot!

    return np.array([x_dot,v_dot])



def rk4(vec,t,dt):

    f0 = derivs(vec,t)

    f1 = derivs(vec + f0*dt/2,t + dt/2)

    f2 = derivs(vec + f1*dt/2,t + dt/2)

    f3 = derivs(vec + f2*dt,t + dt)

    vec_next = vec+(f0+2*f1+2*f2+f3)*dt/6

    return vec_next



ti = 0

t = ti

tf = 10

dt = 0.001



def analytical(t):

    a = p*V*g - m*g

    v = a*t

    x = x_0 + 0.5*a*t*t

    return np.array([x, v, a])



t_list = []

x_list = []

v_list = []

t_analytical = []

x_analytical = []

v_analytical = []



print(vec,t)

t_list.append(t)

x_list.append(vec[0])

v_list.append(vec[1])



analytical_sol = analytical(t)

t_analytical.append(t)

x_analytical.append(analytical_sol[0])

v_analytical.append(analytical_sol[1])

while vec[0]<0:

    vec = rk4(vec,t,dt)

    t += dt

    

    print(vec,t)

    t_list.append(t)

    x_list.append(vec[0])

    v_list.append(vec[1])

    

    analytical_sol = analytical(t)

    t_analytical.append(t)

    x_analytical.append(analytical_sol[0])

    v_analytical.append(analytical_sol[1])

    

    if tf - t <= 1e-9:

        break



# Prevent text from overlapping



print("Upwards is defined as positive")



#graph for velocity over time

fig = plt.figure()

plt.xlabel('Time /s') #

plt.ylabel('Velocity of object /ms^-1')

plt.title('Graph of velocity of object against time')

plt.plot(t_list, v_list, label = "Numerical")

plt.plot(t_analytical, v_analytical, label = "Analytical")

plt.legend()

plt.show()



#graph for depth over time

fig = plt.figure()

plt.xlabel('Time /s') #

plt.ylabel('Depth of object /m')

plt.title('Graph of depth of object against time')

plt.plot(t_list, x_list, label = "Numerical")

plt.plot(t_analytical, x_analytical, label = "Analytical")

plt.legend()

plt.show()
