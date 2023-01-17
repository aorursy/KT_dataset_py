import numpy as np

import matplotlib.pyplot as plt



### Default values

order0= 2                   #Order of the ODE, in our case order =2

Fp0= 80.0

w0= 2.0*np.pi

wp0= np.pi

t_end0= 100.0                

dt0 = 0.001

Xinit0= [0, 0]              #[V,X] at t=0

beta0= 0.



# In this code, V and X will be saved to an array X = [V, X]

# Where V=X[0] and X=X[1]

# The purpose of this is to let the code be expandable to greater order ODEs.

# For example for 4th order ODE, X will be [X''',X'', X', X]



### Functions





def RK4(Fp=Fp0,w=w0,wp=wp0,t_end=t_end0,dt=dt0,Xinit=Xinit0,order=order0, beta=beta0):

    t= np.arange(0.,t_end+dt,dt)

    X= np.zeros((t.shape[0],order))

    X[0]= Xinit

    for i in range(t.shape[0]-1):

        X[i+1]= RK4_h(Fp,w,wp,dt,t[i],X[i],order,beta)

    return X.transpose(),t





def RK4_h(Fp,w,wp,dt,t,X,order,beta):

    h1= np.array([dt*g(Fp,w,wp, beta, t, X, i) for i in range(order)])

    h2= np.array([dt*g(Fp,w,wp, beta, t+dt/2, X+h1/2, i) for i in range(order)])

    h3= np.array([dt*g(Fp,w,wp, beta, t+dt/2, X+h2/2, i) for i in range(order)])

    h4= np.array([dt*g(Fp,w,wp, beta, t+dt, X+h3, i) for i in range(order)])

    return X+ 1/6*(h1+2*h2+2*h3+h4)



def g(Fp,w,wp,beta, t, X, _order):          #Gradient function: g(X[n])=d(X[n])/dt

    if _order==0:

        return -(wp**2)*(1+beta*(X[1]**2))*X[1]+Fp*np.sin(w*t)               #X[1]=X, X[0]=V

    else:

        return X[_order-1]                                #X'=X[1]'=X[0]=V



### Main Program

if __name__ == "__main__":



    rk4_sol, t_rk4 = RK4(beta=10.,Xinit= [0, 0.009])                      #rk4_sol[0]= V, rk4_sol[1]= X



    #Draw all three methods

    fig = plt.figure(figsize=(18,6))

    plt.plot(t_rk4,rk4_sol[1])

    plt.xlim([0,20])

    plt.xlabel('Time (s)')

    plt.ylabel('Displacement (m)')

    plt.savefig('Minh-19B60127.png')

    plt.show()

    plt.close()








