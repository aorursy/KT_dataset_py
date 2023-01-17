import datetime



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



from scipy.integrate import odeint

from scipy.optimize import differential_evolution



from sklearn.metrics import r2_score
covid19_df = pd.read_csv("../input/covid19turkey/Covid19-Turkey.csv")

covid19_df.tail()
def SIR(x, t, BETA, GAMMA):

    S, I, R = x



    N = S + I + R



    dSdt = -(BETA * S * I) / N

    dIdt = ((BETA * S * I) / N) - (GAMMA * I)

    dRdt = GAMMA * I



    return dSdt, dIdt, dRdt
def SIRD(x, t, BETA, GAMMA, MU):

    S, I, R, D = x



    N = S + I + R + D



    dSdt = -(BETA * S * I) / N

    dIdt = ((BETA * S * I) / N) - (GAMMA * I) - (MU * I)

    dRdt = GAMMA * I

    dDdt = MU * I



    return dSdt, dIdt, dRdt, dDdt
T = len(covid19_df) * 10

day_count = len(covid19_df)

t = np.linspace(0, day_count, T)
population = 83_154_997 # Based on December 2019

sir_y0 = (1.0, 1 / population, 0) # num. of susceptibles, num. of initial infected count, num. of removed

sird_y0 = (1.0, 1 / population, 0, 0) # num. of susceptibles, num. of initial infected count, num. of recovered, num. of deceased
def sir_loss_1(x):

    BETA, GAMMA = x



    if BETA < 0 or GAMMA < 0:

        return np.inf

    

    y = odeint(SIR, sir_y0, t, args=(BETA, GAMMA))



    step = np.round((1 / day_count) * T).astype(int)



    I = y[0:step * len(covid19_df):step, 1]

    R = y[0:step * len(covid19_df):step, 2]

    

    return -r2_score(I, covid19_df["Active Cases"].values / population)



    #return (np.abs(I - covid19_df["Active Cases"].values / population).sum() / len(I))



def sir_loss_2(x):

    BETA, GAMMA = x



    if BETA < 0 or GAMMA < 0:

        return np.inf

    

    y = odeint(SIR, sir_y0, t, args=(BETA, GAMMA))



    step = np.round((1 / day_count) * T).astype(int)



    I = y[0:step * len(covid19_df):step, 1]

    R = y[0:step * len(covid19_df):step, 2]

    

    return -r2_score(R, ((covid19_df["Total Recovered"].values + covid19_df["Total Deaths"].values) / population))



    #return (np.abs(R - ((covid19_df["Total Recovered"].values + covid19_df["Total Deaths"].values) / population)).sum() / len(R))



def sir_loss_3(x):

    BETA, GAMMA = x



    if BETA < 0 or GAMMA < 0:

        return np.inf

    

    y = odeint(SIR, sir_y0, t, args=(BETA, GAMMA))



    step = np.round((1 / day_count) * T).astype(int)



    I = y[0:step * len(covid19_df):step, 1]

    R = y[0:step * len(covid19_df):step, 2]

    

    return -(r2_score(I, covid19_df["Active Cases"].values / population) + r2_score(R, ((covid19_df["Total Recovered"].values + covid19_df["Total Deaths"].values) / population)))



#     return (np.abs(I - covid19_df["Active Cases"].values / population).sum() / len(I)) + \

#         (np.abs(R - ((covid19_df["Total Recovered"].values + covid19_df["Total Deaths"].values) / population)).sum() / len(R))
def sird_loss(x):

    BETA, GAMMA, MU = x



    if BETA < 0 or GAMMA < 0 or MU < 0:

        return np.inf

    

    y = odeint(SIRD, sird_y0, t, args=(BETA, GAMMA, MU))



    step = np.round((1 / day_count) * T).astype(int)



    I = y[0:step * len(covid19_df):step, 1]

    R = y[0:step * len(covid19_df):step, 2]

    D = y[0:step * len(covid19_df):step, 3]

    

    return -(r2_score(I, covid19_df["Active Cases"].values / population) + \

        r2_score(R, covid19_df["Total Recovered"].values / population) + \

        r2_score(D, covid19_df["Total Deaths"].values / population))



#     return (np.abs(I - covid19_df["Active Cases"].values / population).sum() / len(I)) + \

#         (np.abs(R - covid19_df["Total Recovered"].values / population).sum() / len(R)) + \

#         (np.abs(D - covid19_df["Total Deaths"].values / population).sum() / len(D))
sir_res_loss_1 = differential_evolution(

    sir_loss_1,

    ((0, 50), (0, 50)),

    tol=0.00001

)



sir_res_loss_2 = differential_evolution(

    sir_loss_2,

    ((0, 100), (0, 100)),

    tol=0.00001

)



sir_res_loss_3 = differential_evolution(

    sir_loss_3,

    ((0, 100), (0, 100)),

    tol=0.00001

)



sird_res_loss = differential_evolution(

    sird_loss,

    ((0, 100), (0, 100), (0, 100)),

    tol=0.00001

)
print("SIR Loss 1:")

print(sir_res_loss_1)

print("R0:", sir_res_loss_1.x[0] / sir_res_loss_1.x[1])

print()



print("SIR Loss 2:")

print(sir_res_loss_2)

print("R0:", sir_res_loss_2.x[0] / sir_res_loss_2.x[1])

print()



print("SIR Loss 3:")

print(sir_res_loss_3)

print("R0:", sir_res_loss_3.x[0] / sir_res_loss_3.x[1])

print()

print(50 * "#")

print()



print("SIRD Loss:")

print(sird_res_loss)

print("R0:", sird_res_loss.x[0] / sird_res_loss.x[1])
step = np.round((1 / day_count) * T).astype(int)



sir_y_loss_1 = odeint(SIR, sir_y0, t, args=(sir_res_loss_1.x[0], sir_res_loss_1.x[1]))

sir_y_loss_2 = odeint(SIR, sir_y0, t, args=(sir_res_loss_2.x[0], sir_res_loss_2.x[1]))

sir_y_loss_3 = odeint(SIR, sir_y0, t, args=(sir_res_loss_3.x[0], sir_res_loss_3.x[1]))



sir_S_loss_1 = sir_y_loss_1[:, 0]

sir_I_loss_1 = sir_y_loss_1[:, 1]

sir_R_loss_1 = sir_y_loss_1[:, 2]



sir_S_loss_2 = sir_y_loss_2[:, 0]

sir_I_loss_2 = sir_y_loss_2[:, 1]

sir_R_loss_2 = sir_y_loss_2[:, 2]



sir_S_loss_3 = sir_y_loss_3[:, 0]

sir_I_loss_3 = sir_y_loss_3[:, 1]

sir_R_loss_3 = sir_y_loss_3[:, 2]
fig, ax = plt.subplots(3, 1, figsize=(20, 10), constrained_layout=True)



ax[0].plot(t, sir_I_loss_1, label="# of infectives")

ax[0].plot(t, sir_R_loss_1, label="# of removed")

ax[0].plot(t[0:step * len(covid19_df):step], covid19_df["Active Cases"].values / population, label="# of real infectives")

ax[0].plot(t[0:step * len(covid19_df):step], (covid19_df["Total Recovered"] + covid19_df["Total Deaths"]) / population, label="# of real removed")

ax[0].set_xlabel("Days")

ax[0].set_ylabel("Normalized Population")

ax[0].set_title("147 Days of Simuation w/ Loss 1")

ax[0].legend()



ax[1].plot(t, sir_I_loss_2, label="# of infectives")

ax[1].plot(t, sir_R_loss_2, label="# of removed")

ax[1].plot(t[0:step * len(covid19_df):step], covid19_df["Active Cases"].values / population, label="# of real infectives")

ax[1].plot(t[0:step * len(covid19_df):step], (covid19_df["Total Recovered"] + covid19_df["Total Deaths"]) / population, label="# of real removed")

ax[1].set_xlabel("Days")

ax[1].set_ylabel("Normalized Population")

ax[1].set_title("147 Days of Simuation w/ Loss 2")

ax[1].legend()



ax[2].plot(t, sir_I_loss_3, label="# of infectives")

ax[2].plot(t, sir_R_loss_3, label="# of removed")

ax[2].plot(t[0:step * len(covid19_df):step], covid19_df["Active Cases"].values / population, label="# of real infectives")

ax[2].plot(t[0:step * len(covid19_df):step], (covid19_df["Total Recovered"] + covid19_df["Total Deaths"]) / population, label="# of real removed")

ax[2].set_xlabel("Days")

ax[2].set_ylabel("Normalized Population")

ax[2].set_title("147 Days of Simuation w/ Loss 3")

ax[2].legend()



plt.show()
step = np.round((1 / day_count) * T).astype(int)



sird_y_loss = odeint(SIRD, sird_y0, t, args=(sird_res_loss.x[0], sird_res_loss.x[1], sird_res_loss.x[2]))



sird_S_loss = sird_y_loss[:, 0]

sird_I_loss = sird_y_loss[:, 1]

sird_R_loss = sird_y_loss[:, 2]

sird_D_loss = sird_y_loss[:, 3]
fig, ax = plt.subplots(1, 1, figsize=(20, 6), constrained_layout=True)



ax.plot(t, sird_I_loss, label="# of infectives")

ax.plot(t, sird_R_loss, label="# of recovered")

ax.plot(t, sird_D_loss, label="# of deceased")

ax.plot(t[0:step * len(covid19_df):step], covid19_df["Active Cases"].values / population, label="# of real infectives")

ax.plot(t[0:step * len(covid19_df):step], covid19_df["Total Recovered"] / population, label="# of real recovered")

ax.plot(t[0:step * len(covid19_df):step], covid19_df["Total Deaths"] / population, label="# of real deceased")

ax.set_xlabel("Days")

ax.set_ylabel("Normalized Population")

ax.set_title("147 Days of Simuation w/ Loss")

ax.legend()



plt.show()