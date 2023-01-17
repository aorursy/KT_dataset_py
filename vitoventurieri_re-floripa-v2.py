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
    dffloripa = pd.read_csv("../input/re-floripa-v4/Re_floripa_v4.csv", sep=',')
    
    dffloripa.iloc[[14, 15, 16],10]
dfbrasil = pd.read_csv("../input/re-cidades/covid_reproduction_number_estimates.csv", sep=',')
dfbrasil = dfbrasil.loc[dfbrasil.state == 'Brazil'].reset_index(drop=True)
dffloripa

casos_brasil = pd.read_csv('../input/covid19-cases-in-brazil-at-city-level/cases-brazil-cities-time.csv', sep=',')
casos_brasil.loc[casos_brasil2.date == '2020-03-27'].reset_index(drop=True)
casos_brasil = pd.read_csv('../input/covid19-cases-in-brazil-at-city-level/cases-brazil-cities-time.csv', sep=',')
casos_brasil = casos_brasil.deaths[casos_brasil.state == 'TOTAL']
casos_brasil = casos_brasil.loc[casos_brasil2.date > '2020-03-13'].reset_index(drop=True)
casos_brasil = casos_brasil/(0.006)
casos_brasil
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

dfbrasil['Rt_mean'].plot()
dfbrasil
a = dffloripa.iloc[:13,10]
dffloripa.sort_values('date', ignore_index=True)
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

# colocar na variavel datasetrz os r(ts)para cada dia

def make_lognormal_params_95_ci(lb, ub):
    mean = (ub*lb)**(1/2)
    std = (ub/lb)**(1/4)
    return mean, std

def run_SEIR_ODE_model(
        N: 'population size',
        E0: 'init. exposed population',
        I0: 'init. infected population',
        R0: 'init. removed population',
        beta: 'infection probability',
        gamma: 'removal probability', 
        alpha_inv: 'incubation period', 
        t_max: 'numer of days to run'
    ) -> pd.DataFrame:

    S0 = N - I0 - R0 - E0
    alpha = 1/alpha_inv

    # A grid of time points (in days)
    t = range(t_max)

    # The SEIR model differential equations.
    def deriv(y, t, N, beta, gamma, alpha):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = -dSdt - alpha*E
        dIdt = alpha*E - gamma*I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    # Initial conditions vector
    y0 = S0, E0, I0, R0

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, alpha))
    S, E, I, R = ret.T

    return pd.DataFrame({'S': S, 'E': E, 'I': I, 'R': R}, index=t)


if __name__ == '__main__':
    #Re_series = pd.Series(range(700,1)) # aqui tem que ir a sequencia de dias do rt
    
    dffloripa = pd.read_csv("../input/re-floripa-v4/Re_floripa_v4.csv", sep=',')
    re_floripa = dffloripa.iloc[13:,13:]
    csv_de_rzeros = re_floripa.iloc[:,0] # iloc 0 para média, 1 para limite superior 2 para inferior
    Re_series = dfbrasil['Rt_mean'] #pd.Series(csv_de_rzeros.iloc[:])
    
        
    #Initial parameters
    
    N = 200_000_000
    E0, I0, R0 = 30, 30, 30
    Sinicial = N - I0 - R0 - E0
    gamma= 1/2.8
    alpha_inv = 2.9 
    
    #solves several ODEs in sequence using r(e) from the Re_series
    errezero = Re_series.iloc[0]
    beta = gamma*errezero
    t_max = 2
    fitado = pd.DataFrame({'S': [Sinicial], 'E': [E0], 'I': [I0], 'R': [R0]})
    j = 0
    
    
    #Roda o intervalo todo usando média, sup ou inf do r(efetivo)
    
    for i in range(len(Re_series)-t_max):
        j = j + (t_max-1)
        results = run_SEIR_ODE_model(N, E0, I0, R0, beta, gamma, alpha_inv, t_max)
        fitado = pd.concat([fitado, results.loc[1:,:]], ignore_index=True)
        E0, I0, R0 = fitado.loc[j,'E'],fitado.loc[j,'I'], fitado.loc[j,'R']
        effectiver = Re_series.iloc[j]
        errezero = effectiver*(N)/(N-(fitado.loc[j,'E'])-(fitado.loc[j,'I'])-(fitado.loc[j,'R']))
        beta = gamma*errezero
        
    #roda o seguimento da curva
        
    def step_ic_re(inf, sup):
        intervalos = np.arange(inf, sup, 0.01)
        return intervalos

    inferior = re_floripa.iloc[-1,2]
    superior = re_floripa.iloc[-1,1]
    rang_sup_inf = step_ic_re(inferior,superior)
    
    
    # aqui é a simulação para o futuro
    effectiver = rang_sup_inf.mean() # #rang_sup_inf[len(rang_sup_inf)-1] 
    errezero = effectiver*(N)/(N-(fitado.loc[j,'E'])-(fitado.loc[j,'I'])-(fitado.loc[j,'R']))
    beta = gamma*errezero
    t_max = 10
    
    results = run_SEIR_ODE_model(N, E0, I0, R0, beta, gamma, alpha_inv, t_max)
    fitado = pd.concat([fitado, results.loc[1:,:]], ignore_index=True)
    E0, I0, R0 = fitado.loc[j,'E'],fitado.loc[j,'I'], fitado.loc[j,'R']

        
    # plot
    #variable
    plt.style.use('ggplot')
    (fitado
    [['E', 'I', 'R']]
     .plot(figsize=(8,6), fontsize=20, logy=False))
    params_title = (
        f'SEIR($\gamma$={gamma}, $\\alpha$={1/alpha_inv}, $N$={N}, '
        f'$E_0$={int(E0)}, $I_0$={int(I0)}, $R_0$={int(R0)})'
    )
    plt.title(f'Numero de Pessoas Atingidas com modelo no "t_final -1" da simulacao:\n r(e)=vide lista, ' + params_title,
              fontsize=20)
    plt.legend(['Expostas', 'Infectadas', 'Recuperadas'], fontsize=20)
    plt.xlabel('Dias', fontsize=20)
    plt.ylabel('Pessoas', fontsize=20)
    
    plt.plot(casos_brasil)
    plt.show()
    
    plt.style.use('ggplot')
    (fitado
    [['E', 'I']]
     .plot(logy=False))
    plt.legend(['Expostas', 'Infectadas'], fontsize=20)
    plt.xlabel('Dias', fontsize=20)
    plt.ylabel('Pessoas', fontsize=20)
    plt.show()
def divide_gamma(x):
    return x*gamma

algo = (fitado.iloc[:,2].apply(divide_gamma)-dffloripa.sort_values('date', ignore_index=True).iloc[:79,8])
algo.std()

    
    #R0_ = npr.lognormal(*map(np.log, R0__params), runs)
    #gamma = 1/npr.lognormal(*map(np.log, gamma_inv_params), runs)
    #alpha = 1/npr.lognormal(*map(np.log, alpha_inv_params), runs)
    #beta = R0_*gamma
    