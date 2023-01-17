# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Algebra
import numpy as np

# Dataframe
import pandas as pd

# Missing Analysis
import missingno as msno

# Modelling
from scipy import integrate
from scipy import optimize

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#US
us_icu_death = pd.read_csv('/kaggle/input/uncover/UNCOVER/ihme/2020_03_30/Hospitalization_all_locs.csv')
us_status = pd.read_csv('/kaggle/input/uncover/UNCOVER/covid_tracking_project/covid-statistics-for-all-us-daily-updates.csv')
us_recover = pd.read_csv('/kaggle/input/uncover/UNCOVER/johns_hopkins_csse/2019-novel-coronavirus-covid-19-2019-ncov-data-repository-recovered.csv')
hosp_cap = pd.read_csv('/kaggle/input/uncover/UNCOVER/harvard_global_health_institute/hospital-capacity-by-state-20-population-contracted.csv')
#Italy
ita_status = pd.read_csv('/kaggle/input/uncover/UNCOVER/github/covid-19-italy-situation-monitoring-by-region.csv')
#France
fra_status = pd.read_csv('/kaggle/input/uncover/UNCOVER/github/covid19-epidemic-french-national-data.csv')
#Spain
sp_status = pd.read_csv('/kaggle/input/uncover/UNCOVER/github/covid19-spain-cases.csv')
#Disease risk factors
risk_factors_1 = pd.read_csv('/kaggle/input/uncover/UNCOVER/us_cdc/us_cdc/u-s-chronic-disease-indicators-cdi.csv')
risk_factors_2 = pd.read_csv('/kaggle/input/uncover/UNCOVER/us_cdc/us_cdc/500-cities-census-tract-level-data-gis-friendly-format-2019-release.csv')
risk_factors_3 = pd.read_csv('/kaggle/input/uncover/UNCOVER/us_cdc/us_cdc/nutrition-physical-activity-and-obesity-behavioral-risk-factor-surveillance-system.csv')
risk_tobacco_1 = pd.read_csv('/kaggle/input/uncover/UNCOVER/us_cdc/us_cdc/global-tobacco-surveillance-system-gtss-global-youth-tobacco-survey-gyts.csv')
risk_tobacco_2 = pd.read_csv('/kaggle/input/uncover/UNCOVER/us_cdc/us_cdc/global-tobacco-surveillance-system-gtss-global-adult-tobacco-survey-gats.csv')
# US
print(us_status.head())
us_status.drop(axis=1,inplace=True,columns=['states','posneg','pending','total','hash','datechecked'])
us_status.sort_values(by=['date'],ascending=True,inplace=True)
print(us_status)
print(us_recover.head())
# group by date
us_recover = us_recover[[]]
# ITALY
ita_status.drop(axis=1, inplace=True, columns=['stato', 'lat', 'long', 'note_it', 'note_en', 'codice_regione', 'denominazione_regione'])
## Group by date
ita_status = ita_status[['data', 'totale_casi', 'terapia_intensiva','dimessi_guariti']].groupby('data').sum().reset_index()
## Change columns to a specific pattern
ita_status.columns = ['date','infected','icu','recovered']
print(fra_status)


# France
fra_status=fra_status[fra_status['maille_code'].isin(['FRA'])]
fra_status=fra_status[fra_status['source_nom'].isin(['Ministère des Solidarités et de la Santé'])]
fra_status.drop(axis=1, inplace=True, columns=['granularite','maille_code','maille_nom','source_nom','source_archive','source_url','source_type'])
fra_status = fra_status[['date', 'cas_confirmes', 'reanimation','gueris']]
## Change columns to a specific pattern
fra_status.columns = ['date','infected','icu','recovered']
fra_status.iloc[[0]['date']]
print(fra_status)

fra_status = fra_status[['date', 'cas_confirmes', 'reanimation','gueris']]
## Change columns to a specific pattern
fra_status.columns = ['date','infected','icu','recovered']



fra_status.reset_index()
print(fra_status)

def seir_model_ode(y, t, params): 
    '''
    Arguments:
    - y: dependent variables
    - t: independent variable (time)
    - params: Model params
    '''
    # Parameters to find
    infection_rate = params[0]
    recovery_rate = params[1]
    exposed_rate = params[2]
    
    # Y variables
    s = y[0]
    e = y[1]
    i = y[2]
    r = y[3]
    
    # SIR EDO System 
    dsdt = -exposed_rate*s*(i+e)
    dedt = (exposed_rate*s*(i+e)) - (infection_rate*e)
    didt = (infection_rate*e) - (recovery_rate*i)
    drdt = recovery_rate*i
    
    # Return our system
    return (dsdt, dedt, didt, drdt)
# FUNCTION - Calculate SEIR Model in t (time as days) based on given parameters
def calculate_seir_model(params, t, initial_condition):
    # Create an alias to our seir ode model to pass params to try
    seir_ode = lambda y,t:seir_model_ode(y, t, params)
    
    # Calculate ode solution, return values to each
    ode_result = integrate.odeint(func=seir_ode, y0=initial_condition, t=t)
    
    # Return results
    return ode_result
# FUNCTION - Auxiliar function to find the best parameters
def fit_seir_model(params_to_fit, t, initial_condition, i_r_true):
    # Calculate ODE solution for possible parameter, return values to each dependent variable:
    # (s, e, i and r)
    fit_result = calculate_seir_model(params_to_fit, t, initial_condition)
    
    # Calculate residual value between predicted VS true
    ## Note: ode_result[0] is S result
    residual_i = i_r_true[0] - fit_result[:,2]
    residual_r = i_r_true[1] - fit_result[:,3]

    # Create a np.array of all residual values for both (i) and (r)
    residual = np.concatenate((residual_i, residual_r))
    
    # Return results
    return residual

# Italy population (~)
N = 1000000*60.59
# Define Initial Condition (necessary for ODE solve)
I_start = ita_status.loc[0, 'infected']/N
E_start = (ita_status.loc[14, 'infected'] - ita_status.loc[0, 'infected'])/N
S_start = 1 - E_start - I_start
R_start = ita_status.loc[0, 'recovered']/N
## Set this values as a tuple
ic = (S_start, E_start, I_start, R_start)
# Create a tuple with the true values in fraction for Infected/Recovered cases (necessary for error measurement)
ita_beforelockdown=ita_status.loc[0:15]#data upto Mar 10, before the lockdown
ita_afterlockdown=ita_status.loc[16:]#data from Mar11 to end, after lockdown
i_r_true_bf = (list(ita_beforelockdown['infected']/N), list(ita_beforelockdown['recovered']/N))
i_r_true_af = (list(ita_afterlockdown['infected']/N), list(ita_afterlockdown['recovered']/N))
# Define a time array measure in days
time_opt_bf = range(0, len(ita_beforelockdown))
time_opt_af = range(0, len(ita_afterlockdown))
time_opt =range(0,len(ita_status))
I_start_af = ita_status.loc[len(ita_beforelockdown), 'infected']/N
E_start_af = (ita_status.loc[len(ita_beforelockdown)+14, 'infected'] - ita_status.loc[len(ita_beforelockdown), 'infected'])/N
S_start_af = 1 - E_start_af - I_start_af
R_start_af = ita_status.loc[len(ita_beforelockdown), 'recovered']/N
## Set this values as a tuple
ic_af = (S_start_af, E_start_af, I_start_af, R_start_af)
# Define a start guess for our parameters [infection_rate, recovered rate]
params_start_guess = [0.01, 0.001, 0.01]
optimal_params, sucess = optimize.leastsq(fit_seir_model,
                                          x0=params_start_guess,
                                          args=(time_opt_bf, ic, i_r_true_bf),
                                          ftol=1.49012e-15)
optimal_params_af, sucess = optimize.leastsq(fit_seir_model,
                                          x0=params_start_guess,
                                          args=(time_opt_af, ic_af, i_r_true_af),
                                          ftol=1.49012e-15)
print('## Italy before lockdown')
print('Optimize infection rate: ', optimal_params[0])
print('Optimize recovered rate: ', optimal_params[1])
print('Optimize exposed rate: ', optimal_params[2])
print('## Italy after lockdown')
print('Optimize infection rate: ', optimal_params_af[0])
print('Optimize recovered rate: ', optimal_params_af[1])
print('Optimize exposed rate: ', optimal_params_af[2])
# Get the optimal parameters
ir = optimal_params[0]
rr = optimal_params[1]
er = optimal_params[2]
ir_af = optimal_params_af[0]
rr_af = optimal_params_af[1]
er_af = optimal_params_af[2]
# Calculate a curve based on those parameters
fit_result_bf = calculate_seir_model((ir, rr, er), time_opt_bf, ic)
fit_result_af = calculate_seir_model((ir_af, rr_af, er_af), time_opt_af, ic_af)
# Plot the results for Infected/Recovered
## Define plot object
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

## Plot process
axes[0].plot(time_opt, i_r_true_bf[0]+i_r_true_af[0], 'g')
axes[0].plot(time_opt, np.hstack((fit_result_bf[:,2],fit_result_af[:,2])), 'y')
axes[0].legend(['Ground truth', 'Predicted'],loc=2, fontsize=15)
axes[0].set_title('Infected cases - ITALY',fontsize=20)
axes[1].plot(time_opt, i_r_true_bf[1]+i_r_true_af[1], 'g')
axes[1].plot(time_opt, np.hstack((fit_result_bf[:,3],fit_result_af[:,3])), 'y')
axes[1].legend(['Ground truth', 'Predicted'],loc=2, fontsize=15)
axes[1].set_title('Recovered cases - ITALY',fontsize=20);