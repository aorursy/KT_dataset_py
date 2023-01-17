# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path

from scipy.integrate import solve_ivp



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import folium

import geopandas as gpd

import numpy as np

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster

import math

import matplotlib.pyplot as plt# Any results you write to the current directory are saved as output.


confirmed_cases_profile=pd.read_csv("/kaggle/input/covid19coronavirusindiadataset/complete.csv")

hospital_beds=pd.read_csv("/kaggle/input/hospitalbedloc/HospitalBedsIndiaLocations.csv")

population=pd.read_csv("/kaggle/input/covid19-in-india/population_india_census2011.csv") #2011 population data



confirmed_cases_profile.head()

confirmed_cases = gpd.GeoDataFrame(confirmed_cases_profile, geometry=gpd.points_from_xy(confirmed_cases_profile.Longitude, confirmed_cases_profile.Latitude))

confirmed_cases.crs = {'init': 'epsg:4326'}



hospital_beds = gpd.GeoDataFrame(hospital_beds, geometry=gpd.points_from_xy(hospital_beds.Longitude, hospital_beds.Latitude))

hospital_beds.crs = {'init': 'epsg:4326'}



m_1 = folium.Map(location=[confirmed_cases.Latitude[19],confirmed_cases.Longitude[19]], tiles='cartodbpositron', zoom_start=4.5)



# Add points to the map

mc = MarkerCluster()

for idx, row in confirmed_cases.iterrows():

    if not math.isnan(row['Longitude']) and not math.isnan(row['Latitude']):

        mc.add_child(Marker([row['Latitude'], row['Longitude']]))

m_1.add_child(mc)



# Display the map

m_1

# Plot 1:  Confirmed cases distribution in India on 22 March 2020
m_2 = folium.Map(location=[hospital_beds.Latitude[1],hospital_beds.Longitude[1]], tiles='cartodbpositron', zoom_start=5)



#It is expected that the hospital bed occupancy rate should be at least 80% 

#Compendium of Norms for Designing of Hospitals & Medical Institutions

#https://cpwd.gov.in/Publication/Compendium_of_Norms_for_Designing_of_Hospitals_and_Medical_Institutions.pdf



bor = 0.8  # Bed Occupancy Rate



def color(magnitude):

    if magnitude<10:

        col='red'

    else: 

        col='green'

    return col





# Add points to the map

mc = MarkerCluster()

for idx, row in hospital_beds.iterrows():



    num_beds=int((1/1000)*(1-bor)*(float(row['NumPublicBeds_HMIS'])+float(row['NumRuralBeds_NHP18'])+float(row['NumUrbanBeds_NHP18']))) # x 1000 beds

    

    for row_beds in range(int(num_beds)):

        if not math.isnan(row['Longitude']) and not math.isnan(row['Latitude']):

            #mc.add_child(folium.Marker([row['Latitude']+0.001*np.random.rand(), 0.001*np.random.rand()+row['Longitude']])) 

            mc.add_child(folium.Marker([row['Latitude']+0.001*np.random.rand(), 0.001*np.random.rand()+row['Longitude']],popup='Magnitude:'+str(num_beds),icon=folium.Icon(color=color(num_beds))))

            m_2.add_child(mc)

   





# Display the map

m_2
# Plot 2: Hospital beds in each state (x1000)
# Projected cases developement



# average incubation period

t_incubation = 5.5  # Mean incubation period (days) World Health Organization (2020). Report of the WHO-China Joint Mission on Coronavirus Disease 2019 (COVID-19). 



# average infectious period

t_infectious = 3.6 # https://www.medrxiv.org/content/10.1101/2020.02.12.20022566v1.full.pdf?fbclid=IwAR3sxNN6gSOIiNulVyTbZOjWm3uEjdnr2QPVJWncS8ZVGZ2Pd7DYoD_beWs



# reproduction number without intervention

# India has a smaller dataset and limited testing until 22 March 2020 which gives a relatively less R0, lets assume R0 from academic database for China

# https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf

# R0 = 2.4 international data

R0 = 3 # assuming agressive expansion



# Percentage population of severe cases requiring hospitalization w.r.t infected cases : 13.5 %

# Source: https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf

severe_fraction = 0.138

# SIER Reference: https://www.kaggle.com/anjum48/seir-model-with-intervention



# Susceptible equation

def dS_dt(S, I, R_t, T_inf):

    return -(R_t / T_inf) * I * S



# Exposed equation

def dE_dt(S, E, I, R_t, T_inf, T_inc):

    return (R_t / T_inf) * I * S - (T_inc**-1) * E



# Infected equation

def dI_dt(I, E, T_inc, T_inf):

    return (T_inc**-1) * E - (T_inf**-1) * I



# Recovered/deceased equation

def dR_dt(I, T_inf):

    return (T_inf**-1) * I



def SEIR_model(t, y, R_t, T_inf, T_inc):

    

    if callable(R_t):

        reproduction = R_t(t)

    else:

        reproduction = R_t

    

    S, E, I, R = y

    S_out = dS_dt(S, I, reproduction, T_inf)

    E_out = dE_dt(S, E, I, reproduction, T_inf, T_inc)

    I_out = dI_dt(I, E, T_inc, T_inf)

    R_out = dR_dt(I, T_inf)

    return [S_out, E_out, I_out, R_out]
    # Reference: https://www.kaggle.com/anjum48/seir-model-with-intervention

    

    def plot_model(solution,state, title='SEIR model'):

        sus, exp, inf, rec = solution.y*N.iloc[0]               # N is the state population

        f = plt.figure(figsize=(16,10)) 

        

        #plt.plot(sus, 'b', label='Susceptible');              

        

        #plt.plot(exp, 'y', label='Exposed');

        

        plt.plot(inf, 'r', label='Infected');

        

        plt.plot(inf*severe_fraction, 'y', label='Severe cases requiring hospitalization');



        #plt.plot(rec, 'c', label='Recovered/deceased');

        

        plt.plot((1-bor)*num_beds*np.ones(rec.shape), 'g', label='Number of available beds in: '+ state)

        

        plt.title(title+' in '+state)

        

        plt.xlabel("Days since 22 March", fontsize=10);

        

        plt.ylabel("Population", fontsize=10);

        

        #plt.yscale('log')

        severe=inf*severe_fraction;

        

        plt.ylim(0,severe.max()) # limit view to max value to severe cases

        

        plt.grid(True)

        

        plt.legend(loc='best');
statelist=hospital_beds.loc[:,'State/UT']



max_days = 80                                         # Number of days into the simulation 



time_to_go_state=np.zeros(statelist.shape)            # Initialize variable, time to go until the number of severe cases requiring hospitalization is more than beds available for the patients



for i,row in hospital_beds.iterrows():

    

    state=row['State/UT']

    

    staterow=hospital_beds[hospital_beds['State/UT']==state]



    num_beds=int((1-bor)*(float(staterow['NumPublicBeds_HMIS'])+float(staterow['NumRuralBeds_NHP18'])+float(staterow['NumUrbanBeds_NHP18'])))

    N = population[population['State / Union Territory']==state]['Population']

    if state in confirmed_cases_profile['Name of State / UT'].values:



        Indian_National = confirmed_cases_profile[confirmed_cases_profile['Name of State / UT']==state]['Total Confirmed cases (Indian National)'][-1:]

        Foreign_National = confirmed_cases_profile[confirmed_cases_profile['Name of State / UT']==state]['Total Confirmed cases ( Foreign National )'][-1:]

        n_infected = Indian_National.iloc[0]+Foreign_National.iloc[0]

        

        # Initial state

        a = (N - n_infected)/ N

        b = 0

        c = n_infected / N

        d = 0.

        

        T_inc = t_incubation  # average incubation period

        T_inf = t_infectious  # average infectious period

        R_0 = R0  # reproduction number

        sol = solve_ivp(SEIR_model, [0, max_days], [a, b, c, d], args=(R_0, T_inf, T_inc), t_eval=np.arange(max_days))

        

        plot_model(sol,state, 'SEIR Model (without intervention)')

        

        # Time till critical healthcare thresholds of the state are hit

        

        sus, exp, inf, rec = sol.y*N.iloc[0]

        

        severe_population=inf*severe_fraction                       # Population of severe cases

        

        available_beds=(1-bor)*num_beds                             # Typical number of beds available



        sign_array=severe_population-available_beds                  

        time_to_go= (len(sign_array)-len(sign_array[sign_array>0])) # Gives time from day 0 until crisis

        time_to_go_state[i]=time_to_go

    

    else:

    

        time_to_go_state[i]=max_days



    

hospital_beds['Time to go before healthcare crisis in days']=time_to_go_state       

        
m_3 = folium.Map(location=[hospital_beds.Latitude[1],hospital_beds.Longitude[1]], tiles='cartodbpositron', zoom_start=7)



#It is expected that the hospital bed occupancy rate should be at least 80% 

#Compendium of Norms for Designing of Hospitals & Medical Institutions

#https://cpwd.gov.in/Publication/Compendium_of_Norms_for_Designing_of_Hospitals_and_Medical_Institutions.pdf



min_time=hospital_beds['Time to go before healthcare crisis in days'].min()



# Add points to the map

mc = MarkerCluster()

for idx, row in hospital_beds.iterrows():



    num=100 * min_time/hospital_beds['Time to go before healthcare crisis in days'][idx]

    

    for row_beds in range(int(num)):

        if not math.isnan(row['Longitude']) and not math.isnan(row['Latitude']):

            mc.add_child(Marker([row['Latitude']+0.001*np.random.rand(), 0.001*np.random.rand()+row['Longitude']]))

            m_3.add_child(mc)

    



# Display the map

m_3

# Plot 3: Relative urgency in action required from each state