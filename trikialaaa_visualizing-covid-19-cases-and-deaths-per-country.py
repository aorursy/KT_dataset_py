import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv("../input/covid19-global-daily-cases-deaths-updated/WHO-COVID-19-global-data.csv")
data.head()
data.columns
def vizualize_covid(country):
    data_country = data[data[' Country'] == country]
    
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(20, 5))
    fig.suptitle(country, fontsize=24)
    axs[0].plot( 'Date_reported', ' New_cases', data=data_country, color='blue', marker='', linewidth=2)
    axs[0].set_title('Daily new cases in '+country, fontsize=20)
    axs[0].set_xticks([])
    
    axs[1].plot( 'Date_reported', ' Cumulative_deaths', data=data_country, color='red', marker='', linewidth=2)
    axs[1].set_title('Cumulative deaths in '+country, fontsize=20)
    axs[1].set_xticks([])
for country in data[' Country'].unique()[:5]:
    vizualize_covid(country)