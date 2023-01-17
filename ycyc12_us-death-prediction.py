import requests
import pandas as pd
import io

BASE_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
CONFIRMED = 'time_series_covid19_confirmed_global.csv'
DEATH = 'time_series_covid19_deaths_global.csv'
RECOVERED = 'time_series_covid19_recovered_global.csv'
CONFIRMED_US = 'time_series_covid19_confirmed_US.csv'
DEATH_US = 'time_series_covid19_deaths_US.csv'

def get_covid_data(subset = 'CONFIRMED'):
    """This function returns the latest available data subset of COVID-19. 
        The returned value is in pandas DataFrame type.
    Args:
        subset (:obj:`str`, optional): Any value out of 5 subsets of 'CONFIRMED',
        'DEATH', 'RECOVERED', 'CONFIRMED_US' and 'DEATH_US' is a valid input. If the value
        is not chosen or typed wrongly, CONFIRMED subet will be returned.
    """    
    switcher =  {
                'CONFIRMED'     : BASE_URL + CONFIRMED,
                'DEATH'         : BASE_URL + DEATH,
                'RECOVERED'     : BASE_URL + RECOVERED,
                'CONFIRMED_US'  : BASE_URL + CONFIRMED_US,
                'DEATH_US'      : BASE_URL + DEATH_US,
                }

    CSV_URL = switcher.get(subset, BASE_URL + CONFIRMED)

    with requests.Session() as s:
        download        = s.get(CSV_URL)
        decoded_content = download.content.decode('utf-8')
        data            = pd.read_csv(io.StringIO(decoded_content))

    return data
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
#import data
death = get_covid_data(subset = 'DEATH')
Death_US = death[death['Country/Region']=='US']

death_train_US = pd.DataFrame(Death_US.iloc[0,4:-2])
death_test_US = pd.DataFrame(Death_US.iloc[0,-2:])
Death_US
countries=['US']

for r in death['Country/Region']:
    if r in countries:
        plt.plot(range(len(death.columns)-4), death.loc[death['Country/Region']==r].iloc[0,4:], label = r) 
plt.legend()
plt.title('Total Number of COVID-19 Death Cases_US')
plt.xlabel('Day')
plt.ylabel('Number of Cases')
plt.grid()
#training data
x_data = range(len(death_train_US))
y_data = death_train_US[225].values

#All data
All = pd.DataFrame(Death_US.iloc[0,4:])
All_x_data = range(len(All))
All_y_data = All[225].values
#training model
def log_curve(x, k, x_0, ymax):
    return ymax / (1 + np.exp(-k*(x-x_0)))

# Fit the curve
popt, pcov = curve_fit(log_curve, x_data, y_data, bounds=([0,0,0],np.inf), maxfev=100000)
estimated_k, estimated_x_0, ymax= popt


# the parameters for the fitted curve
k = estimated_k
x_0 = estimated_x_0
y_fitted = log_curve(All_x_data, k, x_0, ymax)
print(k, x_0, ymax)
#print(y_fitted)
# Plot the predict and death data
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(All_x_data, y_fitted, '--', label='Predicte')
ax.plot(All_x_data, All_y_data, 'o', label='Death')
plt.legend()
#For predict the next two-day's death in US
predict_set = range(len(All)+2)
y_predict = log_curve(predict_set, k, x_0, ymax)
Answer = y_predict[-2:]

#change float into int
Answer.astype(int)