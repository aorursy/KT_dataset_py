# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import torch

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim

%matplotlib inline
%config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, BayesianRidge
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

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
deaths=get_covid_data(subset = 'DEATH') # global deaths
deaths
countries=['Brazil', 'Canada', 'Germany','US','Spain','Italy']
y=deaths.loc[deaths['Country/Region']=='Italy'].iloc[0,4:]
s = pd.DataFrame({'Italy':y})
for c in countries:    
    s[c] = deaths.loc[deaths['Country/Region']==c].iloc[0,4:]
    plt.plot(range(y.shape[0]),s[c],label=c)#    print(s[c])
plt.title('Total Number of Deaths since 1/22/20')
plt.xlabel('Day')
plt.ylabel('Number of Cases')
plt.legend(loc="best")
plt.show()
import matplotlib.pyplot as plt
country_list=deaths['Country/Region'].unique()
confirmed = pd.DataFrame({'Italy':y})
dict={}
a=[]
b=[]
#z=y.shape[0]

for c in country_list:
  #  print(c)
    a.append(c)
   # print(a)
    confirmed=( deaths.loc[deaths['Country/Region']==c].iloc[:,4:].sum(axis=0))
    b.append(confirmed[y.shape[0]-1])  
    dict[c]=confirmed[y.shape[0]-1]
#    print (confirmed[c][84])
dict
f = plt.figure(figsize=(90,40))
f.add_subplot(111)

barWidth=1
plt.axes(axisbelow=True)

plt.bar(a,b,linewidth=17.0)

plt.xlabel("Countries ",fontsize=45)
plt.ylabel("Number of deaths ",fontsize=45)
plt.title("Number of deaths around the world",fontsize=60)
plt.grid(alpha=0.3)
plt.tick_params(size=5,labelsize = 30,rotation=90)
plt.show()
plt.figure(figsize=(15, 8))
canada = deaths.loc[deaths['Country/Region']=='Canada'].iloc[:,4:].sum(axis=0)
canada.tail()
canada.plot(label='Canada')
plt.legend()
plt.xlabel("Date ",fontsize=25)
plt.ylabel("Number of deaths ",fontsize=25)
plt.title("Number of deaths in Canada")
plt.show()
CAN = deaths[deaths['Country/Region']=='Canada']

CAN = pd.DataFrame(CAN.iloc[0,4:-2])

def plot_moving_average(series, window, plot_intervals=False, scale=1.96):

    rolling_mean = series.rolling(window=window).mean()
    
    plt.figure(figsize=(20,8))
    plt.title('Moving average\n window size = {}'.format(window))
    plt.plot(rolling_mean, 'g', label='Rolling mean trend')
    
    #Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bound = rolling_mean - (mae + scale * deviation)
        upper_bound = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
        plt.plot(lower_bound, 'r--')
        
            
    plt.plot(series[window:], label='Actual values')
    plt.legend(loc='best')
    plt.xticks(rotation=90)
    plt.grid(True)

#Smooth by the previous 5 days (by week)
plot_moving_average(CAN, 5)
plot_moving_average(CAN, 30, plot_intervals=True)
dates=deaths.columns.values.tolist()
dates=dates[4:]
d=[]
for i in dates:
  d= deaths.iloc[:,4:].sum(axis=0)
d
X = np.array([i for i in range(len(dates))]).reshape(-1, 1)
Y = np.array(d).reshape(-1, 1)
days_in_future = 15 #next 2 weeks
future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-15]
start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
    
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, Y, test_size=0.10, shuffle=False)
# svm_confirmed = svm_search.best_estimator_


svm_confirmed2 = SVR(C=1,degree=5,kernel='poly',epsilon=0.01)
svm_confirmed2.fit(X_train_d, y_train_d)
svm_pred2 = svm_confirmed2.predict(future_forcast)
svm_test_pred2 = svm_confirmed2.predict(X_test_d)
plt.plot(y_test_d)
plt.plot(svm_test_pred2)
plt.legend(['Test Data', 'SVM Predictions'])
print('MAE:', mean_absolute_error(svm_test_pred2, y_test_d))
print('MSE:',mean_squared_error(svm_test_pred2, y_test_d))

poly = PolynomialFeatures(degree=3)
poly_X_train_d = poly.fit_transform(X_train_d)
poly_X_test_d = poly.fit_transform(X_test_d)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly = PolynomialFeatures(degree=4)
bayesian_poly_X_train_d = bayesian_poly.fit_transform(X_train_d)
bayesian_poly_X_test_d = bayesian_poly.fit_transform(X_test_d)
bayesian_poly_future_forcast = bayesian_poly.fit_transform(future_forcast)
linear_model = LinearRegression(normalize=True, fit_intercept=True)
linear_model.fit(poly_X_train_d, y_train_d)
test_linear_pred = linear_model.predict(poly_X_test_d)
linear_pred = linear_model.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred, y_test_d))
print('MSE:',mean_squared_error(test_linear_pred, y_test_d))
plt.plot(y_test_d)
plt.plot(test_linear_pred)
plt.legend(['Test Data', 'Polynomial Regression Predictions'])
# bayesian ridge polynomial regression
tol = [1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4]

bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2}

bayesian2 = BayesianRidge(fit_intercept=False, normalize=True)
bayesian_search2 = RandomizedSearchCV(bayesian2, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search2.fit(bayesian_poly_X_train_d, y_train_d)
bayesian_d = bayesian_search2.best_estimator_
test_bayesian_pred = bayesian_d.predict(bayesian_poly_X_test_d)
bayesian_pred = bayesian_d.predict(bayesian_poly_future_forcast)
print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_d))
print('MSE:',mean_squared_error(test_bayesian_pred, y_test_d))
plt.plot(y_test_d)
plt.plot(test_bayesian_pred)
plt.legend(['Test Data', 'Bayesian Ridge Polynomial Predictions'])
def plot_predictions_death(x, y, pred, algo_name, color):
    plt.figure(figsize=(7, 5))
    plt.plot(x, y)
    plt.plot(future_forcast, pred, linestyle='dashed', color=color)
    plt.title(' Deaths caused by Coronavirus Over Time', size=15)
    plt.xlabel('Days Since 1/22/2020', size=15)
    plt.ylabel('# of Cases', size=15)
    plt.legend(['Cases of death', algo_name], prop={'size': 15})
    plt.xticks(size=10)
    plt.yticks(size=10)
    plt.show()
plot_predictions_death(adjusted_dates, d, svm_pred2, 'SVM Predictions', 'purple')
plot_predictions_death(adjusted_dates, d, linear_pred, 'Polynomial Regression Predictions', 'orange')
plot_predictions_death(adjusted_dates, d, bayesian_pred, 'Bayesian Ridge Regression Predictions', 'green')
svm_df = pd.DataFrame({'Date': future_forcast_dates[-10:], 'SVM Predicted Deaths Worldwide': np.round(svm_pred2[-10:])})
svm_df
linear_pred = linear_pred.reshape(1,-1)[0]
svm_df = pd.DataFrame({'Date': future_forcast_dates[-10:], 'Polynomial Predicted Number of Deaths Worldwide': np.round(linear_pred[-10:])})
svm_df
# Future predictions using Bayesian Ridge 
svm_df = pd.DataFrame({'Date': future_forcast_dates[-10:], 'Bayesian Ridge Predicted Number of Deaths Worldwide': np.round(bayesian_pred[-10:])})
svm_df
df2 = pd.DataFrame(columns=['ds','y'])
df2
df2['ds'] = pd.to_datetime(dates)
for  j in range(0,len(d)):
 # print(d[j])
  df2['y'][j]=pd.to_numeric(d[j])
df2
from fbprophet import Prophet
m = Prophet(interval_width=0.95)
m.fit(df2)
future = m.make_future_dataframe(periods=7)
future_confirmed = future.copy() # for non-baseline predictions later on
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
confirmed_forecast_plot = m.plot(forecast)
confirmed_cases=get_covid_data(subset = 'CONFIRMED')# confirmed cases
confirmed_cases.loc[confirmed_cases['Country/Region']=='Canada']
c2=[]
for i in dates:
  c2= confirmed_cases.iloc[:,4:].sum(axis=0)
#  world_cases.append(c)
d2=[]
for i in dates:
  d2= deaths.iloc[:,4:].sum(axis=0)
d
from scipy.optimize import curve_fit
details = pd.DataFrame(columns=['ds','Confirmed','Deaths'])

details['ds'] = pd.to_datetime(dates)
for  j in range(0,len(d)):
 # print(d[j])
  details['Confirmed'][j]=pd.to_numeric(c2[j])
  details['Deaths'][j]=pd.to_numeric(d2[j])
x_data = range(len(details.index))
y_data = details['Confirmed']

def log_curve(x, k, x_0, ymax):
    return ymax / (1 + np.exp(-k*(x-x_0)))

# Fit the curve
popt, pcov = curve_fit(log_curve, x_data, y_data)#, bounds=([0,0,0],np.inf), maxfev=50000)
estimated_k, estimated_x_0, ymax= popt


# Plot the fitted curve
k = estimated_k
x_0 = estimated_x_0
y_fitted = log_curve(range(0,160), k, x_0, ymax)
print(k, x_0, ymax)
#print(y_fitted)
y_data.tail()

# Plot everything for illustration
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(0,160), y_fitted, '--', label='fitted')
ax.plot(x_data, y_data, 'o', label='Confirmed Data')
daily_deaths=d.copy()
daily_deaths.head()
plt.figure(figsize=(30, 15))
plt.plot(daily_deaths)
plt.title("Cumulative daily deaths");
plt.tick_params(size=15,labelsize = 15,rotation=90)
plt.show()
daily_deaths = daily_deaths.diff().fillna(daily_deaths[0]).astype(np.int64)
daily_deaths.head

plt.figure(figsize=(30, 15))

plt.plot(daily_deaths)
plt.title("Daily Deaths");

plt.tick_params(size=15,labelsize = 15,rotation=90)
plt.show()

test_data_size2 = 14

train_data2 = daily_deaths[:-test_data_size2]
test_data2 = daily_deaths[-test_data_size2:]

train_data2.shape
scaler = MinMaxScaler()

scaler = scaler.fit(np.expand_dims(train_data2, axis=1))

train_data2 = scaler.transform(np.expand_dims(train_data2, axis=1))

test_data2 = scaler.transform(np.expand_dims(test_data2, axis=1))
def create_sequences(data, seq_length):
    xs2 = []
    ys2 = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs2.append(x)
        ys2.append(y)

    return np.array(xs2), np.array(ys2)
seq_length = 5
X_train2, y_train2 = create_sequences(train_data2, seq_length)
X_test2, y_test2 = create_sequences(test_data2, seq_length)

X_train2 = torch.from_numpy(X_train2).float()
y_train2 = torch.from_numpy(y_train2).float()

X_test2 = torch.from_numpy(X_test2).float()
y_test2 = torch.from_numpy(y_test2).float()
class CoronaVirusPredictor(nn.Module):

  def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
    super(CoronaVirusPredictor, self).__init__()

    self.n_hidden = n_hidden
    self.seq_len = seq_len
    self.n_layers = n_layers

    self.lstm = nn.LSTM(
      input_size=n_features,
      hidden_size=n_hidden,
      num_layers=n_layers,
      dropout=0.5
    )
    self.rnn = nn.RNN( input_size=n_features, hidden_size=n_hidden,  num_layers=n_layers, batch_first=True, nonlinearity='relu')
 

    self.linear = nn.Linear(in_features=n_hidden, out_features=1)

  def reset_hidden_state(self):
    self.hidden = (
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
    )

  def forward(self, sequences):
    lstm_out, self.hidden = self.lstm(
      sequences.view(len(sequences), self.seq_len, -1),
      self.hidden
    )
    last_time_step = \
      lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
    y_pred = self.linear(last_time_step)
    return y_pred
#deaths
def train_model(
  model,
  train_data2,
  train_labels2,
  test_data2=None,
  test_labels2=None
):
  loss_fn = torch.nn.MSELoss(reduction='sum')

  optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
  num_epochs = 100

  train_hist2 = np.zeros(num_epochs)
  test_hist2 = np.zeros(num_epochs)

  for t in range(num_epochs):
    model.reset_hidden_state()

    y_pred2 = model(X_train2)

    loss2 = loss_fn(y_pred2.float(), y_train2)

    if test_data2 is not None:
      with torch.no_grad():
        y_test_pred2 = model(X_test2)
        test_loss2 = loss_fn(y_test_pred2.float(), y_test2)
      test_hist2[t] = test_loss2.item()

      if t % 10 == 0:
        print(f'Epoch {t} train loss: {loss2.item()} test loss: {test_loss2.item()}')
    elif t % 10 == 0:
      print(f'Epoch {t} train loss: {loss2.item()}')

    train_hist2[t] = loss2.item()

    optimiser.zero_grad()

    loss2.backward()

    optimiser.step()

  return model.eval(), train_hist2, test_hist2



model = CoronaVirusPredictor(
  n_features=1,
  n_hidden=90, #
  seq_len=seq_length,
  n_layers=2
)
#deaths
model, train_hist2, test_hist2 = train_model(
  model,
  X_train2,
  y_train2,
  X_test2,
  y_test2
)


scaler = MinMaxScaler()

scaler = scaler.fit(np.expand_dims(daily_deaths, axis=1))

all_data = scaler.transform(np.expand_dims(daily_deaths, axis=1))

all_data.shape
X_all, y_all = create_sequences(all_data, seq_length)

X_all = torch.from_numpy(X_all).float()
y_all = torch.from_numpy(y_all).float()

model = CoronaVirusPredictor(
  n_features=1,
  n_hidden=70,
  seq_len=seq_length,
  n_layers=2
)
model, train_hist2, _ = train_model(model, X_all, y_all)
DAYS_TO_PREDICT = 12

with torch.no_grad():
  test_seq = X_all[:1]
  preds = []
  for _ in range(DAYS_TO_PREDICT):
    y_test_pred2 = model(test_seq)
    pred = torch.flatten(y_test_pred2).item()
    preds.append(pred)
    new_seq = test_seq.numpy().flatten()
    new_seq = np.append(new_seq, [pred])
    new_seq = new_seq[1:]
    test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()
#As before, we’ll inverse the scaler transformation:

predicted_cases = scaler.inverse_transform(
  np.expand_dims(preds, axis=0)
).flatten()
daily_deaths.index[-1]
predicted_index = pd.date_range(
  start=daily_deaths.index[-1],
  periods=DAYS_TO_PREDICT + 1,
  closed='right'
)

predicted_cases = pd.Series(
  data=predicted_cases,
  index=predicted_index
)

plt.plot(predicted_cases, label='Predicted Daily Deaths')
plt.legend();
predicted_cases