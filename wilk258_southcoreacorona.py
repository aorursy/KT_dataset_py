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
import matplotlib.pyplot as plt 

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import folium

from folium import Marker, Circle, PolyLine

from folium.plugins import HeatMap, MarkerCluster
import pandas as pd

case = pd.read_csv("../input/coronavirusdataset/case.csv")

patient = pd.read_csv("../input/coronavirusdataset/patient.csv")

route = pd.read_csv("../input/coronavirusdataset/route.csv")

time = pd.read_csv("../input/coronavirusdataset/time.csv")

trend = pd.read_csv("../input/coronavirusdataset/trend.csv")
case.head(3)
case.info()
case.isnull().sum()
print("confirmed sum:", case.confirmed.sum())

print("confirmed mean:", case.confirmed.mean())
print(case.province.value_counts())

case.province.value_counts().plot(kind='bar', color='blue')
print(case.city.value_counts())

case.city.value_counts().plot(kind='bar', color='green')
print(case.infection_case.value_counts())

case.infection_case.value_counts().plot(kind='bar', color='red')
province=case.groupby(["province"], as_index=False)["confirmed"].sum().sort_values(by=["confirmed"] ,ascending=False)

province
fig = px.line(province, x="province", y="confirmed", title='province')

fig.show()
city=case.groupby(["city"], as_index=False)["confirmed"].sum().sort_values(by=["confirmed"] ,ascending=False)

city
fig = px.line(city, x="city", y="confirmed", title='city')

fig.show()
ic=case.groupby(["infection_case"], as_index=False)["confirmed"].sum().sort_values(by=["confirmed"] ,ascending=False)

ic
fig = px.line(ic, x="infection_case", y="confirmed", title='Infection_Case sum')

fig.show()
#geo visualization

case[['latitude', 'longitude']] = case[['latitude', 'longitude']].replace('-', np.nan)
m_1 = folium.Map(location=[37, 126], tiles='openstreetmap', zoom_start=6)



for idx, row in case.iterrows():

    if pd.notnull(row['latitude']):

        Marker([row['latitude'], row['longitude']], popup=folium.Popup((

                                                            'Province : {province}<br>'

                                                            'City : {city}<br>'

                                                            'Group : {group}<br>'

                                                            'Infection Case :{case}<br>'

                                                            'Confirmed : {confirmed}').format(

                                                            province=row['province'],

                                                            city=row['city'],

                                                            group=row['group'],

                                                            case=row['infection_case'],

                                                            confirmed=row['confirmed']), max_width=450)

              ).add_to(m_1)



        Circle(location=[row['latitude'], row['longitude']],

               radius=row['confirmed']*5,

               fill=True

              ).add_to(m_1)

    

m_1
patient.head(3)
patient.dtypes
patient.isnull().sum()
print(patient.sex.value_counts())

sns.countplot(x="sex", hue="sex", data=patient)
patient.country.value_counts()
print(patient.infection_reason.value_counts())

patient.infection_reason.value_counts().plot(kind='bar', color='blue')
print(patient.confirmed_date.value_counts())

patient.confirmed_date.value_counts().plot(kind='bar', color='blue')
patient.state.value_counts()
route.head(3)
#patient route province 

print(route.province.value_counts())

route.province.value_counts().plot(kind='bar', color='blue')
# patient route province  city 

print(route.city.value_counts())

route.city.value_counts().plot(kind='bar', color='blue')
#patient route visited place 

print(route.visit.value_counts())

route.visit.value_counts().plot(kind='bar', color='blue')
route.head(3)
#route visualization 

m_2 = folium.Map(location=[37, 126], tiles='cartodbpositron', zoom_start=6)



current_id = 1

points = []



for idx, row in route.iterrows():

    if pd.notnull(row['latitude']):

        (Marker([row['latitude'], row['longitude']], 

               icon=folium.Icon(color='blue'),

               popup=folium.Popup((

                                                            'Patient id : {patient_id}<br>'

                                                            'Date : {date}<br>'

                                                            'Province : {province}<br>'

                                                            'City :{city}<br>'

                                                            'Visit : {visit}').format(

                                                            patient_id=row['patient_id'],

                                                            date=row['date'],

                                                            province=row['province'],

                                                            city=row['city'],

                                                            visit=row['visit']), max_width=450)

              )).add_to(m_2)

        

        if row['patient_id'] == current_id:

            points.append(tuple([row['latitude'], row['longitude']]))

        else :

            PolyLine(points, color='blue').add_to(m_2)

            current_id = row['patient_id']

            points = []

            points.append(tuple([row['latitude'], row['longitude']]))



m_2
trend.head(3)
trend.isnull().sum()
trend.dtypes
trend.describe()
trend.head(3)
trend.shape
sns.pairplot(trend[["cold", "flu", "pneumonia", "coronavirus"]], diag_kind="kde")
fig = px.line(trend, x="date", y="cold", title='time serise cold')

fig.show()
fig = px.line(trend, x="date", y="flu", title='time serise flu')

fig.show()
fig = px.line(trend, x="date", y="pneumonia", title='time serise pneumonia')

fig.show()
fig = px.line(trend, x="date", y="coronavirus", title='time serise coronavirus')

fig.show()
trend.head(5)
trend_ft = ['cold', 'flu', 'pneumonia']

X_train= trend[trend_ft]

y = trend.coronavirus

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "corona")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge.min()
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)

rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])
plt.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
plt.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})

preds["corona"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "corona",kind = "scatter")