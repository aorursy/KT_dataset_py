import pandas as pd
import numpy as np
import datetime as dt
from google.cloud import bigquery
from bq_helper import BigQueryHelper
import sklearn as sk
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
QUERY = """
        SELECT location, city, country, value, timestamp,longitude,latitude, pollutant, unit
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE timestamp > "2016-07-01"
        """
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
df = bq_assistant.query_to_pandas(QUERY)
#query_job = client.query(QUERY)
#df = query_job.to_dataframe()
df.head(1)
df.describe()

#Label pollutants into different categories by creating new column name Label
import datetime as dt
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['Date_converted']=df['timestamp'].map(dt.datetime.toordinal)
df['Label_pollutant'] = df['pollutant'].map({'pm25': 0, 'pm10': 1, 'so2':2,'no2': 3, 'o3':4,'co': 5, 'bc':6})
df['Label_country'] = df['country'].map({'CA': 0, 'TH':1, 'IN':2, 'NL':3, 'GB':4, 'CH':5, 'TR':6, 'PL':7, 'PT':8, 'ES':9, 'BR':10
                                         , 'PE':11, 'NO':12, 'HK':13, 'LV':14, 'IT':15, 'CN':16, 'GH':17, 'CL':18, 'CO':19, 'SI':20, 'BD':21
                                        , 'AE':22, 'MT':23, 'VN':24, 'BA':25, 'IE':26, 'BE':27, 'TW':28, 'LT':29, 'KZ':30, 'DE':31, 'SE':32, 'NG':33, 'MK':34
                                       , 'AU':35, 'BH':36, 'PH':37, 'RU':38, 'AD':39, 'AT':40, 'ID':41, 'HU':42, 'LK':43, 'CW':44, 'UG':45, 'FI':46, 'KE':47, 'IL':48
                                        , 'LU':49, 'HR':50, 'XK':51, 'UZ':52, 'GI':53, 'NP':54, 'SG':55, 'DK':56, 'CZ':57, 'MX':58, 'ET':59, 'KW':60, 'MN':61, 'AR':62
                                        , 'RS':63, 'ZA':64, 'SK':65, 'US':66, 'FR':67})
df.head(3)
import seaborn as sns
sns.countplot(x="pollutant", data=df)
df.boxplot(column="value", by='country', figsize=(40,8), layout=(1,2));
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X = df[['Date_converted','longitude','latitude','Label_pollutant']]
y = df['Label_country']
X_std = StandardScaler().fit_transform(X)
X = X_std
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
gnb = GaussianNB()
params = {}
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_pred))

skf = StratifiedKFold(n_splits=5)
gs = GridSearchCV(gnb, cv=skf, param_grid=params, return_train_score=True)
gs.fit(X_train, y_train)
gs.score(X_test, y_test)
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
cross_val_score(clf, X_train, y_train, cv=10)

skf = StratifiedKFold(n_splits=5)
gs = GridSearchCV(clf, cv=skf, param_grid=params, return_train_score=True)
gs.fit(X_train, y_train)
gs.score(X_test, y_test)
X_train.shape, y_train.shape
X_test.shape, y_test.shape
df["value"].plot(figsize=(10,10), linewidth=5, fontsize=20)
plt.xlabel('Days', fontsize=20);
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

#splitting data
X = df[['Date_converted','longitude','latitude']]
y = df['value']
X_std = StandardScaler().fit_transform(X)
X = X_std
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

model = LinearRegression() #create linear regression object
model.fit(X_train, y_train) #train model on train data
y_pred = model.predict(X_test)
# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('MSE :',mean_absolute_error(y_test, y_pred))
# Plot outputs
from sklearn.metrics import accuracy_score
print('Accuracy for linear SVM is',model.score(X_test, y_test))
print(cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error"))
QUERY = """
    SELECT
        extract(DAYOFYEAR from date_local) as day_of_year,
        aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary`
    WHERE
      city_name = "Los Angeles"
      AND state_name = "California"
      AND sample_duration = "24 HOUR"
      AND poc = 1
      AND EXTRACT(YEAR FROM date_local) = 2017
    ORDER BY day_of_year
        """
import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
df = bq_assistant.query_to_pandas(QUERY)
df.plot(x='day_of_year', y='aqi', style='.');
X = df[['day_of_year']]
y = df['aqi']
X_std = StandardScaler().fit_transform(X)
X = X_std

print(X_std.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,accuracy_score
from matplotlib.pyplot import figure

model = LinearRegression() #create linear regression object
model.fit(X_train, y_train) #train model on train data
y_pred = model.predict(X_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('MSE :',mean_absolute_error(y_test, y_pred))
print('Accuracy for linear SVM is',model.score(X_test, y_test))
print(cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error"))
print('Variance score: %.2f' % r2_score(y_test, y_pred))
# Plot outputs
figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.xlabel("day_of_year")
plt.ylabel("Air Quality Index")
plt.xticks(())
plt.show()

QUERY = """WITH normalised_pollution AS
                (
                SELECT country, pollutant, location,
                    CASE
                        WHEN unit = 'ppm' AND pollutant ='o3' AND value > 0 THEN value*1960
                        WHEN unit = 'µg/m³' AND pollutant ='o3' AND value > 0 THEN value
                        WHEN unit = 'ppm' AND pollutant ='no2' AND value > 0 THEN value*1880
                        WHEN unit = 'µg/m³' AND pollutant ='no2' AND value > 0 THEN value
                        WHEN unit = 'ppm' AND pollutant ='so2' AND value > 0 THEN value*2620
                        WHEN unit = 'µg/m³' AND pollutant ='so2' AND value > 0 THEN value
                        WHEN unit = 'ppm' AND pollutant ='co' AND value > 0 THEN value*1150
                        WHEN unit = 'µg/m³' AND pollutant ='co' AND value > 0 THEN value

                END AS converted_unit
                FROM `bigquery-public-data.openaq.global_air_quality`
                )
                SELECT country, pollutant, (SUM(converted_unit)/COUNT(*)) AS normalised_pollution
                    FROM normalised_pollution
                    WHERE converted_unit > 0
                    GROUP BY country, pollutant
                    ORDER by country, pollutant """
df = bq_assistant.query_to_pandas(QUERY)
df.head(15)
one_pollutant = df.loc[df['pollutant'] == 'o3']
ordered_pollution = one_pollutant.sort_values(by='normalised_pollution')
my_range=range(1,len(ordered_pollution.index)+1)
plt.figure(figsize=(15,10))
plt.hlines(y=my_range, xmin=0, xmax=ordered_pollution['normalised_pollution'], color='green')
plt.plot(ordered_pollution['normalised_pollution'], my_range, "C")
plt.yticks(my_range, ordered_pollution['country'])
plt.title("Current pollution values in different countries", loc='center')
plt.xlabel('pollution measurement(ug/m3)')
plt.ylabel('Country')