from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
# set random seed
np.random.seed(123)
df = pd.read_csv("../input/weatherAUS.csv")
# remove RISK_MM as suggested
df = df.drop(['RISK_MM'], axis = 1)
df.head()
df.isna().sum()/df.count().max()*100
# add indicator columns for missing values
missing = df.isnull().astype(int).add_suffix("_missing")
print(missing.head())

# replace missing with mean
cleaned = df.fillna(df.mean())
cleaned = cleaned.dropna() # drop rows with NA for RainToday
cleaned = cleaned.join(missing)

# convert date to year and day of year[0-365]
cleaned['Year'] = pd.to_datetime(cleaned['Date']).dt.year
cleaned['Month'] = pd.to_datetime(cleaned['Date']).dt.month
cleaned['Day'] = pd.to_datetime(cleaned['Date']).dt.day
cleaned['DayOfYear'] = pd.to_datetime(cleaned['Date']).dt.strftime('%j')

# drop Date and unnecessary Missing Attributes
cleaned.drop(['Date','Date_missing','Location_missing','RainTomorrow_missing'],axis = 1, inplace=True)

# Convert categorical to numerical
cleaned = pd.get_dummies(cleaned, columns = ['Location','WindGustDir','WindDir9am','WindDir3pm'])

# Convert Yes/No to Boolean
cleaned = cleaned.replace({'RainToday': {'Yes': 1, 'No': 0}})
cleaned = cleaned.replace({'RainTomorrow': {'Yes': 1, 'No': 0}})
cleaned.head()
# Seperate X and Y
X = cleaned.drop('RainTomorrow',axis=1).astype('float64')
y = cleaned['RainTomorrow']


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
print(X_train.shape)
print(y_train.shape)
# create model
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train,y_train)
print("training score", model.score(X_train,y_train))
print("testing score", model.score(X_test,y_test))
