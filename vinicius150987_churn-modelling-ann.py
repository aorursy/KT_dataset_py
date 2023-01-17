import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid', palette='deep')

import warnings

warnings.filterwarnings('ignore')

import time

bins = range(0,100,10)

%matplotlib inline



import os

print(os.listdir("../input"))
df_raw = pd.read_csv('../input/Churn_Modelling.csv')
df_raw.info() 
df_raw.head()
df_raw.tail()
#Visualizing Dataset

def barchart (feature1, feature2):

    g = pd.crosstab(df_raw[feature1], df_raw[feature2]).plot(kind= 'bar', figsize=(10,10), rot=45)

    ax = g.axes

    for p in ax.patches:

        ax.annotate(f"{p.get_height() * 100 / df_raw.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),

         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),

         textcoords='offset points')

    plt.title('Exited bank for {}'.format(feature1))

    plt.legend(['Did Not Exited', 'Exited'])

    plt.grid(b= True, which='major', linestyle='--')

    plt.xlabel('{}'.format(feature1))

    plt.tight_layout()

    plt.ylabel('Quantity')



def bar_chart_group(feature):

    g = pd.crosstab(pd.cut(df_raw[feature], bins), df_raw['Exited']).plot(kind='bar', figsize=(12,12), rot = 45)

    ax = g.axes

    for p in ax.patches:

     ax.annotate(f"{p.get_height() * 100 / df_raw.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),

         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),

         textcoords='offset points') 

    plt.grid(b=True, which='major', linestyle='--')

    plt.legend(['Did Not Exited', 'Exited'])

    plt.title('Exited bank for {}'.format(feature))

    plt.xlabel('{}'.format(feature))

    plt.tight_layout()

    plt.ylabel('Quantity')



def geography (feature1, feature2):

    df_raw.groupby(feature1)[feature2].sum().sort_values().plot(kind='bar', figsize=(10,10), rot=45)

    plt.title('Geography {}'.format(feature2))

    plt.grid(b=True, which='major', linestyle='--')

    plt.tight_layout()

    plt.ylabel('{}'.format(feature2))    

    
sns.countplot(df_raw.Exited)
barchart('Geography', 'Exited')
barchart('NumOfProducts', 'Exited')
barchart('HasCrCard', 'Exited')
barchart('IsActiveMember', 'Exited')
bar_chart_group('Age')
#Taking latitude and longitude

from geopy.geocoders import Nominatim

lat = np.array([])

lon = np.array([])

country = np.array([])

countries = df_raw.groupby('Geography')['Geography'].unique().sort_values()

for i in range(0, len(countries)):

    geolocator = Nominatim(user_agent='tito', timeout=100)

    location = geolocator.geocode(countries.index[i], timeout=100)

    lat = np.append(lat, location.latitude)

    lon = np.append(lon, location.longitude)

    country = np.append(country, countries.index[i])
#Importing Map

import folium

data = pd.DataFrame({

'lat':lat,

'lon':lon,

'name':country})

data.head()    



m = folium.Map(location=[20, 0], tiles="Mapbox Bright", zoom_start=3 , )

country_map = list(zip(data['name'].values, data['lat'].values, data['lon'].values))

# add features

for country_map in country_map:

    folium.Marker(

        location=[float(country_map[1]), float(country_map[2])],

        popup=folium.Popup(country_map[0], parse_html=True),

        icon=folium.Icon(icon='home')

    ).add_to(m)   

m
geography('Geography','CreditScore')
geography('Geography','EstimatedSalary')
geography('Geography','Balance')
## Correlation with independent Variable 

df2 = df_raw.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)

df2.corrwith(df_raw['Exited']).plot.bar(

        figsize = (10, 10), title = "Correlation with Exited", fontsize = 15,

        rot = 45, grid = True)
sns.set(style="white")

# Compute the correlation matrix

corr = df2.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(10, 10))

# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
## Pie Plots 

df_raw.columns

df2 = df_raw.drop(['RowNumber', 'CustomerId', 'Surname', 'Gender', 

                   'CreditScore', 'Age','Tenure', 'Balance',

                   'EstimatedSalary', 'Exited'], axis=1)

fig = plt.figure(figsize=(20, 20))

plt.suptitle('Pie Chart Distributions', fontsize=20)

for i in range(1, df2.shape[1] + 1):

    plt.subplot(6, 3, i)

    f = plt.gca()

    f.axes.get_yaxis().set_visible(False)

    f.set_title(df2.columns.values[i - 1])

   

    values = df2.iloc[:, i - 1].value_counts(normalize = True).values

    index = df2.iloc[:, i - 1].value_counts(normalize = True).index

    plt.pie(values, labels = index, autopct='%1.1f%%')

    plt.axis('equal')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#Data Analysis

df_raw.describe()    
exited = df_raw['Exited'].value_counts()
countNotExited = len(df_raw[df_raw['Exited'] == 0])     

countExited  = len(df_raw[df_raw['Exited'] == 1]) 

print('Percentage not Exited: {:.2f}%'.format((countNotExited/len(df_raw)) * 100)) 

print('Percentage Exited: {:.2f}%'.format((countExited/len(df_raw)) * 100))
df_raw.groupby(df_raw['Exited']).mean().head()
#Looking for Null Values

sns.heatmap(df_raw.isnull(), yticklabels=False, cbar=False, cmap='viridis')
df_raw.isnull().any()
df_raw.isnull().sum()
null_percentage = (df_raw.isnull().sum()/len(df_raw) * 100)
null_percentage = pd.DataFrame(null_percentage, columns = ['Percentage Null Values (%)'])
null_percentage
#Define X and y

X = df_raw.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)

y = df_raw['Exited']
#Get Dummies

X = pd.get_dummies(X)
#Avoiding Dummies Trap

X.columns

X = X.drop(['Geography_France', 'Gender_Female' ], axis=1)
#Splitting the Dataset into the Training Set and Test Set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=0) 
#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

X_train = pd.DataFrame(sc_x.fit_transform(X_train), columns=X.columns.values)

X_test = pd.DataFrame(sc_x.transform(X_test), columns=X.columns.values)
#Importing Keras libraries e packages

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout
#Initialising the ANN

classifier = Sequential()
#Adding the input layer and the first hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
#Adding the second hidden layer

classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu'))
#Adding the third hidden layer

classifier.add(Dense(output_dim = 24, init = 'uniform', activation = 'relu'))
#Adding the output layer

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Fit classifier to the training test

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
#Predicting the test set result

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5) 
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

acc = accuracy_score(y_test, y_pred)

results = pd.DataFrame([['ANN', acc]],

               columns = ['Model', 'Accuracy'])
results
# Evaluating the ANN

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential

from keras.layers import Dense

def build_classifier():

    classifier = Sequential()

    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

    classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(output_dim = 24, init = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)

accuracies.mean()

accuracies.std()

print("ANN Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))
#### End of Model ####

# Formatting Final Results

df_raw.columns

user_identifier = df_raw['RowNumber']

final_results = pd.concat([y_test, user_identifier], axis = 1).dropna()

final_results['predicted'] = y_pred

final_results = final_results[['RowNumber', 'Exited', 'predicted']].reset_index(drop=True)
final_results.head()