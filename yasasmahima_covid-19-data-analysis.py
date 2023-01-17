import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error

import calmap
import folium
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
# Get Data From CSV File
data=pd.read_csv("D:/Projects/Covid19/DataSet/covid_19_data.csv",parse_dates=['Last Update'])
# Rename Columns
data.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)
# Display Data
data.head()
data2= pd.read_csv('D:\Projects\Covid19\DataSet\covid_19_clean_complete.csv', 
                         parse_dates=['Date'])
data2.head()
data.shape
# Check for Data Types
data.dtypes
# Check for Null Values
data.isnull().sum().to_frame('nulls')
map_data= data2[data2['Date'] == max(data2['Date'])].reset_index()
# World Map from folium Mao
world_map = folium.Map(location=[0, 0], 
               min_zoom=1, max_zoom=4, zoom_start=1)

# Add data to World Map
for i in range(0, len(map_data)):
    folium.Circle(
        location=[map_data.iloc[i]['Lat'], map_data.iloc[i]['Long']],
        color='red', 
        tooltip =   '<li><bold>Country : '+str(map_data.iloc[i]['Country/Region'])+
                    '<li><bold>Province : '+str(map_data.iloc[i]['Province/State'])+
                    '<li><bold>Confirmed : '+str(map_data.iloc[i]['Confirmed'])+
                    '<li><bold>Deaths : '+str(map_data.iloc[i]['Deaths'])+
                    '<li><bold>Recovered : '+str(map_data.iloc[i]['Recovered']),
        radius=(map_data.iloc[i]['Confirmed'])**1.1).add_to(world_map)
#     Display Map
world_map
world_map.save("map1.html")
# Extract Data from the data set
processedData = data2.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()
processedData = processedData.reset_index()
# Add date as the changing variable
processedData['Date'] = pd.to_datetime(processedData['Date'])
processedData['Date'] = processedData['Date'].dt.strftime('%m/%d/%Y')
processedData['size'] = processedData['Confirmed'].pow(0.3)

# Add data to the figure
figure = px.scatter_geo(processedData,locations="Country/Region", locationmode='country names', 
                     color="Confirmed", size='size', hover_name="Country/Region", 
                     range_color= [0, max(processedData['Confirmed'])+2], 
                     projection="natural earth", animation_frame="Date" ,
                     
                     )
# Display Figure
figure.update(layout_coloraxis_showscale=False)
figure.show()
figure.write_html("map2.html")
dataFrame = data.groupby(["Date", "Country"])[['Date', 'Country', 'Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
# Get All the Confiremd Cases all arounfd the world
Confirmed_patients=dataFrame.sort_values('Confirmed',ascending=False)
Confirmed_patients=Confirmed_patients.drop_duplicates('Country')

world_Total_Confirmed=Confirmed_patients['Confirmed'].sum()  #Total Confirmed Patients
world_Total_Deaths=Confirmed_patients['Deaths'].sum()        #Total Deaths
world_Total_Recovered=Confirmed_patients['Recovered'].sum()  #Total Recovered Patients
 
world_Deaths_Rate=(world_Total_Deaths*100)/world_Total_Confirmed           #Get Death Precentage of the World
world_Recovered_rate=(world_Total_Recovered*100)/world_Total_Confirmed    #Get Recovered Precentage in the World

China=Confirmed_patients[Confirmed_patients['Country']=='Mainland China']        #Get Confirmed Patients precentage in china
China_Recovered_rate=(int(China['Recovered'].values)*100)/int(China['Confirmed'].values)  #Get Recovered Patients precentage in china

Italy=Confirmed_patients[Confirmed_patients['Country']=='Italy']        #Get confirmed Patients Precentage in Italy
Italy_Recovered_rate=(int(Italy['Recovered'].values)*100)/int(Italy['Confirmed'].values)  #GEt Recovered Patients Precentage in Italy 

SriLanka=Confirmed_patients[Confirmed_patients['Country']=='Sri Lanka']        #Get Confirmed Patients Percentage in sri lanka
SriLanka_Recovered_rate=(int(SriLanka['Recovered'].values)*100)/int(SriLanka['Confirmed'].values)  #Get Recovered patients Precentage in Sri Lanka

#Add Values to the Table and Display Table
Table={'Total Confirmed Patients in the World : ':world_Total_Confirmed,'Total Deaths Confirmed in the world : ':world_Total_Deaths,'Total Recovered Patients in the world : ':world_Total_Recovered,'Rate of Recovered Patients(Precentage) :':world_Recovered_rate,'Rate of Death Patients(Precentage) :':world_Deaths_Rate,
      'Rate of Recovered China cases(Presentage) :':China_Recovered_rate,'Rate of Recovers Italy Cases(Precentage) : ':Italy_Recovered_rate,'Rate of Recovers in Sri Lankan Cases(Precentage) : ':SriLanka_Recovered_rate}
Table=pd.DataFrame.from_dict(Table,orient='index',columns=['Total'])

Table.style.background_gradient(cmap='Reds')
# Display Total Confirmed Patients/Deaths and Recovered
Graph=Table.head(3)
x=Graph.index
y=Graph['Total'].values
plt.rcParams['figure.figsize'] = (11,6)
sns.barplot(x,y,order=x ).set_title('Covid 19 Data Analysis') 
plt.savefig('dataanalysis.png')

Recovered_rate=(Confirmed_patients['Recovered']*100)/Confirmed_patients['Confirmed']  #Get Recovered Precentage
Deaths_rate=(Confirmed_patients['Deaths']*100)/Confirmed_patients['Confirmed']  #Get Death Precentage
cases_rate=(Confirmed_patients.Confirmed*100)/world_Total_Confirmed    #Get Total Patients Confirmed

#Set Rated to the Table
Confirmed_patients['Recovered Patients Rate']=pd.DataFrame(Recovered_rate)  
Confirmed_patients['Deaths Patients Rate']=pd.DataFrame(Deaths_rate)
Confirmed_patients['Total Patients Rate']=pd.DataFrame(cases_rate)

# Display Table
Confirmed_patients.head(100).style.background_gradient(cmap='Blues')
# Function for get Each Contry's Covid-19 Active and confirmed and recoverd cases

def casesInEachCountry(country):
    data_of_country = data[data['Country']==country]  #Get Data of the given Country
    table = data_of_country.drop(['SNo','Province/State','Last Update'], axis=1)  #Drop unwanted Columns
    table['ActiveCases'] = table['Confirmed'] - table['Recovered'] - table['Deaths']  #Calculate Active Cases in the country
#     Display ActiveCases , Confiremd,recovered and Deaths in each country
    graph = pd.pivot_table(table,values=['ActiveCases','Confirmed', 'Recovered','Deaths'],index=['Date'], aggfunc=np.sum)
    
    
    return  graph.plot().set_title(country+" Covid 19 Data Analysis") 
graph=casesInEachCountry('Mainland China')
plt.savefig("china.png")
graph
#  casesInEachCountry('Italy')
graph=casesInEachCountry('US')
plt.savefig("us.png")
graph
#  casesInEachCountry('Italy')
graph=casesInEachCountry('Italy')
plt.savefig("Italy.png")
graph
#  casesInEachCountry('Sri Lanka')
graph=casesInEachCountry('Sri Lanka')
plt.savefig("lanka.png")
graph
# casesInEachCountry('Iran')
graph=casesInEachCountry('Spain')
plt.savefig("spian.png")
graph
# casesInEachCountry('Iran')
graph=casesInEachCountry('Iran')
plt.savefig("Iran.png")
graph
# Effected Countries- Sorted by Number of Cases (TOP 20)
sorted_By_NumberofCases=Confirmed_patients.head(20)
x=sorted_By_NumberofCases.Country
y=sorted_By_NumberofCases.Confirmed
plt.rcParams['figure.figsize'] = (20, 10)
fig=sns.barplot(x,y,order=x ,palette="rocket").set_title('Top 20 Affected Countries')
plt.savefig("dta2.png")
fig
cases_per_Day = data.groupby(["Date"])['Confirmed','Deaths', 'Recovered'].sum().reset_index()
data_table=cases_per_Day.sort_values('Date',ascending=False)

data_table.style.background_gradient(cmap='Blues')
# data_table.set_title('Increasing of the Virus')

x=cases_per_Day.index

y=cases_per_Day.Confirmed
y1=cases_per_Day.Deaths
y2=cases_per_Day.Recovered

plt.plot(x,y,color='blue',label='Confirmed Patients')
plt.plot(x,y1,color='red' ,label="Deaths Patients")
plt.plot(x,y2,color='green',label="Recovered Patients")
print("Blue : Confirmed Cases ")
print("Red : Deaths Cases ")
print("Green : Recovered Cases ")
plt.xlabel("Date_Range")
plt.ylabel("Number of cases")
plt.title("World Covid-19 Increment")
plt.legend()
plt.savefig("world.png")
plt.show()
x_data=pd.DataFrame(cases_per_Day.index)
y_data=pd.DataFrame(cases_per_Day.Confirmed)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.3,random_state=0)
poly_reg=PolynomialFeatures(degree=7)
x_poly=poly_reg.fit_transform(x_train)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y_train)
plt.scatter(x,y,color='red',label='Acctual Confirmed Cases')
plt.scatter(x_test,lin_reg2.predict(poly_reg.fit_transform(x_test)),color='blue',label='Predicted Cases')
# plt.title("Polynomial Regression Model ")
plt.xlabel("Date_Range")
plt.ylabel("Number of cases")
plt.legend()
plt.savefig("prediction.png")
plt.show()
# Accuracy of the Polynomial regression Model Accuracy
y_pred=lin_reg2.predict(poly_reg.fit_transform(x_test))
print('Accuracy of the Polynomial Regession Model  : ',r2_score(y_test, y_pred))
# Display Graph Function
def displayGraph(X_test,y_test,y_pred):
    plt.scatter(X_test,y_test,color="blue",label="Acctual Cases")
    plt.scatter(X_test,y_pred, color='red',label="Predicted Cases")
    plt.xlabel("Date_Range")
    plt.ylabel("Number of Cases")
    plt.legend()
    plt.show()
# Check Some Algorithms For Prediction
algorithms = []
algorithms.append(('LinearRegression', LinearRegression()))  #Linear Regression
algorithms.append(('BaggingRegressor', BaggingRegressor()))  #Bagging REgressor
algorithms.append(('RandomForest', RandomForestRegressor())) #Random Forest Tree
algorithms.append(('KNeighbours', KNeighborsRegressor()))    #K Neighbours 


# Evaluations
results = []
names = []


for name,model in algorithms:
    
#     Fit data to the model
    model.fit(x_train,y_train)
    
#     Predict data
    predictions = model.predict(x_test)
    
#     Get Varice
    variance = explained_variance_score(y_test, predictions)
#     Get mean absolute error
    meanError = mean_absolute_error(predictions,y_test)
    results.append(meanError)
    names.append(name) 
#     Display result
    result = "%s: %f (%f)" % (name,variance, meanError)
    print(result)
    
    displayGraph(x_test,y_test,predictions)
#     model.save(name.h5)
