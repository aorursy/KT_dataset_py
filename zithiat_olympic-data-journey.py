# importing libraries for EDA

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb

import matplotlib.pyplot as plt

import random

from scipy.stats import variation

from wordcloud import WordCloud

# importing libraries for logistic regression prediction

import statsmodels.api as sm 

import time

import profile

import random

import math

import scipy 

%pylab inline

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



sb.set()
olympics = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')

olympics = olympics[olympics['Year'] > 1993]

olympics.head(5)
olympics.groupby(['Year'])[['Medal']].count().head(10)
olympics.groupby(['Year'])[['Medal']].count().tail(10)
population = pd.read_csv('../input/world-bank-data-1960-to-2016/country_population.csv')

population.head()
lifeexpectancy = pd.read_csv('../input/world-bank-data-1960-to-2016/life_expectancy.csv')

lifeexpectancy.head()
fertility = pd.read_csv('../input/world-bank-data-1960-to-2016/fertility_rate.csv')

fertility.head()
gdp = pd.read_csv('../input/gdp-world-bank-data/GDP by Country.csv', skiprows = range(0,4))

gdp.head()
happiness2016 = pd.read_csv('../input/world-happiness/2016.csv')

happiness2016.head()
location = pd.read_csv("../input/world-capitals-gps/concap.csv")

location.head()
medaltypes = ['Gold', 'Silver', 'Bronze']

medalpoints = ['Gold', 'Silver', 'Bronze', 'Medals', 'Medal_points']
olympics = pd.concat([olympics, pd.get_dummies(olympics['Medal'])], axis = 1)

olympics = olympics.drop('Medal', axis = 1)

olympics.head()
def prepareMedals(origData, medals):

    medalsDF = origData.groupby(['Year','Season','Team','NOC','Event'])[medals].sum()

    for m in medals:

            medalsDF.loc[medalsDF[m] > 0, m] = 1

    medalsDF.reset_index(inplace = True )

    return medalsDF



medals = prepareMedals(olympics, medaltypes)

medals.head(10)
groupForMedals = ['Year','Season','NOC','Team']

medalsTeams = medals.groupby(groupForMedals)[medaltypes].sum()

print(medalsTeams.head(10))
teamlist = olympics['Team'][olympics['Team'].str.contains("-")].unique() 

display(teamlist)
for i in teamlist:

    # we go back to initial list athletes and remove last 2 chars if the name of the team is in the_list.

    olympics.loc[olympics['Team']==i,'Team']=i[:-2]
medals = prepareMedals(olympics, medaltypes)

medalsTeams = medals.groupby(groupForMedals)[medaltypes].sum()

medalsTeams.reset_index(inplace = True)

print(medalsTeams.head(20))
# The Medal_points column is defined above to store the weigh of achieved medals for each country in each event.

# Let assume:

# - Gold = 3 points

# - Silver = 2 points

# - Bronze = 1 point

# - NaN = 0 point

# The Medal column is for the sum of achieved medal by that team

medalsTeams['Medal_points'] = (3 * medalsTeams['Gold']) + (2 * medalsTeams['Silver']) + medalsTeams['Bronze']

medalsTeams['Medals'] = medalsTeams['Gold'] + medalsTeams['Silver'] + medalsTeams['Bronze']

medalsTeams = medalsTeams.reindex(['Year', 'Season', 'Team', 'NOC', 'Gold', 'Silver', 'Bronze', 'Medals', 'Medal_points'] , axis=1)

display(medalsTeams.head(10))
totallist = medalsTeams.groupby(['Team'])[medalpoints].sum()

totallist.reset_index(inplace = True)

display(totallist.head(10))
medalteams = totallist[totallist['Medal_points'] > 0]

wc = WordCloud(background_color='white', max_words=300, max_font_size=20, colormap='plasma').generate(str(medalteams['Team']))

plt.figure()

plt.imshow(wc, interpolation="bilinear")

plt.axis('off')

plt.show()
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(30, 40), sharex=False)



# First graph to show the top 20 countries from 1994 until 2016 based on the weight of medals.

sb.barplot(data=totallist.sort_values(by='Medal_points').reset_index(drop=True).tail(20), x='Team', y='Medal_points', palette="deep", ax=ax1)

# Iterate through the list of axes' patches

for p in ax1.patches:

    ax1.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), fontsize=12, color='Blue', ha='center', va='bottom')

ax1.axhline(0, color="k", clip_on=True)

ax1.set_ylabel("Medal points")

ax1.set_xlabel("Country")



# Second graph for medal count

sb.barplot(data=totallist.sort_values(by='Medals').reset_index(drop=True).tail(20), x='Team', y='Medals', palette="deep", ax=ax2)

for p in ax2.patches:

    ax2.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), fontsize=12, color='Blue', ha='center', va='bottom')

ax2.axhline(0, color="k", clip_on=True)

ax2.set_ylabel("Medals")

ax2.set_xlabel("Country")



# 3rd graph for gold medals 

sb.barplot(data=totallist.sort_values(by='Gold', ascending = False).head(20), x='Team', y='Gold', palette="deep", ax=ax3)

for p in ax3.patches:

    ax3.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), fontsize=12, color='Blue', ha='center', va='bottom')

ax3.axhline(0, color="k", clip_on=True)

ax3.set_ylabel("Gold")

ax3.set_xlabel("Country")



# 4th graph for silver medals 

sb.barplot(data=totallist.sort_values(by='Silver', ascending = False).head(20), x='Team', y='Silver', palette="deep", ax=ax4)

for p in ax4.patches:

    ax4.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), fontsize=12, color='Blue', ha='center', va='bottom')

ax4.axhline(0, color="k", clip_on=True)

ax4.set_ylabel("Silver")

ax4.set_xlabel("Country")



# 5th graph for bronze medals 

sb.barplot(data=totallist.sort_values(by='Bronze', ascending = False).head(20), x='Team', y='Bronze', palette="deep", ax=ax5)

for p in ax5.patches:

    ax5.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), fontsize=12, color='Blue', ha='center', va='bottom')

ax5.axhline(0, color="k", clip_on=True)

ax5.set_ylabel("Bronze")

ax5.set_xlabel("Country")
countrylist = medalsTeams['Team'].unique()

bestyearofeachcountry = pd.DataFrame(columns=['Team','Year','Medal_points'])

for country in countrylist:

    temp = medalsTeams.loc[medalsTeams['Team']==country].sort_values(by='Medal_points', ascending = False).head(1)[['Team','Year','Medal_points']]

    frames = [bestyearofeachcountry, temp]

    bestyearofeachcountry = pd.concat(frames)
bestyearofeachcountry = bestyearofeachcountry.loc[bestyearofeachcountry['Medal_points']>10]



# The Medal_points is type object, hence we must change it into float or int.

bestyearofeachcountry.loc[:,'Medal_points'] = bestyearofeachcountry.Medal_points.astype(np.float)

g, (ax1) = plt.subplots(1, 1, figsize=(20, 10))

sb.scatterplot(data = bestyearofeachcountry, x = 'Team', y = 'Year', size ='Medal_points', sizes=(10,1000), hue ='Medal_points', palette="Set1", ax=ax1)

ax1.axhline(0, color="k", clip_on=True)

ax1.set(ylim=(1990, 2020))

ax1.set_ylabel("Year")

ax1.set_xlabel("Country")

for item in ax1.get_xticklabels():

    item.set_rotation(90)
medalsTeams.groupby('Season').count()
# define the weigh for the plot

explode = (0.3, 0.2, 0.1, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025)



#Athlete participation by country

summerdata = olympics[olympics.Season == 'Summer'] #focus only on Summer olympics (exclude Winter Olympics)

summercount = summerdata.groupby('NOC').count()

summerdf = pd.DataFrame(summercount, columns = ['Name']) 

summersorted = summerdf.sort_values(['Name'], ascending = False) #Total number of athletes by country



# Medal Count by Top 10 Countries:

summertop = summersorted.head(10)

summertop.sum()



#Plot Top 10 Countries by  Athlete Count: Athletics Only

figure = summertop.plot(kind='pie', figsize=(7,5),subplots=True, explode = explode, autopct='%1.1f%%', legend = None)

plt.axis('equal')

plt.title("Top 10 Countries by  Athlete participating in SUMMER events")

plt.tight_layout()

plt.show()  
#Athlete participation by country

winterdata = olympics[olympics.Season == 'Winter'] #focus only on Winter olympics (exclude Summer Olympics)

wintercount = winterdata.groupby('NOC').count()

winterdf = pd.DataFrame(wintercount, columns = ['Name']) 

wintersorted = winterdf.sort_values(['Name'], ascending = False) #Total number of athletes by country



# Top 10

wintertop = wintersorted.head(10)

wintertop.sum()



# Similar as Summer chart

figure = wintertop.plot(kind='pie', figsize=(7,5),subplots=True, explode = explode, autopct='%1.1f%%', legend = None)

plt.axis('equal')

plt.title("Top 10 Countries by  Athletes participating in WINTER events")

plt.tight_layout()

plt.show()  
eventspergame = pd.DataFrame(olympics.groupby(['Year','Season'])['Event'].nunique())

eventspergame.columns = ['Events']

eventspergame.reset_index(inplace = True)

g, (ax1) = plt.subplots(1, 1, figsize = (20, 5))

sb.lineplot(data = eventspergame,x = 'Year', y = 'Events', hue = 'Season', ax = ax1)
def getbestyear(dataFrame, season = 'All', minimal = 5, value = 'Medal_points', columns = ['Team','Year','Medal_points'], sort_ascending = False):

    dataFrame = dataFrame[dataFrame['Medal_points'] > minimal]

    season = [season]

    res = pd.DataFrame(columns = columns)

    for country in countrylist:

        temp = dataFrame.loc[(dataFrame['Team'] == country) & (dataFrame['Season'].isin(season))].sort_values(by = value, ascending = sort_ascending).head(1)[columns]

        res = pd.concat([res, temp])

    res.loc[:,value] = res[value].astype(np.float)

    return res
def drawbybestyear(dataFrame, season = 'all', value = 'Medal_points', textDivisor = 1):

    season = season.lower()

    g, (ax1) = plt.subplots(1, 1, figsize=(20, 8))

    sb.scatterplot(data = dataFrame, x = 'Team', y = 'Year', size = value, sizes = (5,750), hue = value, alpha = 0.7, edgecolors = "red", palette = season, legend = False, ax = ax1)

    ax1.axhline(0, color="k", clip_on=True)

    yearsTicks = dataFrame['Year'].unique().tolist()

    yearsTicks.sort()

    ax1.set(ylim = (yearsTicks[0] - 1, yearsTicks[-1] + 1), yticks = (yearsTicks))

    ax1.set_ylabel("Year")

    ax1.set_title("Year of best result in " + value, pad = 50, loc = "left")

    i = 0

    prevYear = 0

    for index, row in dataFrame.iterrows():

        ax1.text(i, row['Year'] + 1, round(row[value]/textDivisor, 2), color = 'black', withdash=True)

        i += 1

        prevYear = row['Year']

    for item in ax1.get_xticklabels():

        item.set_rotation(90)
bestyearSummer = getbestyear(medalsTeams, "Summer")

drawbybestyear(bestyearSummer, 'Summer')
bestyearWinter = getbestyear(medalsTeams, "Winter")

drawbybestyear(bestyearWinter, 'Winter')
# For Summer

summerCountries = pd.DataFrame(columns=['Team','Game'])

summerCountries['Team'] = bestyearSummer['Team']

summerCountries['Game'] = 1

# For Winter

winterCountries = pd.DataFrame(columns = ['Team','Game'])

winterCountries['Team'] = bestyearWinter['Team']

winterCountries['Game'] = 2

# Merge Summer and Winter

wsCountries = pd.concat([winterCountries, summerCountries])

wsCountries = wsCountries.groupby(['Team'])['Game'].sum().to_frame()

wsCountries.reset_index(inplace = True)

# Now we plot the merged dataframe

g, (ax1) = plt.subplots(1, 1, figsize = (20, 7))

sb.scatterplot(data = wsCountries, x = 'Team', y = 'Game', s = 1000, ax = ax1, alpha = 0.5, edgecolors = "gray", linewidth = 1, palette="Set2")

ax1.axhline(0.6, clip_on = True)

ax1.set(ylim = (0.6, 3.4), yticks = (1, 2, 3), yticklabels = ('Summer','Winter', 'Both'))

ax1.set_ylabel("Olympic season")

for item in ax1.get_xticklabels():

    item.set_rotation(90)
# create dataframe with unique participants grouped by year, season and team

participants = pd.DataFrame(olympics.groupby(['Year','Season','Team'])['ID'].nunique())

participants.columns = ['UniqueParticipants']

participants.reset_index(inplace=True)

#now we merge it with dataframe we worked with (until now that is)

medalsTeamsParticipants = medalsTeams.merge(participants, on=['Year','Season','Team'])

medalsTeamsParticipants.head()
def resultsallyears(tmpmedalsteams, season):

    season = season[0].upper() + season.lower()[1:]

    if season!='All':

        tmpmedalsteams = tmpmedalsteams[tmpmedalsteams['Season'] == season]

    tmplistcountries = np.unique(tmpmedalsteams.loc[tmpmedalsteams['Medal_points'] > 5, ['Team']].values)

    # dataframe for the list

    numcountries = pd.DataFrame(tmplistcountries)

    numcountries.columns = ['Team']

    # Add index column

    numcountries['Id'] = numcountries.index

    #Merge two dataframes 

    tmpmedalsteams = tmpmedalsteams.loc[tmpmedalsteams['Team'].isin(tmplistcountries)]

    tmpmedalsteams = tmpmedalsteams.merge(numcountries, on = 'Team')

    return tmpmedalsteams
def computecoeffvars(dataFrame):

    coeffvardf = pd.DataFrame(columns=['Team', 'Coeff'])

    countries = dataFrame['Team'].unique()

    for c in countries:

        points = dataFrame.loc[dataFrame['Team'] == c]['Medal_points']

        coef = variation(points)

        #add values to dataframe

        coeffvardf = coeffvardf.append({'Team':c, 'Coeff':coef}, ignore_index=True) 

    return coeffvardf
def drawcoeffvar(dataFrame, palette): # palette for the graph

    maxim = dataFrame.sort_values(by = 'Coeff', ascending = False)['Coeff'].head(1)

    maxim = round((maxim.values[0] * 1.1), 2)

    palette = palette.lower()

    g, (ax1) = plt.subplots(1, 1, figsize = (20, 5))

    sb.barplot(data = dataFrame, x = 'Team', y = 'Coeff', palette = palette, ax = ax1)

    ax1.axhline(0, color = "k", clip_on=True)

    ax1.set(ylim = (0, maxim))

    ax1.set_ylabel("Coefficient of variation")

    #writting the datalabels

    i = -0.25

    prevValue = 0

    for index, row in dataFrame.iterrows():

        value = round(row['Coeff'], 2) #this is indendent to avoid text over text

        if abs(1 - prevValue / value) < 0.1:

            position = value + (maxim * 0.05)

        else:

            position = value

        prevValue = position

        ax1.text(i, position, value, color = 'black', withdash = True)

        i += 1

    for item in ax1.get_xticklabels():

        item.set_rotation(90)
drawcoeffvar(computecoeffvars(resultsallyears(medalsTeams, 'Summer')) ,'Summer')
drawcoeffvar(computecoeffvars(resultsallyears(medalsTeams, 'Winter')) ,'Winter')
years = np.sort(olympics['Year'].unique())

print(str(years).split())
years_string=str(years)[1:-1].split()

# In the GDP and Population, the Country is under 'Country Name'

# We will get rid of 'Country Name' later after we merge with the Olympics data

# For now, we leave it to filter the list.

cntrname = 'Country Name'

years_string.insert(0, cntrname)

print(years_string)
populationdata = population[years_string]

display(populationdata.head())
gdpdata = gdp[years_string]

display(gdpdata.head())
olympic_countries = olympics['Team'].unique()

gdp_countries = gdpdata['Country Name'].unique()
np.setdiff1d(olympic_countries, gdp_countries)
np.setdiff1d(gdp_countries, olympic_countries)
change = {

'Bahamas, The':'Bahamas', 

'Cabo Verde':'Cape Verde',

'Congo, Rep.':'Congo (Brazzaville)',

'Russian Federation':'Russia',

'St. Vincent and the Grenadines':'Saint Vincent and the Grenadines',

'Venezuela, RB':'Venezuela',

'Congo, Dem. Rep.':'Congo (Kinshasa)',

'Micronesia, Fed. Sts.':'Federated States of Micronesia',

'Gambia, The':'Gambia',

'Guinea-Bissau':'Guinea Bissau',

'Iran, Islamic Rep.':'Iran',

'St. Kitts and Nevis':'Saint Kitts and Nevis',

'Slovak Republic':'Slovakia',

'Syrian Arab Republic':'Syria',

'Hong Kong SAR, China':'Hong Kong',

'Kyrgyz, Republic':'Kyrgyzstan',

'Macedonia, FYR':'Macedonia',

'Korea, Dem. People’s Rep.':'North Korea',

'St. Lucia':'Saint Lucia',

'Timor-Leste':'Timor Leste',

'Brunei Darussalam':'Brunei',

'Egypt, Arab Rep.':'Egypt',

'United Kingdom':'Great Britain',

'Korea, Rep.':'South Korea',

'Virgin Islands (U.S.)':'United States Virgin Islands',

'Yemen, Rep.':'Yemen'}



gdpmelted = gdpdata.melt(id_vars = ['Country Name'], value_vars = years_string[1:], var_name = 'Years')

gdpmelted.columns = ['Team','Year','GDP']

gdpmelted['Country'] = gdpmelted['Team']

gdpmelted.head()

# Loop to apply the changes

for key in change:

    value = change[key]

    gdpmelted.loc[gdpmelted['Team'] == key,'Team'] = value

    

# we need to convert the year back to int.

gdpmelted['Year'] = gdpmelted['Year'].apply(int)

gdpmelted.head()
combineddata = pd.merge(medalsTeamsParticipants, gdpmelted, on = ['Team','Year'])

combineddata.drop(['Country'], axis = 1, inplace = True)

combineddata.head()
populationmelted = populationdata.melt(id_vars = ['Country Name'], value_vars = years_string[1:], var_name = 'Years')

populationmelted.columns = ['Team','Year','Population']

populationmelted['Year'] = populationmelted['Year'].apply(int)

populationmelted.head()
combineddata = pd.merge(combineddata, populationmelted, on = ['Team','Year'])

combineddata.head()
combineddata['MedalsPerPopulation'] = combineddata['Medals']/(combineddata['Population']/100000)

combineddata['MedalPointsPerPopulation'] = combineddata['Medal_points']/(combineddata['Population']/100000)

combineddata['GDPWorthPerMedal'] = (combineddata['GDP']/combineddata['Medals'])

combineddata['GDPWorthPerMedalPoints'] = (combineddata['GDP']/(combineddata['Medal_points']))

combineddata['ParticipantsPerPopulation'] = combineddata['UniqueParticipants']/(combineddata['Population']/100000)

combineddata.head()
# This is to safer use, in case we make any messy.

working_data = combineddata.copy()
summerbest = getbestyear(combineddata, "Summer", 10, 'GDPWorthPerMedalPoints', ['Team', 'Year', 'GDPWorthPerMedalPoints'], True)

drawbybestyear(summerbest, 'Summer', 'GDPWorthPerMedalPoints', 1000000000)
winterbest = getbestyear(combineddata, "Winter", 10, 'GDPWorthPerMedalPoints', ['Team', 'Year', 'GDPWorthPerMedalPoints'], True)

drawbybestyear(winterbest, 'Winter', 'GDPWorthPerMedalPoints', 1000000000)
def comparisongraph(country, palette, data1, datalb1, data2, datalb2, d1div = 1, d2div = 1):

    tempData = resultsallyears(combineddata, 'All')

    dataFrame = tempData.loc[tempData['Team'] == country]

    palette = palette.lower()

    g, ax1 = plt.subplots(figsize = (20, 7))

    sb.barplot(dataFrame['Year'].apply(lambda x: str(x)), dataFrame[data1].apply(lambda x: x / d1div), ax = ax1, palette = palette)

    ax1.axhline(0, color = "k", clip_on = True)

    ax2 = ax1.twinx()

    sb.lineplot(dataFrame['Year'].apply(lambda x: str(x)), dataFrame[data2].apply(lambda x: x / d2div), linewidth = 4, marker = 's', markersize = 12, ax = ax2)

    ax2.grid(False)

    ax1.set_title('Comparison of ' + datalb1 + ' and ' + datalb2 + ' through years for ' + country)

    ax1.set_ylabel(datalb1)

    ax2.set_ylabel(datalb2)   
comparisongraph("Germany", "seismic", "Medal_points", "Medal points", "GDP", "GDP (billion USD)", 1, 1000000000)
comparisongraph("United States", "Rocket", "Medal_points", "Medal points", "GDP", "GDP (billion USD)", 1, 1000000000)
comparisongraph("China", "deep", "Medal_points", "Medal points", "GDP", "GDP (billion USD)", 1, 1000000000)
combineddata = working_data.copy()

summermostparticipants = getbestyear(combineddata, "Summer", 5, 'ParticipantsPerPopulation', ['Team', 'Year', 'ParticipantsPerPopulation'], False)

drawbybestyear(summermostparticipants, 'Summer', 'ParticipantsPerPopulation')
summermostparticipants = getbestyear(combineddata, "Winter", 5, 'ParticipantsPerPopulation', ['Team', 'Year', 'ParticipantsPerPopulation'], False)

drawbybestyear(summermostparticipants, 'Winter', 'ParticipantsPerPopulation')
shortlistofsportycountries = ['Latvia','Slovenia', 'New Zealand']

for c in shortlistofsportycountries:

    comparisongraph(c, "deep", "UniqueParticipants", "Unique participants", "Population", "Population (in millions)", 1, 1000000)
summerdata = combineddata.copy()

summerdata = summerdata[summerdata['Season'] == 'Summer']

summerdata.head()
summerdata["Team"] = summerdata["Team"].astype('category')

summerdata.dtypes
summerdata["Team_Num"] = summerdata["Team"].cat.codes

summerdata.head()
summerdata["GDP"].fillna(0, inplace = True)

summerdata["Population"].fillna(0, inplace = True)

summerdata["UniqueParticipants"].fillna(0, inplace = True)
summerdata.isnull().sum()
summerdata['Winner'] = summerdata.Medal_points.apply(lambda x: 1 if x > 0 else 0) 

summerdata.head()
Independent_var = summerdata.iloc[:, [8,9,10,11,17]].values  

Dependent_var = summerdata.iloc[:, 18].values 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Independent_var, Dependent_var, test_size = 0.2, random_state = 10)  
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier



#Create KNN Classifier

knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets

knn.fit(X_train, y_train)

#Predict the response for test dataset

y_pred = knn.predict(X_test)



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report, confusion_matrix  

print(confusion_matrix(y_test, y_pred))  

print(classification_report(y_test, y_pred))  
error = []



# Calculating error for K values between 1 and 40

for i in range(1, 40):  

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error.append(np.mean(pred_i != y_test))



plt.figure(figsize=(12, 6))  

plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  

         markerfacecolor='blue', markersize=10)

plt.title('Error Rate K Value')  

plt.xlabel('K Value')  

plt.ylabel('Mean Error')  
#Import Random Forest Model

from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)



y_pred=clf.predict(X_test)



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier



# Create Decision Tree classifer object

clf = DecisionTreeClassifier(criterion="entropy", max_depth=20)



clf = clf.fit(X_train, y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.linear_model import LinearRegression  

from sklearn.neural_network import MLPClassifier  

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier  

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import metrics

import matplotlib.pyplot as plt



plt.figure()



# Add the models to the list that you want to view on the ROC plot

models = [

{

    'label': 'Random Forest',

    'model': RandomForestClassifier(n_estimators=100),

},

{

    'label': 'Decision Tree',

    'model': DecisionTreeClassifier(max_depth=20)  

}

    ,

    {  

     'label': 'Neural Network',

     'model':MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000) 

    }

     ,

    {

     'label': 'Logistic Regression',

    'model': LogisticRegression(),

    }

]



# Below for loop iterates through your models list

for m in models:

    model = m['model'] # select the model

    model.fit(X_train, y_train) # train the model

    y_pred=model.predict(X_test) # predict the test data

# Compute False postive rate, and True positive rate

    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:,1])

# Calculate Area under the curve to display on the plot

    auc = metrics.roc_auc_score(y_test,model.predict(X_test))

# Now, plot the computed values

    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))

# Custom settings for the plot 

#plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('1-Specificity(False Positive Rate)')

plt.ylabel('Sensitivity(True Positive Rate)')

plt.title('Receiver Operating Characteristic')

plt.legend(loc="lower right")

plt.show()   # Display
# We skip autosklearn for now as we need to deal with some packages installation

#import autosklearn.classification

#import sklearn.model_selection

#import sklearn.datasets

#import sklearn.metrics

#X, y = sklearn.datasets.load_digits(return_X_y=True)

#X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

#automl = autosklearn.classification.AutoSklearnClassifier()

#automl.fit(X_train, y_train)

#y_hat = automl.predict(X_test)

#print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))



from tpot import TPOTClassifier

from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split



digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.75, test_size=0.25)



pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,

                                    random_state=42, verbosity=2)

pipeline_optimizer.fit(X_train, y_train)

print(pipeline_optimizer.score(X_test, y_test))