import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import random

from scipy.stats import variation

%matplotlib inline

import seaborn as sns

sns.set()
listOfMedals = ['Gold','Silver','Bronze']

listOfMedalsPointsSum = ['Gold','Silver','Bronze','Medal_pts','Medals']

groupingOfMedals = ['Year','Season','Team']
#athletes = pd.read_csv("athlete_events.csv")

athletes = pd.read_csv("../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv")

athletes.head(10)

#import os

#print(os.listdir("../input/")) # there will be subfolder
athletes = pd.concat([athletes, pd.get_dummies(athletes['Medal'])], axis = 1)

athletes = athletes.drop('Medal', axis = 1)

athletes.head()
athletes.describe()
athletes.info()
athletes1994 = athletes[athletes['Year']>1993]
athletes1994.head()
# I've created this as function since I know will use it multiple times since this isn't the first version :)

def prepareMedals(basicData, listOfMedals):

    medalsDF = basicData.groupby(['Year','Season','Team','Event'])[listOfMedals].sum()

    for m in listOfMedals:

            medalsDF.loc[medalsDF[m] > 0, m] = 1

    medalsDF.reset_index(inplace = True )

    return medalsDF
medals = prepareMedals(athletes1994, listOfMedals)

medals.head(25)
medalsTeams = medals.groupby(groupingOfMedals)[listOfMedals].sum()

print(medalsTeams.head(5))
# this is list of all the "duplicated" teams (Like "Austria-1")

the_list = athletes1994['Team'][athletes1994['Team'].str.contains("-")].unique() 

display(the_list)
for i in the_list:

    # we go back to initial list athletes and remove last 2 chars if the name of the team is in the_list.

    athletes1994.loc[athletes1994['Team']==i,'Team']=i[:-2]

for i in the_list:

    # this is actually optional since this is the first list with old records.

    athletes.loc[athletes['Team']==i,'Team']=i[:-2]
# this is repeated code, just for checking if we merged the teams.

medals = prepareMedals(athletes1994, listOfMedals)

medalsTeams = medals.groupby(groupingOfMedals)[listOfMedals].sum()

medalsTeams.reset_index(inplace = True)

print(medalsTeams.head(20))
medals.to_csv('medalje.csv')

medalsTeams.to_csv('medalsTeam.csv')
medalsTeams['Medal_pts'] = (3*medalsTeams['Gold'])+(2*medalsTeams['Silver'])+medalsTeams['Bronze']

medalsTeams['Medals'] = medalsTeams['Gold']+medalsTeams['Silver']+medalsTeams['Bronze']

medalsTeams.head(10)
medalsTeamsTotals = medalsTeams.groupby(['Team'])[listOfMedalsPointsSum].sum()

medalsTeamsTotals.reset_index(inplace = True)

medalsTeamsTotals.head(10)
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(15, 20), sharex=False)



#The first graph is who has the most medal points (weighted sum of medals)

sns.barplot(data=medalsTeamsTotals.sort_values(by='Medal_pts', ascending = False).head(10), x='Team', y='Medal_pts', palette="rocket", ax=ax1)

ax1.axhline(0, color="k", clip_on=True)

ax1.set_ylabel("Medal points")

#The second graph is who has the most medals 

sns.barplot(data=medalsTeamsTotals.sort_values(by='Medals', ascending = False).head(10), x='Team', y='Medals', palette="rocket", ax=ax2)

ax2.axhline(0, color="k", clip_on=True)

ax2.set_ylabel("Medals")

#The third graph is who has the most gold medals 

sns.barplot(data=medalsTeamsTotals.sort_values(by='Gold', ascending = False).head(10), x='Team', y='Gold', palette="rocket", ax=ax3)

ax3.axhline(0, color="k", clip_on=True)

ax3.set_ylabel("Gold")

#The fourth graph is who has the most silver medals 

sns.barplot(data=medalsTeamsTotals.sort_values(by='Silver', ascending = False).head(10), x='Team', y='Silver', palette="rocket", ax=ax4)

ax4.axhline(0, color="k", clip_on=True)

ax4.set_ylabel("Silver")

#The fifth graph is who has the most bronze medals 

sns.barplot(data=medalsTeamsTotals.sort_values(by='Bronze', ascending = False).head(10), x='Team', y='Bronze', palette="rocket", ax=ax5)

ax5.axhline(0, color="k", clip_on=True)

ax5.set_ylabel("Bronze")

listOfCountries = medalsTeams['Team'].unique()

bestyears = pd.DataFrame(columns=['Team','Year','Medal_pts'])

for country in listOfCountries:

    temp = medalsTeams.loc[medalsTeams['Team']==country].sort_values(by='Medal_pts', ascending = False).head(1)[['Team','Year','Medal_pts']]

    frames = [bestyears, temp]

    bestyears = pd.concat(frames)
bestyears = bestyears.loc[bestyears['Medal_pts']>5]

#since Medal_pts is type object we must change it into float or int.

bestyears.loc[:,'Medal_pts'] = bestyears.Medal_pts.astype(np.float)

g, (ax1) = plt.subplots(1, 1, figsize=(20, 5))

sns.scatterplot(data = bestyears, x = 'Team', y = 'Year', size ='Medal_pts', sizes=(5,1000) , hue ='Medal_pts' ,palette="coolwarm", ax=ax1)

ax1.axhline(0, color="k", clip_on=True)

ax1.set(ylim=(1992, 2018))

ax1.set_ylabel("Year")

for item in ax1.get_xticklabels():

    item.set_rotation(90)
EventsPerGames=pd.DataFrame(athletes1994.groupby(['Year','Season'])['Event'].nunique())

EventsPerGames.columns=['Events']

EventsPerGames.reset_index(inplace = True)

g, (ax1) = plt.subplots(1, 1, figsize=(20, 5))

sns.lineplot(data=EventsPerGames,x='Year',y='Events', hue='Season', ax=ax1)
#dataframe to get flexibility if we want to differently do things

#Season: Winter / Summer / All

#minimal to drop rows that in best years do not have more then "minimal" number of medals points

def returnDFOfBestYears(dataFrame, season = 'All', minimal = 5, value = 'Medal_pts', columns = ['Team','Year','Medal_pts'], sort_ascending = False):

    dataFrame=dataFrame[dataFrame['Medal_pts']>minimal]

    #Make sure season is with capital first letter

    season = season[0].upper() + season.lower()[1:]

    # define columns that will be used in all cases (a bit less typing)

    if season == 'All':

        season = dataFrame['Season'].unique()

    else:

        season = [season]

    #create empty dataframe

    bestyearsTemp = pd.DataFrame(columns=columns)

    #list of countries already exists

    for country in listOfCountries:

        temp = dataFrame.loc[(dataFrame['Team']==country) & (dataFrame['Season'].isin(season))].sort_values(by=value, ascending = sort_ascending).head(1)[columns]

        bestyearsTemp = pd.concat([bestyearsTemp, temp])

    #since Medal_pts is type object we must change it into float or int.

    bestyearsTemp.loc[:,value] = bestyearsTemp[value].astype(np.float)

    return bestyearsTemp



#dataframe: what we draw

#season actually transfers to palette since I like it when Winter Games use winter palette ;)

#value we mighty do graphs on different values :)

#textDivisor is value we divide the value for labels to avoid big numbers (like in case of GDP)

def drawGraphBestYears(dataFrame, season = 'all', value = 'Medal_pts', textDivisor = 1):

    season = season.lower()

    if season == 'all':

        season = 'coolwarm'

    g, (ax1) = plt.subplots(1, 1, figsize=(20, 8))

    # color palette for the graph "winter" :)

    sns.scatterplot(

                    data = dataFrame

                    , x = 'Team'

                    , y = 'Year'

                    , size = value

                    , sizes=(5,750)

                    , hue = value

                    , alpha=0.7

                    , edgecolors="red"

                    , palette=season

                    , legend = False

                    , ax=ax1)

    ax1.axhline(0, color="k", clip_on=True)

    # lets define max and min of y axis based on actual data, and set ticks on actual data

    yearsTicks = dataFrame['Year'].unique().tolist()

    yearsTicks.sort()

    ax1.set(ylim=(yearsTicks[0]-1, yearsTicks[-1]+1),yticks = (yearsTicks))

    ax1.set_ylabel("Year")

    ax1.set_title("When a team had the best results in " + value, pad = 25, loc = "left" )

    i = 0

    prevYear = 0

    for index, row in dataFrame.iterrows():

        #to avoid overlaping of datalabels

        if prevYear == row['Year']:

            addition = 2.5

        else:

            addition = 1.5

        ax1.text(i, row['Year']+addition, round(row[value]/textDivisor,2), color='black', withdash=True)

        i += 1

        prevYear = row['Year']

    for item in ax1.get_xticklabels():

        item.set_rotation(90)
bestyearsW = returnDFOfBestYears(medalsTeams, "winter")

drawGraphBestYears(bestyearsW, 'winter')
bestyearsS = returnDFOfBestYears(medalsTeams, "summer")

drawGraphBestYears(bestyearsS, 'summer')
# First create dataframe with two columns: Team - list of teams, and Game - value 1 for summer games

summerCountries = pd.DataFrame(columns=['Team','Game'])

summerCountries['Team'] = bestyearsS['Team']

summerCountries['Game'] = 1

# First create dataframe with two columns: Team - list of teams, and Game - value 2 for winter games

winterCountries = pd.DataFrame(columns=['Team','Game'])

winterCountries['Team'] = bestyearsW['Team']

winterCountries['Game'] = 2

#now we concatenate two dataframes

wsCountries = pd.concat([winterCountries, summerCountries])

#and now we group per team and sum values of game

#In case a team had more then 5 medals in summer value of Game is 1, in case a team had

#more then 5 medals in winter value of Game is 2, and if a team had more then 5 medals

#in both summer and winter the value is 3.

wsCountries = wsCountries.groupby(['Team'])['Game'].sum().to_frame()

wsCountries.reset_index(inplace = True)

g, (ax1) = plt.subplots(1, 1, figsize=(20, 7))

sns.scatterplot(data = wsCountries, x = 'Team', y = 'Game', s=1000, ax=ax1, cmap="Blues", alpha=0.4, edgecolors="gray", linewidth=2)

ax1.axhline(0.6, color="k", clip_on=True)

ax1.set(ylim=(0.6, 3.4),yticks = (1,2,3), yticklabels = ('Only summer','Only winter', 'Both'))

ax1.set_ylabel("Olympic season")

for item in ax1.get_xticklabels():

    item.set_rotation(90)

medalsTeamsTotalsW = medalsTeams.loc[medalsTeams['Season']=='Winter'].groupby(['Team'])[listOfMedalsPointsSum].sum()

medalsTeamsTotalsW.reset_index(inplace = True)

print(medalsTeamsTotalsW.head(10))

medalsTeamsTotalsS = medalsTeams.loc[medalsTeams['Season']=='Summer'].groupby(['Team'])[listOfMedalsPointsSum].sum()

medalsTeamsTotalsS.reset_index(inplace = True)

print(medalsTeamsTotalsS.head(10))
rows = 5

columns = 2

f, (ax1) = plt.subplots(rows, columns, figsize=(15, 30), sharex=False)



for x in range(rows):

    for y in range(columns):

        i = -0.25

        # dictionary containing options for labels of y axis - based on row number

        ylabelChoices = {0: ("Medal points"), 1: ("Medals"), 2: ("Gold"), 3: ("Silver"), 4: ("Bronze")}

        # dictionary containing options for sorting data - based on row number

        sortChoices = {0: ("Medal_pts"), 1: ("Medals"), 2: ("Gold"), 3: ("Silver"), 4: ("Bronze")}

        # dictionary of pallets which reflect season - based on column number

        palletChoices = {0: ("winter"), 1: ("summer")}

        # dictionary of variables from which we get winter / summer data - based on column number

        sourceChoices = {0: (medalsTeamsTotalsW), 1: (medalsTeamsTotalsS)}

        #the data for this graph

        dataFrame = sourceChoices.get(y,'').sort_values(by=sortChoices.get(x, ''), ascending = False).head(10)

        sns.barplot(

            data = dataFrame

            , x='Team'

            , y=sortChoices.get(x, '')

            , palette=palletChoices.get(y,'')

            , ax=ax1[x,y])

        ax1[x,y].axhline(0, color="k", clip_on=True)

        ax1[x,y].set_ylabel(ylabelChoices.get(x, '') + " - " + palletChoices.get(y,''))

        

        #writting the values

        for index, row in dataFrame.iterrows():

            ax1[x,y].text(i,round(row[sortChoices.get(x, '')],0) , round(row[sortChoices.get(x, '')],0), color='black', withdash=True)

            i += 1

        for item in ax1[x,y].get_xticklabels():

            item.set_rotation(30)

#List of top 5 countries for summer

summerCountries = medalsTeamsTotalsS.sort_values(by = 'Medal_pts', ascending = False)['Team'].head(5)

#List of top 5 countries for winter

winterCountries = medalsTeamsTotalsW.sort_values(by = 'Medal_pts', ascending = False)['Team'].head(5)



#empty dictionaries for summer and winter

summerTop5CountriesMedals = {}

winterTop5CountriesMedals = {}



#list - list of countries to prepare data

#season - summer or winter

def returnTop5CountriesMedals (list, season):

    tempDict = {} # empty temporary dictionary

    for c in list:

        # now we fill dictionary for country "c" with data

        tempDict[c] = medals.loc[(medals['Season']==season) & 

                                 (medals['Team']==c) & 

                                 ((medals['Gold'] > 0)|

                                  (medals['Silver'] > 0)|

                                  (medals['Bronze'] > 0))][['Team','Event','Gold','Silver','Bronze']]

        #group the data and sum

        tempDict[c] = tempDict[c].groupby(['Team','Event'])[listOfMedals].sum() #list of medals is variable

        tempDict[c].reset_index(inplace = True)

        #calculate Medal points based on medals in the data

        tempDict[c]['Medal_pts'] = (3*tempDict[c]['Gold'])+(2*tempDict[c]['Silver'])+tempDict[c]['Bronze']

    return tempDict



summerTop5CountriesMedals = returnTop5CountriesMedals (summerCountries, 'Summer')

winterTop5CountriesMedals = returnTop5CountriesMedals (winterCountries, 'Winter')
def printingTop5Countries(dict, heading):

    print(heading)

    for c in dict:

        print(c)

        display(dict[c].sort_values(by='Medal_pts', ascending = False).head(10)[['Event','Gold','Silver','Bronze', 'Medal_pts']])

        print('\n\r')

    

printingTop5Countries(summerTop5CountriesMedals,'Top 10 events of top 5 countries for Summer Olympic Games')

printingTop5Countries(winterTop5CountriesMedals,'Top 10 events of top 5 countries for Winter Olympic Games')
def drawGraphTop5Countries(dict, heading):

    # "detect" season from the heading and based on the season use pallete

    if 'summer' in heading.lower():

        palette = 'summer'

    else:

        palette = 'winter'

    rows = 5

    columns = 1

    f, (ax1) = plt.subplots(rows, columns, figsize=(15, 40), sharex = True)

    axIndex = 0

    print(heading)

    for c in dict:

        data = dict[c].sort_values(by='Medal_pts', ascending = False).head(10)

        sns.barplot(data=data

                    , y='Event'

                    , x='Medal_pts'

                    , palette=palette

                    , ax=ax1[axIndex])

        ax1[axIndex].axvline(0, color="k", clip_on=True)

        ax1[axIndex].set_xlabel('Medal_pts')

        ax1[axIndex].set_ylabel('Events')

        ax1[axIndex].set_title(c)

        #writting the values

        i = 0.1

        for index, row in data.iterrows():

            ax1[axIndex].text(row['Medal_pts'],i , row['Medal_pts'], color='black', withdash=True)

            i += 1

        axIndex += 1
drawGraphTop5Countries(summerTop5CountriesMedals,'Top 10 events of top 5 countries for Summer Olympic Games')
drawGraphTop5Countries(winterTop5CountriesMedals,'Top 10 events of top 5 countries for Winter Olympic Games')
# create dataframe with unique participants grouped by year, season and team

participants = pd.DataFrame(athletes1994.groupby(['Year','Season','Team'])['ID'].nunique())

participants.columns = ['UniqueParticipants']

participants.reset_index(inplace=True)

# create dataframe with sports that the team compete on grouped by year, season and team

sports = pd.DataFrame(athletes1994.groupby(['Year','Season','Team'])['Sport'].nunique())

sports.columns = ['ParticipatingOnSports']

sports.reset_index(inplace=True)

# create dataframe with events that the team compete on grouped by year, season and team

# just a reminder in context of this dataset, event is running 100m and sport is athletics

events = pd.DataFrame(athletes1994.groupby(['Year','Season','Team'])['Event'].nunique())

events.columns = ['ParticipatingOnEvents']

events.reset_index(inplace=True)

#now we merge it with dataframe we worked with (until now that is)

medalsTeamsParticipants = medalsTeams.merge(participants, on=['Year','Season','Team']).merge(sports,  on=['Year','Season','Team']).merge(events, on=['Year','Team','Season'])

# we can check / test if number of participants was appended properly

display(participants.loc[participants['Team']=='Croatia'])

display(sports.loc[sports['Team']=='Croatia'])

display(events.loc[events['Team']=='Croatia'])

display(medalsTeamsParticipants.loc[medalsTeamsParticipants['Team']=='Croatia'])
medalsTeamsParticipants['MedalPtsPerPart'] = medalsTeamsParticipants['Medal_pts'] /medalsTeamsParticipants['UniqueParticipants']

medalsTeamsParticipants['MedalPtsPerSport'] = medalsTeamsParticipants['Medal_pts'] /medalsTeamsParticipants['ParticipatingOnSports']

medalsTeamsParticipants['MedalPtsPerEvent'] = medalsTeamsParticipants['Medal_pts'] /medalsTeamsParticipants['ParticipatingOnEvents']

#Again just for test - Croatia

display(medalsTeamsParticipants.loc[medalsTeamsParticipants['Team']=='Croatia'])
bestyearsS = returnDFOfBestYears(medalsTeamsParticipants, "summer",5,'MedalPtsPerEvent', ['Team', 'Year', 'MedalPtsPerEvent'])

drawGraphBestYears(bestyearsS, 'summer', 'MedalPtsPerEvent')
bestyearsW = returnDFOfBestYears(medalsTeamsParticipants, "winter",5,'MedalPtsPerEvent', ['Team', 'Year', 'MedalPtsPerEvent'])

drawGraphBestYears(bestyearsW, 'winter', 'MedalPtsPerEvent')
def resultsForAllYearsMin5Pts(copyOfmedalsTeams, season):

    #Make sure that season has only 1st character capital

    season = season[0].upper() + season.lower()[1:]

    #Get only records of the season, and if season is "All" then just skip filtering

    if season!='All':

        copyOfmedalsTeams = copyOfmedalsTeams.loc[(copyOfmedalsTeams['Season']==season)]

    #Temporary list (np.array) of records where country had more then 5 medal points.

    #We need to get only unique values since some countries might have more then one row.

    tempListOfCountries = np.unique(copyOfmedalsTeams.loc[copyOfmedalsTeams['Medal_pts'] > 5, ['Team']].values)

    #Put the list in a dataframe

    numericListOfCountries = pd.DataFrame(tempListOfCountries)

    numericListOfCountries.columns = ['Team']

    #Turn a dataframe index to column

    numericListOfCountries['Id'] = numericListOfCountries.index

    #Merge two dataframes 

    copyOfmedalsTeams = copyOfmedalsTeams.loc[copyOfmedalsTeams['Team'].isin(tempListOfCountries)]

    copyOfmedalsTeams = copyOfmedalsTeams.merge(numericListOfCountries, on = 'Team')

    return copyOfmedalsTeams

def drawAllYearsMin5Pts(theData):

    f = plt.figure(figsize=(15, 15))

    numberOfCountries = theData['Team'].nunique()

    dashes = []

    g, h = 1, 1

    for i in range(numberOfCountries):

        dashes.append((random.randint(1,7),random.randint(1,7)))



    X, Y, hue = theData['Year'], theData['Medal_pts'], theData['Team']



    f=sns.lineplot(X,Y,hue = hue

                        , style=hue

                        , dashes=dashes)

    f.set_xticks(theData['Year'].unique())

    f.set_ylabel('Medal points')
drawAllYearsMin5Pts(resultsForAllYearsMin5Pts(medalsTeams, 'winter')) #note that winter is written with lower first letter :)
drawAllYearsMin5Pts(resultsForAllYearsMin5Pts(medalsTeams, 'summer')) #note that summer is written with lower first letter :)
tempData = resultsForAllYearsMin5Pts(medalsTeams, 'summer')

drawAllYearsMin5Pts(tempData.loc[tempData['Team']=='China'])
tempData = resultsForAllYearsMin5Pts(medalsTeams, 'winter')

drawAllYearsMin5Pts(tempData.loc[tempData['Team']=='Canada'])
#datafame - data we operate upon

def calculateCoeffVar(dataFrame):

    #create empty dataframe

    coeffVarDF = pd.DataFrame(columns=['Team','Coeff'])

    countries = dataFrame['Team'].unique()

    for c in countries:

        data = dataFrame.loc[dataFrame['Team']==c]['Medal_pts']

        coef = variation(data)

        #add values to dataframe

        coeffVarDF=coeffVarDF.append({'Team':c,'Coeff':coef}, ignore_index=True) 

    return coeffVarDF



#datafrane = data we draw

#palette = palette for the graph

def drawCoeffVar(dataFrame, palette):

    #definition of maxim which is used for modification of position of data labels

    #and in definition of limits

    maxim = dataFrame.sort_values(by = 'Coeff', ascending = False)['Coeff'].head(1)

    maxim = round((maxim.values[0] * 1.1),2)

    palette = palette.lower()

    g, (ax1) = plt.subplots(1, 1, figsize=(20, 7))

    sns.barplot(data = dataFrame, x = 'Team', y = 'Coeff', palette=palette, ax=ax1)

    ax1.axhline(0, color="k", clip_on=True)

    ax1.set(ylim=(0, maxim))

    ax1.set_ylabel("Coefficient of variation")

    #writting the datalabels

    i = -0.25

    prevValue = 0

    for index, row in dataFrame.iterrows():

        #this is indendent to avoid text over text writing

        value = round(row['Coeff'],2)

        if abs(1 - prevValue / value)<0.1:

            position = value + (maxim * 0.05)

        else:

            position = value

        prevValue = position

        ax1.text(i,position, value, color='black', withdash=True)

        i += 1

    for item in ax1.get_xticklabels():

        item.set_rotation(90)
drawCoeffVar(calculateCoeffVar(resultsForAllYearsMin5Pts(medalsTeams, 'winter')) ,'winter')
drawCoeffVar(calculateCoeffVar(resultsForAllYearsMin5Pts(medalsTeams, 'summer')),'summer')
tempData = resultsForAllYearsMin5Pts(medalsTeams, 'winter')

print(tempData.loc[tempData['Team']=='Norway'])
tempData = resultsForAllYearsMin5Pts(medalsTeams, 'summer')

print(tempData.loc[tempData['Team']=='France'])
tempData = resultsForAllYearsMin5Pts(medalsTeams, 'winter')

print(tempData.loc[tempData['Team']=='Estonia'])
tempData = resultsForAllYearsMin5Pts(medalsTeams, 'summer')

print(tempData.loc[tempData['Team']=='Egypt'])
#gdpdata = pd.read_csv('GDP by Country.csv', skiprows = range(0,4))

gdpdata = pd.read_csv('../input/gdp-world-bank-data/GDP by Country.csv', skiprows = range(0,4))
gdpdata.head()
years = np.sort(athletes1994['Year'].unique())

print("This is the list:")

print(str(years).split())
years_string=str(years)[1:-1].split()

cntrname = 'Country Name'

years_string.insert(0, cntrname)

print(years_string)
gdpdata_work = gdpdata[years_string]

gdpdata_work.head()
melted_gdp = gdpdata_work.melt(id_vars=['Country Name'], value_vars=years_string[1:], var_name='Years')
melted_gdp.columns=['Team','Year','GDP']
melted_gdp.head()
#get both list - countries from both dataframes

olympic_countries = athletes1994['Team'].unique()

gdp_countries = gdpdata_work['Country Name'].unique()
#compare both list. Result is list of countries that competed on Olympic games and dont have GDP data.

np.setdiff1d(olympic_countries,gdp_countries)
#compare both list. Result is list of countries that have GDP data and didn't competed on Olympic games.

np.setdiff1d(gdp_countries,olympic_countries)
melted_gdp['Country']=melted_gdp['Team']

melted_gdp.head()
#the dictionary with changes

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



# Short for loop for the change

for key in change:

    value = change[key]

    print("Changing \"{}\" into \"{}\"".format(key, value))

    melted_gdp.loc[melted_gdp['Team']==key,'Team']=value
#Olympic countries didn't change

#olympic_countries = medalsTeamsParticipants['Team'].unique()

gdp_countries = melted_gdp['Team'].unique()
np.setdiff1d(olympic_countries,gdp_countries)
melted_gdp['Year']=melted_gdp['Year'].apply(int)
total_data = pd.merge(medalsTeamsParticipants, melted_gdp, on=['Team','Year'])
total_data.head()
total_data.drop(['Country'], axis = 1, inplace = True)
#population = pd.read_csv('country_population.csv')

population = pd.read_csv('../input/world-bank-data-1960-to-2016/country_population.csv')

population.head()
pop_work = population[years_string]
olympic_countries = total_data['Team'].unique()

pop_countries = pop_work['Country Name'].unique()

print ("If the list is empty then all countries in total data have population data")

print("And the list is: ",np.setdiff1d(olympic_countries,gdp_countries))
melted_pop = pop_work.melt(id_vars=['Country Name'], value_vars=years_string[1:], var_name='Years')

melted_pop.columns=['Team','Year','Population']

melted_pop['Year']=melted_pop['Year'].apply(int)

melted_pop.head()
total_data = pd.merge(total_data, melted_pop, on=['Team','Year'])
total_data[total_data['Team']=='Croatia']
total_data['GDPPercapita']=total_data['GDP']/total_data['Population']

total_data['MedalsPer100kcapita']=total_data['Medals']/(total_data['Population']/100000)

total_data['MedalPointsPer100kcapita']=total_data['Medal_pts']/(total_data['Population']/100000)

total_data['GDPPerMedal']=(total_data['GDP']/total_data['Medals'])#.apply(lambda x: '{:,.2f}'.format(float(x))) 

total_data['GDPPerMedalPoints']=(total_data['GDP']/(total_data['Medal_pts']))#.apply(lambda x: '{:,.2f}'.format(float(x))) 

total_data['ParticipantsPer100kPop']=total_data['UniqueParticipants']/(total_data['Population']/100000)
total_data[total_data['Team']=='United States']
working_data = total_data.copy()

working_data.info()
bestyearsS = returnDFOfBestYears(working_data, "summer",10,'GDPPerMedalPoints', ['Team', 'Year', 'GDPPerMedalPoints'], True)

drawGraphBestYears(bestyearsS, 'summer', 'GDPPerMedalPoints', 1000000000)
bestyearsW = returnDFOfBestYears(working_data, "winter",5,'GDPPerMedalPoints', ['Team', 'Year', 'GDPPerMedalPoints'], True)

drawGraphBestYears(bestyearsW, 'winter', 'GDPPerMedalPoints', 1000000000)
def drawComparison(

    country

    , palette

    , firstData

    , firstDataLabel

    , secondData

    , secondDataLabel

    , firstDataDivide = 1

    , secondDataDivide = 1):

    #country = country[0].upper() + country.lower()[1:]

    tempData = resultsForAllYearsMin5Pts(working_data, 'All')

    dataFrame = tempData.loc[tempData['Team']==country]

    palette = palette.lower()

    g, ax1 = plt.subplots(figsize=(20, 7))

    sns.barplot(

        dataFrame['Year'].apply(lambda x: str(x)), 

        dataFrame[firstData].apply(lambda x: x / firstDataDivide), 

        ax=ax1, palette = palette)

    ax1.axhline(0, color="k", clip_on=True)

    ax2 = ax1.twinx()

    sns.lineplot(

        dataFrame['Year'].apply(lambda x: str(x)), 

        dataFrame[secondData].apply(lambda x: x / secondDataDivide), 

        linewidth = 4, marker = 's', markersize = 12,

        ax=ax2)

    ax2.grid(False)

    ax1.set_title('Comparison of '+ firstDataLabel +' and '+ secondDataLabel +' through years for ' + country)

    ax1.set_ylabel(firstDataLabel)

    ax2.set_ylabel(secondDataLabel)   

drawComparison(

    "Bulgaria",

    "Rocket", 

    "Medal_pts", 

    "Medal points", 

    "GDP", 

    "GDP (billion USD)", 

    1, 

    1000000000)
drawComparison(

    "Norway",

    "seismic", 

    "Medal_pts", 

    "Medal points", 

    "GDP", 

    "GDP (billion USD)", 

    1, 

    1000000000)
theSportiestSummer = returnDFOfBestYears(working_data, 

                                         "summer",

                                         5,

                                         'ParticipantsPer100kPop', 

                                         ['Team', 'Year', 'ParticipantsPer100kPop'], 

                                         False)

drawGraphBestYears(theSportiestSummer, 'summer', 'ParticipantsPer100kPop')
theSportiestWinter = returnDFOfBestYears(working_data, 

                                         "winter",

                                         5,

                                         'ParticipantsPer100kPop', 

                                         ['Team', 'Year', 'ParticipantsPer100kPop'], 

                                         False)

drawGraphBestYears(theSportiestWinter, 'winter', 'ParticipantsPer100kPop')
print("Looking only at the best years the average number of participants in summer per 100.000 habitants of a country is: ", round(theSportiestSummer.ParticipantsPer100kPop.mean(),2), " with standard deviation: ", round(theSportiestSummer.ParticipantsPer100kPop.std(),2))

print("Looking only at the best years the average number of participants in winter per 100.000 habitants of a country is: ", round(theSportiestWinter.ParticipantsPer100kPop.mean(),2), " with standard deviation: ", round(theSportiestWinter.ParticipantsPer100kPop.std(),2))
theSportiestCountries = ['Latvia','Slovenia', 'New Zealand']

for c in theSportiestCountries:

    print("ParticipantsPer100kPop over years for ",c)

    tempResult = working_data.loc[working_data['Team'] == c][['Year','Season','ParticipantsPer100kPop']]

    display(tempResult)

    print('The average values are:')

    print('For summer: ', tempResult.loc[tempResult['Season'] == 'Summer']['ParticipantsPer100kPop'].mean())

    print('For winter: ', tempResult.loc[tempResult['Season'] == 'Winter']['ParticipantsPer100kPop'].mean())

    print()

for c in theSportiestCountries:

    drawComparison(

        c,

        "seismic", 

        "UniqueParticipants", 

        "Unique participants", 

        "Population", 

        "Population (in millions)", 

        1, 

        1000000)
bestyearsS = returnDFOfBestYears(working_data, "summer",5,'MedalPointsPer100kcapita', ['Team', 'Year', 'MedalPointsPer100kcapita'], False)

drawGraphBestYears(bestyearsS, 'summer', 'MedalPointsPer100kcapita', 1)
bestyearsW = returnDFOfBestYears(working_data, "Winter",5,'MedalPointsPer100kcapita', ['Team', 'Year', 'MedalPointsPer100kcapita'], False)

drawGraphBestYears(bestyearsW, 'Winter', 'MedalPointsPer100kcapita', 1)
theSportiestCountries = ['Jamaica', 'Norway']

for c in theSportiestCountries:

    print("MedalPointsPer100kcapita over years for ",c)

    tempResult = working_data.loc[working_data['Team'] == c][['Year','Season','MedalPointsPer100kcapita','Medal_pts', 'Population']]

    display(tempResult)

    print('The average values are:')

    print('For summer: ', tempResult.loc[tempResult['Season'] == 'Summer']['MedalPointsPer100kcapita'].mean())

    print('For winter: ', tempResult.loc[tempResult['Season'] == 'Winter']['MedalPointsPer100kcapita'].mean())

    print()