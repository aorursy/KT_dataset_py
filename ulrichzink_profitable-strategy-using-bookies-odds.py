#Imports 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3

import matplotlib.pyplot as plt





import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



path = "../input/"  #Insert path here

database = path + 'database.sqlite'
conn = sqlite3.connect(database)



tables = pd.read_sql("""SELECT *

                        FROM sqlite_master

                        WHERE type='table';""", conn)

tables
# Subsetting the data that we are interested in



detailed_matches = pd.read_sql("""SELECT Country.name AS country_name, 

                                        League.name AS league_name, 

                                        season, 

                                        date,

                                        home_team_goal, 

                                        away_team_goal,

                                        PSH, PSD, PSA 

                                FROM Match

                                JOIN Country on Country.id = Match.country_id

                                JOIN League on League.id = Match.league_id

                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id

                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id

                                /*WHERE country_name = 'Spain'*/

                                ORDER by date

                                """, conn)

detailed_matches.dropna(inplace = True)

detailed_matches['date'] = pd.to_datetime(detailed_matches.date)

detailed_matches.head()
# Convert Odds to probability

detailed_matches[['PSH','PSD','PSA']] = detailed_matches[['PSH','PSD','PSA']].apply(lambda x:(1/x), axis=1)



# Check for margin AKA bookies' profit (sum of probility > 100 = bookies' margin)

detailed_matches['Total'] =detailed_matches[['PSH','PSD','PSA']].sum(axis=1)
# Helper function to create column for 3 possible outcomes (HW:Home Win, AW: Away Win, D: Draw)



def create_outcomes_columns(df,home_team_goal,away_team_goal):

    df['resultHW'] = np.where(df.home_team_goal > df.away_team_goal,1,0)

    df['resultAW'] = np.where(df.home_team_goal < df.away_team_goal,1,0)

    df['resultD'] = np.where(df.home_team_goal == df.away_team_goal,1,0)



create_outcomes_columns(detailed_matches,'home_team_goal','away_team_goal')



detailed_matches.head()
# Total number of matches and repartition accross the 3 classes



print(f"Matches won by home team {detailed_matches.resultHW.sum()}\nMatches won by away team {detailed_matches.resultAW.sum()}\nMatches ended in a draw {detailed_matches.resultD.sum()} ")



print(f"\nTotal number of matches: {len(detailed_matches)}")
kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=40)

plt.rcParams["figure.figsize"] = (10,10)

plt.hist(x=detailed_matches.PSH, **kwargs,label='Home Win')

plt.hist(x=detailed_matches.PSA, **kwargs,label='Away Win')

plt.hist(x=detailed_matches.PSD, **kwargs,label='Draw')



plt.xlabel('Pinnacle odds')

plt.ylabel('Occurrence')

plt.title('Betting Odds Frequency by Class')

plt.legend();
# Helper functions to analyse the data







def checkProba(oddsCol,actualColName,df):

    bin = np.arange(0,1.01,0.05)

    return df.groupby([pd.cut(df[oddsCol],bin)])[actualColName].mean().reset_index()

    

def create_bin_df(sourceDf, dfColumns):

    rslt = pd.DataFrame()



    for i in range(0,len(dfColumns)):

        tmp0 = sourceDf[dfColumns[i]].rename(columns={ sourceDf[dfColumns[i]].columns[0]: "odds" })

        tmp1 = checkProba(tmp0.columns[0],tmp0.columns[1],tmp0)

        if rslt.empty:

            rslt = tmp1

        else:

            rslt = pd.merge(rslt, tmp1,how='outer')

    

    return rslt



def format_result(df):

    c1 = 'background-color: orange'

    c2 = ''

    c3 = 'background-color: green'

    #compare columns

    mask1 = df.iloc[:,1:].apply(lambda x: x > np.arange(0.025,1,.05))

    mask2 = df.iloc[:,1:].apply(lambda x: x > np.arange(0.05,1.05,.05))

    #DataFrame with same index and columns names as original filled empty strings

    df1 =  pd.DataFrame(c2, index=df.index, columns=df.columns)

    #modify values of df1 column by boolean mask

    df1.iloc[:,1:][mask1] = c1

    df1.iloc[:,1:][mask2] = c3

    

    return df1
# Identifying market inefficiencies i.e. when probability of outcome > odds we are value betting



dfc = [['PSH','resultHW'],['PSA','resultAW'],['PSD','resultD']]

create_bin_df(detailed_matches, dfc).style.apply(format_result, axis=None)
import matplotlib.lines as mlines

import matplotlib.ticker as mticker



def plot_graph(df,sizeParam = 10,Country=''):

    

    plt.rcParams["figure.figsize"] = (sizeParam,sizeParam)

    fig, ax = plt.subplots()

    

    y = df[df.columns[0]].astype(str)

    

    # plot calibrated reliability

    labels = [['HomeWin'],['AwayWin'],['Draw']]

    

    for i in np.arange(1 ,len(df.columns)):

        plt.plot(y, df[df.columns[i]], marker='.',label= df.columns[i])



    plt.xlabel('Pinnacle odds')

    plt.ylabel('Actual')

    plt.title('Betting Odds as Prediction Indicator '+ Country)

    plt.legend()



    line = mlines.Line2D([0, 1], [0, 1], color='red',linestyle='--')

    transform = ax.transAxes

    line.set_transform(transform)

    ax.add_line(line)



    myLocator = mticker.MultipleLocator(2)

    ax.xaxis.set_major_locator(myLocator)



    plt.show()



plot_graph(create_bin_df(detailed_matches, dfc))
#List countries



countriesChampionship = list(dict.fromkeys(detailed_matches.country_name))

countriesChampionship
def reshape_source_data (df,filter_by,outcome):

    rslt  = pd.DataFrame()

    for i in list(dict.fromkeys(df[filter_by])):

        tmp1 = df.loc[(df[filter_by] == i)][outcome]

        tmp2 = create_bin_df(tmp1,[[tmp1.columns[0],tmp1.columns[1]]])

        tmp2.rename(index=str, columns={tmp2.columns[1]: i},inplace= True)

        if rslt.empty:

                rslt = tmp2

        else:

                rslt = pd.merge(rslt, tmp2,how='outer')

    return rslt
# Identifying market inefficiencies by countries i.e. probability of outcome > odds



from IPython.display import display

dfc = [['PSH','resultHW'],['PSA','resultAW'],['PSD','resultD']]

for i in dfc:

    display(i[1])

    display(reshape_source_data (detailed_matches,'country_name',i).style.apply(format_result, axis=None))


def plot_graph_by_country (df,sizeParam = 10,Country='', outcome = ''):

    

    # plt.rcParams["figure.figsize"] = (sizeParam,sizeParam)



    #df = reshape_source_data (detailed_matches,'country_name',['PSH','resultHW'])



    fig, ax = plt.subplots()



    y = df[df.columns[0]].astype(str)



    # plot calibrated reliability

    for i in np.arange(1 ,len(df.columns)):

        plt.plot(y, df[df.columns[i]], marker='.',label= df.columns[i])



    plt.xlabel('Pinnacle odds')

    plt.ylabel('Actual')

    plt.title('Betting Odds as Prediction Indicator ' + outcome)

    plt.legend()



    line = mlines.Line2D([0, 1], [0, 1], color='red',linestyle='--')

    transform = ax.transAxes

    line.set_transform(transform)

    ax.add_line(line)



    myLocator = mticker.MultipleLocator(2)

    ax.xaxis.set_major_locator(myLocator)



    plt.show()

plot_graph_by_country(reshape_source_data(detailed_matches,'country_name',['PSH','resultHW'])[['odds','Scotland','Spain','Germany']],outcome = 'Home Win') 

## Does the pattern occures over time? Let's check by season.



#List seasons



seasons = list(dict.fromkeys(detailed_matches.season))

seasons
# First lets see the graph for each seasons

for i in seasons :

    display(i)

    plot_graph_by_country(reshape_source_data(detailed_matches.loc[(detailed_matches['season'] == i)],'country_name',['PSH','resultHW'])[['odds','Scotland','Spain','Germany']],outcome = 'Home Win') 
# Now lets look at the data per season



for i in seasons:

    display(i)

    display(reshape_source_data(detailed_matches.loc[(detailed_matches['season'] == i)],'country_name',['PSH','resultHW'])[['odds','Scotland','Spain','Germany']].style.apply(format_result, axis=None))
## Would we have made some money with this?



# First lets check with the data in historical dataset
detailed_matches['profitAW'] = np.where((detailed_matches['resultAW'] == 1), 1/detailed_matches['PSA']-1, -1)

detailed_matches['profitHW'] = np.where((detailed_matches['resultHW'] == 1), 1/detailed_matches['PSH']-1, -1)

detailed_matches['profitD'] = np.where((detailed_matches['resultD'] == 1), 1/detailed_matches['PSA']-1, -1)



detailed_matches.head()
moneyDF = detailed_matches[detailed_matches.country_name.isin(['Scotland','Spain','Germany'])]

profitHWCut = moneyDF.groupby([pd.cut(moneyDF.PSH,np.arange(0,1.1,.05)).astype(str)])['profitHW'].sum().reset_index()

profitAWCut = moneyDF.groupby([pd.cut(moneyDF.PSA,np.arange(0,1.1,.05)).astype(str)])['profitAW'].sum().reset_index()

profitDCut = moneyDF.groupby([pd.cut(moneyDF.PSD,np.arange(0,1.1,.05)).astype(str)])['profitD'].sum().reset_index()
HWHistorical = moneyDF.loc[(moneyDF['PSH'] < .20)&(moneyDF['PSH'] > .05)]['profitHW'].sum()

nbBettsHst = len(moneyDF.loc[(moneyDF['PSH'] < .20)&(moneyDF['PSH'] > .05)]['resultHW'])

nbBettsHWWnHst =  moneyDF.loc[(moneyDF['PSH'] < .20)&(moneyDF['PSH'] > .05)]['resultD'].sum()



print ("Had we bett 1$ on each match where home win was priced between 5% and 20% over the course of four seasons since 2012/2013")

print ('We would have bet '+str(nbBettsHst)+' times/$'+ ' and won '+str(nbBettsHWWnHst)+' times.')

print ('We would have made '+ str(HWHistorical) + '$')

print ('The rate of return is '+str(round(HWHistorical/nbBettsHst*100,2))+'%')
##Lets validtae further by checking with the latest Season
## Download and format data for the 2018/2019 season from www.football-data.co.uk

eng =  pd.read_csv('http://www.football-data.co.uk/mmz4281/1819/E0.csv')

nl =  pd.read_csv('http://www.football-data.co.uk/mmz4281/1819/N1.csv')

de =  pd.read_csv('http://www.football-data.co.uk/mmz4281/1819/D1.csv')

it =  pd.read_csv('http://www.football-data.co.uk/mmz4281/1819/I1.csv')

sp =  pd.read_csv('http://www.football-data.co.uk/mmz4281/1819/SP1.csv')

be =  pd.read_csv('http://www.football-data.co.uk/mmz4281/1819/B1.csv')

fr =  pd.read_csv('http://www.football-data.co.uk/mmz4281/1819/F1.csv')

sct = pd.read_csv('http://www.football-data.co.uk/mmz4281/1819/SC0.csv',encoding = 'unicode_escape')
season1819 = pd.concat([eng,de,it,sp,be,fr,sct,nl], ignore_index=True, sort=False)#[['Div','Date','FTHG','FTAG','PSCH','PSCD','PSCA']]
season1819[season1819.Div == 'E0'].tail()
season1819['resultHW'] = np.where(season1819.FTHG > season1819.FTAG,1,0)

season1819['resultAW'] = np.where(season1819.FTHG < season1819.FTAG,1,0)

season1819['resultD'] = np.where(season1819.FTHG == season1819.FTAG,1,0)
# Convert Odds to probability

season1819[['PSH','PSD','PSA']] = season1819[['PSCH','PSCD','PSCA']].apply(lambda x:(1/x), axis=1)
season1819['profitAW'] = np.where((season1819['resultAW'] == 1), 1/season1819['PSA']-1, -1)

season1819['profitHW'] = np.where((season1819['resultHW'] == 1), 1/season1819['PSH']-1, -1)

season1819['profitD'] = np.where((season1819['resultD'] == 1), 1/season1819['PSA']-1, -1)
HW1819 = season1819.loc[(season1819['Div'].isin(['SC0','SP1','D1']))&(season1819['PSH'] < .20)&(season1819['PSH'] > .05)]['profitHW'].sum()

nbBettsHW1819 = len(season1819.loc[(season1819['Div'].isin(['SC0','SP1','D1']))&(season1819['PSH'] < .20)&(season1819['PSH'] > .05)])

nbBettsWnHW1818 = season1819.loc[(season1819['Div'].isin(['SC0','SP1','D1']))&(season1819['PSH'] < .20)&(season1819['PSH'] > .05)]['resultD'].sum()





print ("Had we bett 1$ on each match where home win was priced between 5% and 20% during the 2018/2019 season")

print ('We would have bet '+str(nbBettsHW1819)+' times/$'+ ' and won '+str(nbBettsWnHW1818)+' times.')

print ('We would have made '+ str(round(HW1819,2)) + '$')

print ('The a rate of return is '+str(round(HW1819/nbBettsHW1819*100,2))+'% Not bad!!!')
## ForDraws Scotland and the Netherlands seem to be good candidates



plot_graph_by_country(reshape_source_data(detailed_matches,'country_name',['PSD','resultD'])[['odds','Scotland','Netherlands']],outcome = 'Home Win') 

#Lets see the graph by seasons



for i in seasons :

    display(i)

    plot_graph_by_country(reshape_source_data(detailed_matches.loc[(detailed_matches['season'] == i)],'country_name',['PSD','resultD'])[['odds','Scotland','Netherlands']],outcome = 'Home Win') 
# Lets have a look at the data
# Now lets look at the data per season



for i in seasons:

    display(i)

    display(reshape_source_data(detailed_matches.loc[(detailed_matches['season'] == i)],'country_name',['PSD','resultD'])[['odds','Scotland','Netherlands']].style.apply(format_result, axis=None))
moneyDF = detailed_matches[detailed_matches.country_name.isin(['Scotland','Netherlands'])]
profitHWCut = moneyDF.groupby([pd.cut(moneyDF.PSH,np.arange(0,1.1,.05)).astype(str)])['profitHW'].sum().reset_index()

profitAWCut = moneyDF.groupby([pd.cut(moneyDF.PSA,np.arange(0,1.1,.05)).astype(str)])['profitAW'].sum().reset_index()

profitDCut = moneyDF.groupby([pd.cut(moneyDF.PSD,np.arange(0,1.1,.05)).astype(str)])['profitD'].sum().reset_index()
# Lest check how much we would make using our historical dataset
DHst = moneyDF.loc[(moneyDF['PSD'] < .25)&(moneyDF['PSD'] > .05)]['profitD'].sum()

nbBettsDHst = len (moneyDF.loc[(moneyDF['PSD'] < .25)&(moneyDF['PSD'] > .05)])

nbBettsWnDHst = moneyDF.loc[(moneyDF['PSD'] < .25)&(moneyDF['PSD'] > .05)]['resultD'].sum()



print ("Had we bett 1$ on each match where draw  was priced above 5% but below 25% over the course of the four seasons since 2012/2013.")

print ('We would have bet '+str(nbBettsDHst)+' times/$'+ ' and won '+str(nbBettsWnDHst)+' times.')

print ('We would have made '+ str(round(DHst,2)) + '$')

print ('The rate of return is '+str(round(DHst/nbBettsDHst*100,2))+'%. Very impressive can we confirm this performance with the 2018/2019 season?')
# lets validate with season 2018/2019
D1819 = season1819.loc[(season1819['Div'].isin(['SC0','N1']))&(season1819['PSD'] < .25)&(season1819['PSD'] > .05)]['profitD'].sum()

nbBettsD1819 = len(season1819.loc[(season1819['Div'].isin(['SC0','N1']))&(season1819['PSD'] < .25)&(season1819['PSD'] > .05)])

nbBettsWnD1818 = season1819.loc[(season1819['Div'].isin(['SC0','N1']))&(season1819['PSD'] < .25)&(season1819['PSD'] > .05)]['resultD'].sum()



print ("Had we bett 1$ on each match where draw  was priced above 5% but below 25% during the 2018/2019 season.")

print ('We would have bet '+str(nbBettsD1819)+' times/$'+ ' and won '+str(nbBettsWnD1818)+' times.')

print ('We would have made '+ str(round(D1819,2)) + '$')

print ('The rate of return is '+str(round(D1819/nbBettsD1819*100,2))+'%. Not as impressive but still decent.')