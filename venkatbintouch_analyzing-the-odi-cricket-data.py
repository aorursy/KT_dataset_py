import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

sns.set_style('darkgrid')



#-------------------------------------------------------------------------------------
######################## load and sanitize data ######################################
#-------------------------------------------------------------------------------------
ContinousDatasetModified_df = pd.read_csv("/kaggle/input/continousdatasetmodified/ContinousDatasetModified.csv")
ContinousDatasetModified_df.drop('Scorecard',  axis='columns', inplace=True)
#print (ContinousDatasetModified_df)

#observe that some matches were played on multiple days, so lets sanitize the date column
def sanitizeMatchDate(date):
    date = re.sub("\d+-", "", date)
    return date
ContinousDatasetModified_df['Match Date'] = ContinousDatasetModified_df['Match Date'].apply(func = sanitizeMatchDate)
ContinousDatasetModified_df["Match Date"] = pd.to_datetime(ContinousDatasetModified_df["Match Date"])

#There is just one record where "Venue_Team1"  is "Away" but all others are "Home", so just get rid of it
ContinousDatasetModified_df.drop(ContinousDatasetModified_df[ContinousDatasetModified_df.Venue_Team1 == 'Away'].index, inplace=True)

#--------------------------------------------------------------------------------------
##################### lets build some overall statistics ##############################
#--------------------------------------------------------------------------------------
# Lets build a Data Frame with over all Team wise played, won, lost etc. statistics
def teamWiseStatistics(fromDate, minNoOfMataches, ContinousDatasetModified_df, tillDate = pd.Timestamp(2017,12,1)):
    ContinousDatasetModified_df  = ContinousDatasetModified_df[
        (ContinousDatasetModified_df['Match Date']>fromDate ) & (ContinousDatasetModified_df['Match Date']<tillDate) ]
    #print ("\nContinousDatasetModified_df (Date > "+str(fromDate)+" ):\n",ContinousDatasetModified_df)

    # Create empty df with unique team names(as column "Team") from both "Team 1" and "Team 2" columns of  ContinousDatasetModified_df
    column_values = ContinousDatasetModified_df[["Team 1", "Team 2"]].values.ravel()
    unique_values =  pd.unique(column_values)
    Team_wise_statistics_df = pd.DataFrame({"Team": pd.Series(unique_values)})

    # ----------------------------------------------
    # Now calculate the Total matches played, by each team
    Total_Matches_Played_series = ContinousDatasetModified_df['Team 1'].append(
                                                                ContinousDatasetModified_df.loc[
                                                                    ContinousDatasetModified_df['Team 1'] != ContinousDatasetModified_df['Team 2'], 'Team 2'
                                                                ]).value_counts()
    temp_df = Total_Matches_Played_series.rename_axis('Team').reset_index(name='Total_Played')
    Team_wise_statistics_df = Team_wise_statistics_df.merge(temp_df, on="Team", how='outer').fillna(0)
    #let us calculate total wins and total losses, by each team
    winner_series = ContinousDatasetModified_df['Winner'].value_counts()
    temp_df = winner_series.rename_axis('Team').reset_index(name='Total_Wins')
    Team_wise_statistics_df = Team_wise_statistics_df.merge(temp_df, on="Team", how='outer').fillna(0)
    Team_wise_statistics_df['Total_Lost'] = Team_wise_statistics_df['Total_Played'] - Team_wise_statistics_df['Total_Wins']
    Team_wise_statistics_df = Team_wise_statistics_df[Team_wise_statistics_df['Total_Played']>minNoOfMataches]
    
    # ----------------------------------------------
    # Now let us calculate the no of matches played at Home, by each team
    Matches_Played_Home_series = ContinousDatasetModified_df[ContinousDatasetModified_df['Venue_Team1'] == 'Home']['Team 1'].value_counts()
    temp_df = Matches_Played_Home_series.rename_axis('Team').reset_index(name='Played_Home')
    Team_wise_statistics_df = Team_wise_statistics_df.merge(temp_df, on="Team", how='left').fillna(0)
    #let us calculate Home wins and losses, by each team
    temp_df = ContinousDatasetModified_df[ContinousDatasetModified_df['Venue_Team1'] == 'Home']
    Home_winner_series = temp_df[temp_df['Team 1'] == temp_df['Winner']]['Winner'].value_counts()
    temp_df = Home_winner_series.rename_axis('Team').reset_index(name='Home_Wins')
    Team_wise_statistics_df = Team_wise_statistics_df.merge(temp_df, on="Team", how='left').fillna(0)
    Team_wise_statistics_df['Home_Lost'] = Team_wise_statistics_df['Played_Home'] - Team_wise_statistics_df['Home_Wins']
    
    # ----------------------------------------------
    # Now let us calculate the no of matches played Away, by each team
    Matches_Played_Away_series = ContinousDatasetModified_df[ContinousDatasetModified_df['Venue_Team2'] == 'Away']['Team 2'].value_counts()
    temp_df = Matches_Played_Away_series.rename_axis('Team').reset_index(name='Played_Away')
    Team_wise_statistics_df = Team_wise_statistics_df.merge(temp_df, on="Team", how='left').fillna(0)
    #let us calculate Away wins and losses, by each team
    temp_df = ContinousDatasetModified_df[ContinousDatasetModified_df['Venue_Team2'] == 'Away']
    Away_winner_series = temp_df[temp_df['Team 2'] == temp_df['Winner']]['Winner'].value_counts()
    temp_df = Away_winner_series.rename_axis('Team').reset_index(name='Away_Wins')
    Team_wise_statistics_df = Team_wise_statistics_df.merge(temp_df, on="Team", how='left').fillna(0)
    Team_wise_statistics_df['Away_Lost'] = Team_wise_statistics_df['Played_Away'] - Team_wise_statistics_df['Away_Wins']

    # ----------------------------------------------
    # Now let us calculate the no of matches played at a Nuetral venue, by each team
    temp_df = ContinousDatasetModified_df[ContinousDatasetModified_df['Venue_Team1'] == 'Neutral']
    Matches_Played_Nuetral_series = temp_df['Team 1'].append(
                                                            temp_df.loc[
                                                                temp_df['Team 1'] != temp_df['Team 2'], 'Team 2'
                                                                ]).value_counts()
    temp_df = Matches_Played_Nuetral_series.rename_axis('Team').reset_index(name='Played_Nuetral')
    Team_wise_statistics_df = Team_wise_statistics_df.merge(temp_df, on="Team", how='left').fillna(0)
    #let us calculate Nuetral wins and losses, by each team    
    temp_df = ContinousDatasetModified_df[ContinousDatasetModified_df['Venue_Team1'] == 'Neutral']
    nuetral_winner_series = temp_df['Winner'].value_counts()
    temp_df = nuetral_winner_series.rename_axis('Team').reset_index(name='Nuetral_Wins')
    Team_wise_statistics_df = Team_wise_statistics_df.merge(temp_df, on="Team", how='left').fillna(0)
    Team_wise_statistics_df['Nuetral_Lost'] = Team_wise_statistics_df['Played_Nuetral'] - Team_wise_statistics_df['Nuetral_Wins']

    # ----------------------------------------------
    Team_wise_statistics_df.set_index("Team", inplace = True)
    #Team_wise_statistics_df = Team_wise_statistics_df.apply(pd.to_numeric)
    #Team_wise_statistics_df[["Played_Home", "Home_Wins"]] = Team_wise_statistics_df[["Played_Home", "Home_Wins"]].apply(pd.to_numeric)
    Team_wise_statistics_df["Played_Home"] = pd.to_numeric(Team_wise_statistics_df["Played_Home"], errors='ignore')

    #print statistics
    heading = str(fromDate).split(" ")[0].split("-")[0]+"-"+str(tillDate).split(" ")[0].split("-")[0]
    if minNoOfMataches > 0:
        heading = heading + " (with minimum "+str(minNoOfMataches)+" matches):"
    #print ("\nTeam wise over all statistics during "+heading)
    #print (Team_wise_statistics_df)
    return Team_wise_statistics_df

tws_df = teamWiseStatistics(pd.Timestamp(1971,1,1), 0, ContinousDatasetModified_df)
print ("Team wise over all statistics:\n",tws_df)

#-------------------------------------------------------------------------------------
######################## Exploratory Data analysis ###################################
#-------------------------------------------------------------------------------------

"""
Based on statistics, there are 10 teams which played a lot of cricket from 1971 till 2017
Lets us see over all win loss record for the top playing nations with minimum 300 minimum matches
"""

# let us visualize which team plays a lot of cricket and also check overall wins and losses
minNoOfMataches = 300
fromDate = pd.Timestamp(1971,1,1)
tws_df = teamWiseStatistics(fromDate, minNoOfMataches, ContinousDatasetModified_df )

tws_df = tws_df.sort_values('Total_Played', ascending=False)
plt.figure(figsize=(10,8))
plt.xticks(rotation=35)
plt.title("Top Cricket Playing Nations (> "+str(minNoOfMataches)+" matches), since "+str(fromDate).split(" ")[0], fontsize=20, fontweight="bold")
plt.ylabel('Total Mataches Played', fontsize=12, fontweight="bold")
# let us plot a bar chart
ax1 = plt.bar(tws_df.index, tws_df.Total_Wins, color='green')
ax2 = plt.bar(tws_df.index, tws_df.Total_Lost, bottom=tws_df.Total_Wins, color='salmon');
plt.legend(['Win','Loss'])
for r1, r2 in zip(ax1, ax2):
    h1 = r1.get_height()
    h2 = r2.get_height()
    h3 = h1 + h2
    plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., str( round( (h1*100)/h3 ))+"%", ha="center", va="center", color="white", fontsize=12)
    plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., str( round( (h2*100)/h3 ))+"%", ha="center", va="center", color="black", fontsize=12)
    plt.text(r2.get_x() + r2.get_width() / 2., h3 + 10, "%d" % h3, ha="center", va="center", color="black", fontsize=12, fontweight="bold")
plt.savefig("TotalWinsVsLosses-since "+str(fromDate).split(" ")[0]+".png")
plt.show()

"""
Inferences:
So it can be observed that over all there are 10 teams which played frequently
And India, Australia, Pakistan played the most number of matches for the entire period of the dataset
And Australia, South Africa has the best over all winning percentages till date

But it is dfficult to conclude which team is going strong based on the over all data.
So lets visualize the same data again for the recent times, during  2015-2017 
"""
minNoOfMataches = 30
fromDate = pd.Timestamp(2015,1,1)
tws_df = teamWiseStatistics(fromDate, minNoOfMataches, ContinousDatasetModified_df )

tws_df = tws_df.sort_values('Total_Played', ascending=False)
plt.figure(figsize=(10,8))
plt.xticks(rotation=35)
plt.title("Top Cricket Playing Nations (> "+str(minNoOfMataches)+" matches), since "+str(fromDate).split(" ")[0], fontsize=20, fontweight="bold")
plt.ylabel('Total Mataches Played', fontsize=12, fontweight="bold")
# let us plot a bar chart
ax1 = plt.bar(tws_df.index, tws_df.Total_Wins, color='green')
ax2 = plt.bar(tws_df.index, tws_df.Total_Lost, bottom=tws_df.Total_Wins, color='salmon');
plt.legend(['Win','Loss'])
for r1, r2 in zip(ax1, ax2):
    h1 = r1.get_height()
    h2 = r2.get_height()
    h3 = h1 + h2
    plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., str( round( (h1*100)/h3 ))+"%", ha="center", va="center", color="white", fontsize=12)
    plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., str( round( (h2*100)/h3 ))+"%", ha="center", va="center", color="black", fontsize=12)
    plt.text(r2.get_x() + r2.get_width() / 2., h3 + 1, "%d" % h3, ha="center", va="center", color="black", fontsize=12, fontweight="bold")
plt.savefig("TotalWinsVsLosses-since "+str(fromDate).split(" ")[0]+".png")
plt.show()

"""
Inferences:
A handful of teams NewZealand, England, India, Australia, South Africa are doing good in the recent times

Now let us see which team plays a lot at home vs away vs nuetral venues
"""
minNoOfMataches = 30
fromDate = pd.Timestamp(2015,1,1)
tws_df = teamWiseStatistics(fromDate, minNoOfMataches, ContinousDatasetModified_df )
tws_df = tws_df[['Played_Home','Played_Away','Played_Nuetral']].transpose()
print ("\nTeam wise total mataches played at Home/Away/Nuetral venues",tws_df)
tws_df.drop(tws_df.columns[[0, 1, -2, -1]], inplace = True, axis = 1)

#fig, (ax1, ax2) = plt.subplots(1, 2)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0,0].pie(tws_df['India'],  autopct='%1.1f%%', startangle=180, textprops={'fontsize': 12, 'fontweight':"bold"});
axes[0,0].set_title('India')
axes[0,1].pie(tws_df['Sri Lanka'],  autopct='%1.1f%%', startangle=180, textprops={'fontsize': 12, 'fontweight':"bold"});
axes[0,1].set_title('Sri Lanka')
axes[0,2].pie(tws_df['New Zealand'],  autopct='%1.1f%%', startangle=180, textprops={'fontsize': 12, 'fontweight':"bold"});
axes[0,2].set_title('New Zealand')
axes[0,3].pie(tws_df['West Indies'],  autopct='%1.1f%%', startangle=180, textprops={'fontsize': 12, 'fontweight':"bold"});
axes[0,3].set_title('West Indies')
axes[1,0].pie(tws_df['Pakistan'],  autopct='%1.1f%%', startangle=180, textprops={'fontsize': 12, 'fontweight':"bold"});
axes[1,0].set_title('Pakistan')
axes[1,1].pie(tws_df['Australia'],  autopct='%1.1f%%', startangle=180, textprops={'fontsize': 12, 'fontweight':"bold"});
axes[1,1].set_title('Australia')
axes[1,2].pie(tws_df['England'],  autopct='%1.1f%%', startangle=180, textprops={'fontsize': 12, 'fontweight':"bold"});
axes[1,2].set_title('England')
axes[1,3].pie(tws_df['South Africa'],  autopct='%1.1f%%', startangle=180, textprops={'fontsize': 12, 'fontweight':"bold"});
axes[1,3].set_title('South Africa')
plt.legend(tws_df.index)
fig.suptitle("Matches played at Home/Away/Nuetral venues by each team, since "+str(fromDate).split(" ")[0],fontsize=20, fontweight="bold")
plt.savefig("Teamwise-Home-Away-Nuetral"+str(fromDate).split(" ")[0]+".png")
plt.show()

"""
Inferences:
India plays almost the same amount of matches at all 3 types venues.
Australia and NewZealand played mostly either at Home or Away
Pakistan hosted a very few matches at home, due to security concerns in the recent past


Now lets look at how many wins each Team has recorded during 1990 till 2017
"""
def teamWiseWinsPerYear(fromDate, ContinousDatasetModified_df, tillDate = pd.Timestamp(2020,1,1)):
    ContinousDatasetModified_df  = ContinousDatasetModified_df[
        (ContinousDatasetModified_df['Match Date']>fromDate ) & (ContinousDatasetModified_df['Match Date']<tillDate) ]
    #print ("\nContinousDatasetModified_df (Date > "+str(fromDate)+" ):\n",ContinousDatasetModified_df)
    teamWiseWinsPerYear_series = ContinousDatasetModified_df['Winner'].value_counts()
    return teamWiseWinsPerYear_series


teamWiseWinsPerYear_df = pd.DataFrame()
for i in range(1990,2018):
    temp_df = teamWiseWinsPerYear(pd.Timestamp(i,1,1),
                                  ContinousDatasetModified_df,
                                  tillDate = pd.Timestamp(i+1,1,1)).rename_axis('Team').reset_index(name=str(i))
    if teamWiseWinsPerYear_df.empty:
        teamWiseWinsPerYear_df = temp_df
    else:
        teamWiseWinsPerYear_df = teamWiseWinsPerYear_df.merge(temp_df, on = "Team", how='left')#.fillna(0)
teamWiseWinsPerYear_df = teamWiseWinsPerYear_df.set_index('Team')
print ("\nTeam wise Total Wins in each year",teamWiseWinsPerYear_df)

plt.figure(figsize=(20,5))
sns.heatmap(data=teamWiseWinsPerYear_df, annot=True, cmap='Blues', linewidths=0.05)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.xlabel('Year', size=14, fontweight="bold")
plt.ylabel('Team', size=14, fontweight="bold")
plt.title('TeamWise Total Wins Per Year, during 1990-2017', size=20, fontweight="bold")
plt.savefig("teamWiseWinsPerYear_heatmap.png")
plt.show()

"""
Inferences:
Australia registered most no of wins in the years 1999, 2003, 2007. And it can be observed that the team has been doing consistently great during that period which also shows why Aus could win 3 consecutive world cups during that time


Generally a teams growth or performance is measured by away winning percentages. Lets draw a line chart how each of the top playing nations in terms of away win percentages
"""
startDate = pd.Timestamp(1971,1,1)
each_5yr_summary ={}
for i in range(0, 10):
    start_Date = pd.Timestamp(1971,1,1)+np.timedelta64(5*i, 'Y')
    till_Date = startDate+np.timedelta64(5*(i+1), 'Y')
    teamWiseStatistics_df = teamWiseStatistics(start_Date, 0, ContinousDatasetModified_df,
                                        tillDate = till_Date)
    teamWiseStatistics_df.drop(columns=['Total_Lost',  'Played_Home', 'Home_Wins', 'Home_Lost',
                                            'Away_Lost', 'Played_Nuetral', 'Nuetral_Wins','Nuetral_Lost'], inplace=True)
    decade = str(start_Date.year)+"-"+str(till_Date.year)

    teamWiseStatistics_df[decade] = (teamWiseStatistics_df['Away_Wins']/teamWiseStatistics_df['Played_Away'])*100
    each_5yr_summary[decade] = teamWiseStatistics_df[decade].to_dict()

transposed_df = pd.DataFrame(each_5yr_summary)[:6].transpose()
print ("\nTeam wise Away win % in each 5 year block:\n",transposed_df)
plt.figure(figsize=(16,8))
plt.plot(transposed_df, 's-',linewidth=2,dash_capstyle='round')
plt.xticks(rotation=35)
plt.xlabel('Progress between each 5yr period', fontsize=14, fontweight="bold")
plt.ylabel('Win percent',fontsize=14, fontweight="bold");
plt.title("Rise or Fall of Top teams during 1971-2017, based on Away win %", fontsize=20, fontweight="bold")
plt.legend(transposed_df.columns)
plt.savefig("Rise_Fall_basedon_Away_wins.png")
plt.show() 

"""
inferences:
Australia has been consistently good for the over all period, with above 50% away win rates except for a couple of time slots during 1976-80, 2015-17
India has been consistently doing good by improving the away winning percentage in each 5yr time slot
West Indies had recorded a decreasing trend of away win percentages


Now lets try to list the top grounds with hosted the most number of matches in the recent times.
And then lets predict if batting first or second should be considered based on statistics
"""
fromDate = pd.Timestamp(2010,1,1)
tillDate = pd.Timestamp(2018,12,31)
ContinousDatasetModified_df  = ContinousDatasetModified_df[
    (ContinousDatasetModified_df['Match Date']>fromDate ) & (ContinousDatasetModified_df['Match Date']<tillDate) ]    

df = pd.DataFrame({'count' : ContinousDatasetModified_df.groupby( [ 'Host_Country','Ground'] ).size()}).reset_index()
df.drop(df[df['count'] < 15 ].index, inplace=True)
   
pivoted = df.pivot('Host_Country', 'Ground', 'count')
pivoted.plot.bar()
plt.xticks(rotation=0)
plt.xlabel('Host Country', size=14, fontweight="bold")
plt.ylabel('No. of Matches Hosted', size=14, fontweight="bold")
plt.title('Country wise top Grounds during 2010-2017\n which hosted a minimum of 15 matches', size=20, fontweight="bold")
plt.savefig("Country wise Top Grounds since.png")
plt.show()

#Now for each of these top grounds lets find whether batting first recommended
df = ContinousDatasetModified_df.loc[ContinousDatasetModified_df['Ground'].isin(df['Ground'])]
df = pd.DataFrame({'count' : df.groupby( ['Host_Country', 'Ground','Margin'] ).size()}).reset_index()
print ("\nBatting 1st vs 2nd results, grouped by Grounds and Country:\n",df)
    
fig, ax1 = plt.subplots(figsize=(20,10))
graph = sns.barplot(ax=ax1,x = 'Ground', y = 'count', hue='Margin', data=df)
no_of_patches = len(graph.patches)
mid = int(no_of_patches/2)
for i in range (1, no_of_patches+1):
    p = graph.patches[i-1]
    height = p.get_height()
    if i-1 < mid:
        corresponding_patch = i-1+mid
    elif i-1 >= mid:
        corresponding_patch = i-1-mid
    percent = str(int( (  height/ (height+  (graph.patches[corresponding_patch].get_height() )  ) )*100))+"%"
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,int(height) ,ha="center")
    graph.text(p.get_x()+p.get_width()/2., height/2, percent ,ha="center", color="white")
    
plt.xticks(rotation=0)
plt.xlabel('Ground', size=14, fontweight="bold")
plt.ylabel('No. of Wins', size=14, fontweight="bold")
plt.title('Batting 1st vs 2nd Results at heavily played Grounds, during 2010-17', size=20, fontweight="bold")
plt.savefig("Batting1stvs2nd.png")
plt.show()

"""
Inferences:
Batting 2nd or the chasing teams won more frequently at Bulawayo, Harare, Dubai, Hambatonta, Oval
Batting 1st or the defending teams won more frequently at Melbourne, Dublin, Abudabi
"""