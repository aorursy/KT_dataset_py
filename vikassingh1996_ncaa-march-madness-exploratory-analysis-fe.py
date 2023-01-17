'''Import basic modules.'''

import pandas as pd

import numpy as np



'''visualization Tools'''

import altair as alt

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

%matplotlib inline



'''Bokeh (interactive visualization)'''

from bokeh.plotting import figure, show,output_notebook, ColumnDataSource

from bokeh.layouts import row

output_notebook() 



'''Plotly visualization .'''

import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

py.init_notebook_mode(connected = True)
mteams = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MTeams.csv')

mseasons = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MSeasons.csv')

mtourney_seed = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MNCAATourneySeeds.csv')

mseason_results = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')

mtourney_results = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')

conference = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/Conferences.csv')

team_conference = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MTeamConferences.csv')
# data preparation

wteam = mtourney_results.rename(columns={'WTeamID':'TeamID'}) # rename the WteamID as TeamID for merge with mteams dataframe

win_team =  wteam.merge(mteams, on='TeamID')                  # merge with mteams dataframe

#win_team = win_team.rename(columns={'TeamID':'WTeamID_X'})

temp = win_team['TeamName'].value_counts().head(15).reset_index()



# Create ColumnDataSource from data frame

source = ColumnDataSource(temp)

win_team_list = source.data['index'].tolist()



# Add Plot

p = figure(

    y_range = win_team_list,

    plot_width=600,

    plot_height=320,

    title="Most Tournament Wins since 1985",

    y_axis_label='Winners',

    tools=""

    

)

p.title.text_font = 'helvetica'

p.title.text_font_size = '12pt'

p.title.text_font_style = "bold"



p.hbar(

        y='index',

        right='TeamName',

        height=0.8,

        color = 'blue',

        line_color="black", 

        line_width=1,

        fill_alpha=0.7,

        source = source

)

show(p)



# data preparation

lteam = mtourney_results.rename(columns={'LTeamID':'TeamID'})

lost_team =  lteam.merge(mteams, on='TeamID')

lost_team = lost_team.rename(columns={'TeamID':'LTeamID_X'})

temp = lost_team['TeamName'].value_counts().head(15).reset_index()



# Create ColumnDataSource from data frame

source = ColumnDataSource(temp)

lost_team_list = source.data['index'].tolist()



# Add Plot

p = figure(

    y_range = lost_team_list,

    plot_width=600,

    plot_height=320,

    title="Most Tournament Lost since 1985",

    y_axis_label='Runner-Up',

    tools=""

    

)

p.title.text_font = 'helvetica'

p.title.text_font_size = '12pt'

p.title.text_font_style = "bold"



p.hbar(

        y='index',

        right='TeamName',

        height=0.8,

        color = 'orange',

        line_color="black", 

        line_width=1,

        fill_alpha=0.7,

        source = source

)



show(p)
#data preparation

ncaa_win_camp = win_team[win_team['DayNum'] ==154]['TeamName'].value_counts().reset_index()



# Create ColumnDataSource from data frame

source = ColumnDataSource(ncaa_win_camp )

win_camp_list = source.data['index'].tolist()



# Add Plot

p = figure(

    y_range = win_camp_list,

    plot_width=600,

    plot_height=320,

    title="Tournament Championship Wins since 1985",

    y_axis_label='Winners',

    tools=""

    

)

p.title.text_font = 'helvetica'

p.title.text_font_size = '12pt'

p.title.text_font_style = "bold"



p.hbar(

        y='index',

        right='TeamName',

        height=0.8,

        color = 'blue',

        line_color="black", 

        line_width=1,

        fill_alpha=0.7,

        source = source

)



show(p)



#data preparation

ncaa_lost_camp  = lost_team[lost_team['DayNum']==154]['TeamName'].value_counts().reset_index()



# Create ColumnDataSource from data frame

source = ColumnDataSource(ncaa_lost_camp)

lost_camp_list = source.data['index'].tolist()



# Add Plot

p = figure(

    y_range = lost_camp_list,

    plot_width=600,

    plot_height=300,

    title="Tournament Championship Lost since 1985",

    y_axis_label='Runner-Up',

    tools=""

    

)

p.title.text_font = 'helvetica'

p.title.text_font_size = '12pt'

p.title.text_font_style = "bold"



p.hbar(

        y='index',

        right='TeamName',

        height=0.8,

        color = 'orange',

        line_color="black", 

        line_width=1,

        fill_alpha=0.7,

        source = source

)



show(p)
mtourney_seed['Region'] = mtourney_seed['Seed'].apply(lambda x: x[0][:1])

mtourney_seed['Seed'] = mtourney_seed['Seed'].apply(lambda x: int(x[1:3]))
#data preparation

seed_win_team =  win_team.merge(mtourney_seed, on=['TeamID','Season'])

seed_win_camp = seed_win_team[seed_win_team['DayNum'] ==154]['Seed'].value_counts().reset_index()



seed = list(seed_win_camp['index'].astype(str))

count =  list(seed_win_camp['Seed'])



#plot

dot = figure(title="Seeds With The Most Titles since 1985", tools="", toolbar_location=None,

            y_range=seed, x_range=[0,25], plot_width=600, plot_height=400,)

dot.title.text_font = 'helvetica'

dot.title.text_font_size = '12pt'

dot.title.text_font_style = "bold"



dot.segment(0, seed, count, seed, line_width=3, line_color="green", )

dot.circle(count, seed, size=15, fill_color="orange", line_color="green", line_width=3, )



show(dot)

#----------------------------------------

from bokeh.models import LabelSet

#data preparation

team_conf =  team_conference.merge(conference, on='ConfAbbrev') 

conf_win_team =  win_team.merge(team_conf, on=['TeamID', 'Season'])

conf_win_camp = conf_win_team[conf_win_team['DayNum']==154]['Description'].value_counts().reset_index()



# Create ColumnDataSource from data frame

source = ColumnDataSource(conf_win_camp)

conf_team_list = source.data['index'].tolist()



# Add Plot

p = figure(

    y_range = conf_team_list,

    plot_width=800,

    plot_height=400,

    title="'NCAA Championships by Conference 1985",

    tools=""

    

)

p.title.text_font = 'helvetica'

p.title.text_font_size = '12pt'

p.title.text_font_style = "bold"



p.hbar(

        y='index',

        right='Description',

        height=0.8,

        color = 'green',

        line_color="black", 

        line_width=1,

        fill_alpha=0.7,

        source = source

)

labels = LabelSet(y='index', x='Description', text='Description', x_offset=-18, y_offset=-5.5, source=source, render_mode='canvas')

p.add_layout(labels)

show(p)
conf_win_team = conf_win_team[conf_win_team['DayNum']==154]

temp_df = pd.crosstab(conf_win_team.Description, conf_win_team.TeamName)

plt.rcParams['figure.figsize'] = (8,8)

sns.set_style("white")

sns.heatmap(temp_df, cmap="YlGnBu", annot=True,fmt='g', cbar=False)

plt.xlabel('Team Name',fontsize = 20)

plt.ylabel('Conference',fontsize = 20)

plt.title('Conference Matchups With Teams NCAA Tournament', fontsize = 20)

plt.show()
tour_results = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv')

season_results = pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MRegularSeasonDetailedResults.csv')

season_results.head()
#Points Winning/Losing Team

season_results['WPts'] = season_results.apply(lambda row: 2*row.WFGM + row.WFGM3 + row.WFTM, axis=1)

season_results['LPts'] = season_results.apply(lambda row: 2*row.LFGM + row.LFGM3 + row.LFTM, axis=1)



#Calculate Winning/losing Team Possesion Feature

wPos = season_results.apply(lambda row: 0.96*(row.WFGA + row.WTO + 0.44*row.WFTA - row.WOR), axis=1)

lPos = season_results.apply(lambda row: 0.96*(row.LFGA + row.LTO + 0.44*row.LFTA - row.LOR), axis=1)

#two teams use almost the same number of possessions in a game

#(plus/minus one or two - depending on how quarters end)

#so let's just take the average

season_results['Pos'] = (wPos+lPos)/2



'''Advanced Metrics'''

#Offensive efficiency (OffRtg) = 100 x (Points / Possessions)

season_results['WOffRtg'] = season_results.apply(lambda row: 100 * (row.WPts / row.Pos), axis=1)

season_results['LOffRtg'] = season_results.apply(lambda row: 100 * (row.LPts / row.Pos), axis=1)



#Defensive efficiency (DefRtg) = 100 x (Opponent points / Opponent possessions)

season_results['WDefRtg'] = season_results.LOffRtg

season_results['LDefRtg'] = season_results.WOffRtg



#Net Rating = Off.Rtg - Def.Rtg

season_results['WNetRtg'] = season_results.apply(lambda row:(row.WOffRtg - row.WDefRtg), axis=1)

season_results['LNetRtg'] = season_results.apply(lambda row:(row.LOffRtg - row.LDefRtg), axis=1)

                         

#Assist Ratio : Percentage of team possessions that end in assists

season_results['WAstR'] = season_results.apply(lambda row: 100 * row.WAst / (row.WFGA + 0.44*row.WFTA + row.WAst + row.WTO), axis=1)

season_results['LAstR'] = season_results.apply(lambda row: 100 * row.LAst / (row.LFGA + 0.44*row.LFTA + row.LAst + row.LTO), axis=1)



#Turnover Ratio: Number of turnovers of a team per 100 possessions used.

#(TO * 100) / (FGA + (FTA * 0.44) + AST + TO)

season_results['WTOR'] = season_results.apply(lambda row: 100 * row.WTO / (row.WFGA + 0.44*row.WFTA + row.WAst + row.WTO), axis=1)

season_results['LTOR'] = season_results.apply(lambda row: 100 * row.LTO / (row.LFGA + 0.44*row.LFTA + row.LAst + row.LTO), axis=1)



#The Shooting Percentage : Measure of Shooting Efficiency (FGA/FGA3, FTA)

season_results['WTSP'] = season_results.apply(lambda row: 100 * row.WPts / (2 * (row.WFGA + 0.44 * row.WFTA)), axis=1)

season_results['LTSP'] = season_results.apply(lambda row: 100 * row.LPts / (2 * (row.LFGA + 0.44 * row.LFTA)), axis=1)



#eFG% : Effective Field Goal Percentage adjusting for the fact that 3pt shots are more valuable 

season_results['WeFGP'] = season_results.apply(lambda row:(row.WFGM + 0.5 * row.WFGM3) / row.WFGA, axis=1)      

season_results['LeFGP'] = season_results.apply(lambda row:(row.LFGM + 0.5 * row.LFGM3) / row.LFGA, axis=1)   



#FTA Rate : How good a team is at drawing fouls.

season_results['WFTAR'] = season_results.apply(lambda row: row.WFTA / row.WFGA, axis=1)

season_results['LFTAR'] = season_results.apply(lambda row: row.LFTA / row.LFGA, axis=1)

                         

#OREB% : Percentage of team offensive rebounds

season_results['WORP'] = season_results.apply(lambda row: row.WOR / (row.WOR + row.LDR), axis=1)

season_results['LORP'] = season_results.apply(lambda row: row.LOR / (row.LOR + row.WDR), axis=1)



#DREB% : Percentage of team defensive rebounds

season_results['WDRP'] = season_results.apply(lambda row: row.WDR / (row.WDR + row.LOR), axis=1)

season_results['LDRP'] = season_results.apply(lambda row: row.LDR / (row.LDR + row.WOR), axis=1)                                      



#REB% : Percentage of team total rebounds

season_results['WRP'] = season_results.apply(lambda row: (row.WDR + row.WOR) / (row.WDR + row.WOR + row.LDR + row.LOR), axis=1)

season_results['LRP'] = season_results.apply(lambda row: (row.LDR + row.LOR) / (row.WDR + row.WOR + row.LDR + row.LOR), axis=1) 
from matplotlib.font_manager import FontProperties

font = FontProperties()

font.set_family('serif')

sns.set_style("whitegrid")

f, axes = plt.subplots (5,2, figsize=(10,20))  



ax1 = sns.kdeplot(season_results['WPts'], shade = True,ax=axes[0,0], label="W",color = 'k')

ax1 = sns.kdeplot(season_results['LPts'], shade = True,ax=axes[0,0], label="L", color = 'm')

ax1.set(xlabel='Points')

axes[0,0].set_title('Macth Points',loc='left',fontsize=15, FontProperties=font, fontweight='bold')

plt.setp(ax1.get_legend().get_texts(), fontsize='10')



ax2 = sns.kdeplot(wPos, shade = True,ax=axes[0,1], label="W",color = 'k')

ax2 = sns.kdeplot(lPos, shade = True,ax=axes[0,1], label="L", color = 'm')

ax2.set(xlabel='Possesion Points')

axes[0,1].set_title('Team Possesion',loc='left',fontsize=15, FontProperties=font, fontweight='bold')

plt.setp(ax2.get_legend().get_texts(), fontsize='10')



ax3 = sns.kdeplot(season_results['WOffRtg'], shade = True,ax=axes[1,0], label="W",color = 'k')

ax3 = sns.kdeplot(season_results['LOffRtg'], shade = True,ax=axes[1,0], label="L", color = 'm')

ax3.set(xlabel='Offensive efficiency')

axes[1,0].set_title('Offensive Efficiency',loc='left',fontsize=15, FontProperties=font, fontweight='bold')

plt.setp(ax3.get_legend().get_texts(), fontsize='10')



ax4 = sns.kdeplot(season_results['WDefRtg'], shade = True,ax=axes[1,1], label="W",color = 'k')

ax4 = sns.kdeplot(season_results['LDefRtg'], shade = True,ax=axes[1,1], label="L", color = 'm')

ax4.set(xlabel='Defensive efficiency')

axes[1,1].set_title('Defensive Efficiency',loc='left',fontsize=15, FontProperties=font, fontweight='bold')

plt.setp(ax4.get_legend().get_texts(), fontsize='10')



ax5 = sns.kdeplot(season_results['WAstR'], shade = True,ax=axes[2,0], label="W",color = 'k')

ax5 = sns.kdeplot(season_results['LAstR'], shade = True,ax=axes[2,0], label="L", color = 'm')

ax5.set(xlabel='Assist')

axes[2,0].set_title('Assist Ratio',loc='left',fontsize=15, FontProperties=font, fontweight='bold')

plt.setp(ax5.get_legend().get_texts(), fontsize='10')



ax6 = sns.kdeplot(season_results['WTOR'], shade = True,ax=axes[2,1], label="W",color = 'k')

ax6 = sns.kdeplot(season_results['LTOR'], shade = True,ax=axes[2,1], label="L", color = 'm')

ax6.set(xlabel='Turnover')

axes[2,1].set_title('Turnover Ratio',loc='left',fontsize=15, FontProperties=font, fontweight='bold')

plt.setp(ax6.get_legend().get_texts(), fontsize='10')



ax7 = sns.kdeplot(season_results['WTSP'], shade = True,ax=axes[3,0], label="W",color = 'k')

ax7 = sns.kdeplot(season_results['LTSP'], shade = True,ax=axes[3,0], label="L", color = 'm')

ax7.set(xlabel='Shooting')

axes[3,0].set_title('Goal Shooting Percentage',loc='left',fontsize=15, FontProperties=font, fontweight='bold')

plt.setp(ax7.get_legend().get_texts(), fontsize='10')



ax8 = sns.kdeplot(season_results['WeFGP'], shade = True,ax=axes[3,1], label="W",color = 'k')

ax8 = sns.kdeplot(season_results['LeFGP'], shade = True,ax=axes[3,1], label="L", color = 'm')

ax8.set(xlabel='3pt Goal Shots %')

axes[3,1].set_title('3pt Goal Shots',loc='left',fontsize=15, FontProperties=font, fontweight='bold')

plt.setp(ax8.get_legend().get_texts(), fontsize='10')



ax9 = sns.kdeplot(season_results['WORP'], shade = True,ax=axes[4,0], label="W",color = 'k')

ax9 = sns.kdeplot(season_results['LORP'], shade = True,ax=axes[4,0], label="L", color = 'm')

ax9.set(xlabel='Offensive %')

axes[4,0].set_title('Offensive Rebounding  Efficiency',loc='left',fontsize=15, FontProperties=font, fontweight='bold')

plt.setp(ax9.get_legend().get_texts(), fontsize='10')



ax10 = sns.kdeplot(season_results['WDRP'], shade = True,ax=axes[4,1], label="W",color = 'k')

ax10 = sns.kdeplot(season_results['LDRP'], shade = True,ax=axes[4,1], label="L", color = 'm')

ax10.set(xlabel='Defensive %')

axes[4,1].set_title('Defensive Rebounding Efficiency',loc='left',fontsize=15, FontProperties=font, fontweight='bold')

plt.setp(ax10.get_legend().get_texts(), fontsize='10')



plt.subplots_adjust(wspace=0.4)

plt.subplots_adjust(hspace=0.4)

plt.show()
sns.set_style("white")

f,axes = plt.subplots(1,4, figsize=(12,12)) 



Wabsolute_values = season_results[['WScore','WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 

                          'WFTA', 'WOR', 'WDR','WAst', 'WTO', 'WStl', 'WBlk', 'WPF']]

Wrelative_values = season_results[['WPts','Pos', 'WOffRtg', 'WDefRtg', 'WNetRtg','WAstR', 

                          'WTOR', 'WTSP', 'WeFGP', 'WFTAR', 'WORP', 'WDRP', 'WRP']]

Labsolute_values = season_results[['LScore','LFGM', 'LFGA', 'LFGM3', 'LFGA3','LFTM', 

                                   'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF',]]

Lrelative_values = season_results[['LPts','Pos', 'LOffRtg', 'LDefRtg', 'LNetRtg','LAstR', 

                          'LTOR', 'LTSP', 'LeFGP', 'LFTAR', 'LORP', 'LDRP', 'LRP']]



corr1 = Wabsolute_values.corr()

mask = np.zeros_like(corr1, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

ax1 = sns.heatmap(corr1, mask=mask, cmap="RdGy",linewidths=.5, 

                  vmin=0, vmax=1,ax=axes[0],cbar=False,square=True)

ax1.tick_params(axis='both', which='major', labelsize=8) 

ax1.set_title("Winning Absolute Variables", fontsize=8,

              FontProperties=font, fontweight='bold')





corr2 = Wrelative_values.corr()

mask = np.zeros_like(corr2, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

ax2 = sns.heatmap(corr2, mask=mask, cmap="RdGy",linewidths=.5, 

                  vmin=0, vmax=1,ax=axes[1],cbar=False,square=True)

ax2.tick_params(axis='both', which='major', labelsize=8) 

ax2.set_title("Winning Relative Variables", fontsize=8,

             FontProperties=font, fontweight='bold')





corr3 = Labsolute_values.corr()

mask = np.zeros_like(corr3, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

ax3 = sns.heatmap(corr3, mask=mask, cmap="RdGy",linewidths=.5, 

                  vmin=0, vmax=1,ax=axes[2],cbar=False,square=True)

ax3.tick_params(axis='both', which='major', labelsize=8) 

ax3.set_title("Losing Absolute Variables", fontsize=8,

             FontProperties=font, fontweight='bold')





corr4 = Lrelative_values.corr()

mask = np.zeros_like(corr4, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

ax4 = sns.heatmap(corr4, mask=mask, cmap="RdGy",linewidths=.5, 

                  vmin=0, vmax=1,ax=axes[3],square=True, cbar = False) 

ax4.tick_params(axis='both', which='major', labelsize=8) 

ax4.set_title("Losing Relative Variables", fontsize=8,

             FontProperties=font, fontweight='bold')





plt.subplots_adjust(hspace=1)

plt.show()
win_s = season_results.groupby(['Season','WTeamID'])['WTeamID'].count().to_frame('rsW')

win_t = tour_results.groupby(['Season','WTeamID'])['WTeamID'].count().to_frame('tW')

win_team = win_s.merge(win_t,on=['Season','WTeamID']).reset_index()



sns.set_style("whitegrid")

ax=sns.lmplot(x='rsW' , y='tW', data=win_team, col='Season', col_wrap=4,line_kws={'color': 'black'},)

plt.show()
w_season_results = season_results.rename(columns={'WTeamID':'TeamID'}) # rename the WteamID as TeamID for merge with mteams dataframe

st_seed_win_team = w_season_results.merge(mtourney_seed, on=['TeamID','Season'])

st_seed_win_team = st_seed_win_team.rename(columns={'TeamID':'WTeamID'})

seed_win_team = st_seed_win_team.merge(win_t,on=['Season','WTeamID']).reset_index()



sns.set_style("whitegrid")

plt.figure(figsize=(15,10))

ax=sns.regplot(x='Seed' , y='tW', marker='o', data=seed_win_team, line_kws={'color': 'black'})

ax.set_title('Tournament Wins by Seed',loc='left',fontsize=30, FontProperties=font, fontweight='bold')

ax.set_xlabel('Seed',fontsize=12)

ax.set_ylabel('Tournament Wins', fontsize=12)

plt.show()
tw_season_results = season_results.merge(win_t,on=['Season','WTeamID']).reset_index()

temp = tw_season_results[tw_season_results['tW'] >=4]



h1 = sns.jointplot(x="WFGM", y="WFTA", data=temp,

                  kind="scatter",color='k',s=10,height=6)

h1.ax_marg_x.hist(temp['WFGM'], color="steelblue", alpha=.6)

h1.ax_marg_y.hist(temp['WFTA'], color="steelblue", alpha=.6,

                     orientation="horizontal")

h1.fig.suptitle('Regular Season Shooting Performance of Tournament Teams',fontsize=20, FontProperties=font, fontweight='bold')

h1.set_axis_labels('Field Goals','Free Throws Attempted', fontsize=12)

h1.fig.subplots_adjust(top=0.9)



h2 = sns.jointplot(x="WFGM", y="WAst", data=temp,

                  kind="scatter",color='k',s=10,height=6)

h2.ax_marg_x.hist(temp['WFGM'], color="darkred", alpha=.6)

h2.ax_marg_y.hist(temp['WAst'], color="darkred", alpha=.6,

                     orientation="horizontal")

h2.fig.suptitle('Regular Season Assist Performance of Tournament Teams',fontsize=20, FontProperties=font, fontweight='bold')

h2.set_axis_labels('Field Goals','Assist', fontsize=12)

h2.fig.subplots_adjust(top=0.9)



h3 = sns.jointplot(x="WFGM", y="WStl", data=temp,

                  kind="scatter",color='k',s=10,height=6)

h3.ax_marg_x.hist(temp['WFGM'], color="darkgreen", alpha=.6)

h3.ax_marg_y.hist(temp['WStl'], color="darkgreen", alpha=.6,

                     orientation="horizontal")

h3.fig.suptitle('Regular Season Steals Performance of Tournament Teams',fontsize=20, FontProperties=font, fontweight='bold')

h3.set_axis_labels('Field Goals','Steal', fontsize=12)

h3.fig.subplots_adjust(top=0.9)