# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/LeagueofLegends.csv')
def reset_and_drop(in_df):

    in_df = in_df.reset_index()

    in_df.drop('index', axis=1, inplace=True)

    return in_df



def select_team(team_name, in_df):

    return_df = in_df[ (in_df['blueTeamTag'] == team_name) | (in_df['redTeamTag'] == team_name) ]

    return_df = reset_and_drop(return_df)

    return return_df



def select_split(split, year, in_df):

    in_df = in_df[ (in_df["Season"] == split) & (in_df["Year"] == year) ]

    in_df = reset_and_drop(in_df)

    return in_df



def calc_team_win_loss(team_name, team_df):

    team_df['Win/Loss'] = ( ((team_df['blueTeamTag'] == team_name) & (team_df['bResult'] == "1")) | ((team_df['redTeamTag'] == team_name) & (team_df['rResult'] == "1")))

    team_df['Win/Loss'] = team_df['Win/Loss'].apply(lambda x : 1 if x is True else -1)

    team_df['Win/Loss'] = team_df['Win/Loss'].cumsum()

    return team_df



def calc_wins(in_df):

    in_df['Wins'] = ((in_df.index+1) + in_df['Win/Loss']) / 2

    return in_df



def calc_losses(in_df):

    in_df['Losses'] = ((in_df.index+1) - in_df['Win/Loss']) / 2

    return in_df



def print_record(team_name, split, year):

    team_df = select_team(team_name, in_df=df)

    team_df = select_split(split, year, team_df)

    team_df = calc_team_win_loss(team_name, team_df)

    calc_wins(team_df)

    calc_losses(team_df)

    print ("{}: Wins {}: Losses".format( team_df['Wins'][len(team_df['Wins'])-1] , team_df['Losses'][len(team_df['Losses'])-1] ))

    

def record_over_time(team_name, split, year):

    team_df = select_team(team_name=team_name, in_df=df)

    team_df = select_split(split, year, team_df)

    team_df = calc_team_win_loss(team_name, team_df)

    team_df.plot.line(y='Win/Loss')
print_record("TSM", "Spring_Season", "2016")
record_over_time("TSM", "Spring_Season", "2016")
# This function should display a graph of the Win/Loss record of a champion

# By default the function will display Aatrox's record throughout the dataset

def champion_record(champ_name="Aatrox", player_name="All", team_name="All", split="All", year="All", graph=True):

    champ_df = select_champ(champ_name, df)

    

    if (player_name != "All"):

        champ_df = select_player_playing_champ(player_name, champ_name, champ_df)

    if (team_name != "All"):

        champ_df = select_team(team_name, champ_df)

    # If split is present, but year is not, the year will default to 2015

    if (split != "All") & (year != "All"):

        champ_df = select_split(split, year, champ_df)

    elif (split != "All"):

            champ_df = select_split(split, "2015", champ_df)

    elif (year != "All"):

            champ_df2 = champ_df.copy()

            spring_df = select_split("Spring_Season", year, champ_df2)

            summer_df = select_split("Summer_Season", year, champ_df)

            champ_df = pd.concat([summer_df, spring_df], axis=0)

            

    champ_df = reset_and_drop(champ_df)

    calc_champ_win_loss(champ_name, champ_df)

    

    if(graph==True):

        champ_df.plot.line(y="Win/Loss")

    else:

        calc_wins(champ_df)

        calc_losses(champ_df)

        print("Wins: {} Losses: {}".format(champ_df['Wins'][len(champ_df['Wins'])-1],champ_df['Losses'][len(champ_df['Losses'])-1]))





def select_player(player_name, in_df):

    return in_df[ player_is_on_blue(player_name, in_df) | player_is_on_red(player_name, in_df) ]



def select_player_playing_champ(player_name, champ_name, in_df):

    bool_series = ((in_df["blueTopChamp"] == champ_name) & (in_df["blueTop"] == player_name) |

                ((in_df["blueJungleChamp"] == champ_name) & (in_df["blueJungle"] == player_name)) |

                ((in_df["blueMiddleChamp"] == champ_name) & (in_df["blueMiddle"] == player_name)) |

                ((in_df["blueADCChamp"] == champ_name) & (in_df["blueADC"] == player_name)) |

                ((in_df["blueSupportChamp"] == champ_name) & (in_df["blueSupport"] == player_name)) |

                   

                ((in_df["redTopChamp"] == champ_name) & (in_df["redTop"] == player_name)) |

                ((in_df["redJungleChamp"] == champ_name) & (in_df["redJungle"] == player_name)) |

                ((in_df["redMiddleChamp"] == champ_name) & (in_df["redMiddle"] == player_name)) |

                ((in_df["redADCChamp"] == champ_name) & (in_df["redADC"] == player_name)) |

                ((in_df["redSupportChamp"] == champ_name) & (in_df["redSupport"] == player_name)))

    return in_df[ bool_series ]



def player_is_on_blue(player_name, in_df):

    return ((in_df["blueTop"] == player_name) | (in_df["blueJungle"] == player_name) | (in_df["blueMiddle"] == player_name) | (in_df["blueADC"] == player_name) | (in_df["blueSupport"] == player_name))



def player_is_on_red(player_name, in_df):

    return ((in_df["redTop"] == player_name) | (in_df["redJungle"] == player_name) | (in_df["redMiddle"] == player_name) | (in_df["redADC"] == player_name) | (in_df["redSupport"] == player_name))        



def select_champ(champ_name, in_df):

    return in_df[ champ_is_on_blue(champ_name, in_df) | champ_is_on_red(champ_name, in_df) ]



def calc_champ_win_loss(champ_name, in_df):

    in_df['Win/Loss'] = ( ( champ_is_on_blue(champ_name, in_df) & blue_won(in_df) ) | ( champ_is_on_red(champ_name, in_df) & red_won(in_df) ))

    in_df['Win/Loss'] = in_df['Win/Loss'].apply(lambda x : 1 if x is True else -1)

    in_df['Win/Loss'] = in_df['Win/Loss'].cumsum()

    return in_df



def champ_is_on_blue(champ_name, in_df):

    return ((in_df["blueTopChamp"] == champ_name) | (in_df["blueJungleChamp"] == champ_name) | (in_df["blueMiddleChamp"] == champ_name) | (in_df["blueADCChamp"] == champ_name) | (in_df["blueSupportChamp"] == champ_name))



def champ_is_on_red(champ_name, in_df):

    return ((in_df["redTopChamp"] == champ_name) | (in_df["redJungleChamp"] == champ_name) | (in_df["redMiddleChamp"] == champ_name) | (in_df["redADCChamp"] == champ_name) | (in_df["redSupportChamp"] == champ_name))



def red_won(in_df):

    return in_df["rResult"] == "1"



def blue_won(in_df):

    return in_df["bResult"] == "1"

champion_record(champ_name="Ashe")
champion_record("Maokai", player_name="Balls")
champion_record(champ_name="Maokai", team_name="FLY")
champion_record("Fizz", player_name="Faker")
champion_record("Gnar", split="Spring_Season")
champion_record(champ_name="Ashe", player_name="WildTurtle", team_name="IMT")
champion_record(champ_name="Ashe", player_name="WildTurtle", team_name="TSM")
champion_record("Ashe", graph=False)
champion_record("Maokai", player_name="Balls", graph=False)
champion_record("Maokai", team_name="FLY", graph=False)
champion_record("Gnar", split="Spring_Season", graph=False)
champion_record(champ_name="Leblanc", player_name="Bjergsen", year="2015", graph = False)
champion_record(champ_name="Ahri", player_name="Bjergsen", year="2015", split="Spring_Season", graph = False)
champion_record(champ_name="Ashe", player_name="WildTurtle", team_name="IMT", graph=False)
champion_record(champ_name="Varus", player_name="WildTurtle", team_name="TSM", graph=False)