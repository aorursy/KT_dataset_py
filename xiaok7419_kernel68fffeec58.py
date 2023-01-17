import pandas as pd





def GetPlayerData(player):

    df_shots_player = None

    

    shot_logs_path = "../input/shot_logs.csv"

    df_shots = pd.read_csv(shot_logs_path)

    df_shots_player = df_shots[df_shots["player_name"] == player]

    

    return df_shots_player

df_shots_player = GetPlayerData("stephen curry")

assert df_shots_player.shape == (968, 21)

df_shots_player = GetPlayerData("brian roberts")

assert df_shots_player.shape == (372, 21)
def AddMonth(df_shots_player):

    d = {'JAN':1, 'FEB':2, 'MAR':3, 'APR':4, 'MAY':5, 'JUN': 6, 'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12 }



    df_shots_player["Month"] = df_shots_player.MATCHUP.apply(lambda x:d[x.split()[0]])

    

    return df_shots_player
df_shots_player = GetPlayerData("stephen curry")

df_shots_player = AddMonth(df_shots_player)

assert round(df_shots_player["Month"].describe()["mean"], 2) == 6.65



df_shots_player = AddMonth(GetPlayerData("brian roberts"))

assert round(df_shots_player["Month"].describe()["mean"], 2) == 6.69
def GetShotType(row):

    """

    This function takes a row of the dataset and returns

    "3-pointer", "short" or "mid-range" depending on the shot

    type.

    """

    return_type = "3-pointer"



    if(row.PTS_TYPE == 2 and row.SHOT_DIST <= 7):

        return_type = "short"

    elif(row.PTS_TYPE == 2 and row.SHOT_DIST > 7):

        return_type = "mid-range"

    

    return return_type



def AddShotType(df_shots_player):

    

    types = []

    for index,row in df_shots_player.iterrows():

        types.append(GetShotType(row))



    df_shots_player.insert(0,"Shot_Type",types)



    return df_shots_player
import pandas as pd



df_shots_player = GetPlayerData("stephen curry")

df_shots_player = AddMonth(df_shots_player)

df_shots_player = AddShotType(df_shots_player)



assert df_shots_player[df_shots_player["Shot_Type"] == "3-pointer"].shape[0] == 456

assert df_shots_player[df_shots_player["Shot_Type"] == "short"].shape[0] == 224

assert df_shots_player[df_shots_player["Shot_Type"] == "mid-range"].shape[0] == 288
def ComputeGoalType(df_shots_player):

    FG_perc_by_month = None

    

    months = []

    perc_3_point = []

    perc_mid = []

    perc_short = []

    for i in range(1,13):

        month_i_short = df_shots_player[(df_shots_player["Month"]==i) & (df_shots_player["Shot_Type"]=="short")]

        month_i_mid = df_shots_player[(df_shots_player["Month"]==i) & (df_shots_player["Shot_Type"]=="mid-range")]

        month_i_3_point = df_shots_player[(df_shots_player["Month"]==i) & (df_shots_player["Shot_Type"]=="3-pointer")]

        if(month_i_3_point.shape[0]>=30 and month_i_mid.shape[0]>=30 and month_i_short.shape[0]>=30):

            months.append(i)

            perc_3_point.append(month_i_3_point[month_i_3_point.SHOT_RESULT == "made"].shape[0]/month_i_3_point.shape[0])

            perc_mid.append(month_i_mid[month_i_mid.SHOT_RESULT == "made"].shape[0]/month_i_mid.shape[0])

            perc_short.append(month_i_short[month_i_short.SHOT_RESULT == "made"].shape[0]/month_i_short.shape[0])



    d={"Month":months, '3-pointer':perc_3_point, 'mid-range':perc_mid, 'short':perc_short}

    FG_perc_by_month=pd.DataFrame(data=d)

    

    return FG_perc_by_month
df_shots_player = GetPlayerData("stephen curry")

df_shots_player = AddMonth(df_shots_player)

df_shots_player = AddShotType(df_shots_player)

FG_perc_by_month = ComputeGoalType(df_shots_player)

assert [round(x, 2) for x in FG_perc_by_month["3-pointer"].tolist()] == [0.41, 0.47, 0.43, 0.35]

assert [round(x, 2) for x in FG_perc_by_month["short"].tolist()] == [0.65, 0.65, 0.65, 0.70]