#ライブラリをインポートする。

import pandas as pd

import numpy as np

from scipy import stats

import plotly.graph_objects as go

from plotly.subplots import make_subplots



import gc
#PlayList.csvをデータフレーム化する。

PlayList = pd.read_csv('../input/nfl-playing-surface-analytics/PlayList.csv')

#InjuryRecord.csvをデータフレーム化する。

InjuryRecord = pd.read_csv('../input/nfl-playing-surface-analytics/InjuryRecord.csv')
#データフレーム生成時のメモリを制御する。

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df



#怪我率を算出したデータフレームを作成する関数。

def make_inj_rate_df(df, var):

    all_sum = pd.DataFrame(df[var].value_counts())

    all_sum = all_sum.rename(columns={var:'all_sum'}).reset_index()



    inj = df[df['inj_flg'] == '1']

    inj_sum = pd.DataFrame(inj[var].value_counts())

    inj_sum = inj_sum.rename(columns={var:'inj_sum'}).reset_index()



    inj_rate_df = pd.merge(all_sum, inj_sum, on='index', how='left')

    inj_rate_df = inj_rate_df.fillna(0)

    inj_rate_df['inj_rate'] = inj_rate_df['inj_sum'] / inj_rate_df['all_sum'] * 100

    inj_rate_df = inj_rate_df.rename(columns={'index':var})

    return inj_rate_df



#対象変数の数と怪我率のグラフを作成する関数。

def make_graph_number_and_injury_rate(graph_data, var, title, yaxis):

    fig = make_subplots(specs=[[{"secondary_y": True}]])



    fig.add_trace(go.Scatter(x=graph_data[var], y=graph_data['inj_rate'], name='Injury rate',

                             text=graph_data['inj_rate'], textposition='top right'), secondary_y=True)

    fig.update_traces(marker=dict(size=16))



    fig.add_trace(go.Bar(x=graph_data[var], y=graph_data['all_sum'], name=yaxis,

                         text=graph_data['all_sum'], textposition='auto'), secondary_y=False)



    if var == 'RosterPosition':

        graph_data['inj_rate'] = 105 / 255 * 100

    elif var == 'FieldType' or var == 'StadiumType' or var == 'Weather':

        graph_data['inj_rate'] = 104 / 5712 * 100

    elif var == 'PlayType':

        graph_data['inj_rate'] = 77 / 267006 * 100

    elif var == 'event':

        graph_data['inj_rate'] = 490 / 1839873 * 100

    

    fig.add_trace(go.Scatter(x=graph_data[var], y=graph_data['inj_rate'], mode='lines', name='Average of injury rate'), secondary_y=True)

    

    fig.update_layout(

        title_text=title,

        xaxis=dict(showline=True, showgrid=False, showticklabels=True,),

        #legend_orientation="h",

        plot_bgcolor='white'

    )



    fig.update_yaxes(title_text=yaxis, secondary_y=False)

    fig.update_yaxes(title_text="Injury Rate", secondary_y=True)



    fig.show()



#フィールドタイプと対象変数の数と怪我率のグラフを作成する関数。

def make_graph_fieldtype_number_and_injury_rate(graph_data1, graph_data2, var, title, yaxis):

    

    graph_data = pd.merge(graph_data1, graph_data2, on=var, how='outer')

    graph_data = graph_data.fillna(0)



    fig = make_subplots(specs=[[{"secondary_y": True}]])



    fig.add_trace(go.Scatter(x=graph_data[var], y=graph_data['inj_rate_x'], name='Injury rate (Natural)',

                             text=graph_data['inj_rate_x'], textposition='top right', marker_color='pink'), secondary_y=True)

    fig.add_trace(go.Scatter(x=graph_data[var], y=graph_data['inj_rate_y'], name='Injury rate (Synthetic)',

                             text=graph_data['inj_rate_y'], textposition='top right', marker_color='lightblue'), secondary_y=True)

    fig.update_traces(marker=dict(size=16))



    fig.add_trace(go.Bar(x=graph_data[var], y=graph_data['all_sum_x'], name=yaxis + ' (Natural)',

                         text=graph_data['all_sum_x'], textposition='auto', marker_color='red'), secondary_y=False)

    fig.add_trace(go.Bar(x=graph_data[var], y=graph_data['all_sum_y'], name=yaxis + ' (Synthetic)',

                         text=graph_data['all_sum_y'], textposition='auto', marker_color='blue'), secondary_y=False)

    

    if var == 'StadiumType' or var == 'Weather':

        graph_data['inj_rate'] = 104 / 5712 * 100

    elif var == 'PlayType':

        graph_data['inj_rate'] = 77 / 267006 * 100

    elif var == 'event':

        graph_data['inj_rate'] = 490 / 1839873 * 100

    

    fig.add_trace(go.Scatter(x=graph_data[var], y=graph_data['inj_rate'], mode='lines', marker_color='green', name='Average of injury rate'), secondary_y=True)



    fig.update_layout(

        title_text=title,

        xaxis=dict(showline=True, showgrid=False, showticklabels=True,),

        #legend_orientation="h",

        plot_bgcolor='white'

    )



    fig.update_yaxes(title_text=yaxis, secondary_y=False)

    fig.update_yaxes(title_text="Injury Rate", secondary_y=True)



    fig.show()



#怪我有無別に四分位点のグラフを作成する関数。

def make_graph_distribution_injury_and_not(graph_data, var, title):

    normal_df = graph_data[graph_data['inj_flg'] == 'Normal']

    injury_df = graph_data[graph_data['inj_flg'] == 'Injury']

    normal = graph_data[var].values

    injury = graph_data[var].values

    p_normal_injury = stats.ttest_ind(normal, injury, equal_var=False)

    pvalue = round(p_normal_injury[1],3)

    name = title+'(p-value='+str(pvalue)+')'

    

    fig = go.Figure()



    fig.add_trace(go.Violin(x=graph_data['inj_flg'], y=graph_data[var], box_visible=True, meanline_visible=True))



    fig.update_layout(

        title_text=name,

        xaxis=dict(showline=True, showgrid=False, showticklabels=True),

        yaxis_title=var,

        plot_bgcolor='white'

    )



    fig.show()



#フィールドタイプ別の怪我期間のファネルグラフを作成する関数。

def funnel_fieldtype_injury_duration(Natural_dm, Synthetic_dm, title):

    fig = go.Figure()



    fig.add_trace(go.Funnel(

        name = 'Natural',

        y = list(Natural_dm['variable'].values),

        x = list(Natural_dm['value'].values),

        textinfo = "value+percent initial"))



    fig.add_trace(go.Funnel(

        name = 'Synthetic',

        orientation = "h",

        y = list(Synthetic_dm['variable'].values),

        x = list(Synthetic_dm['value'].values),

        textposition = "inside",

        textinfo = "value+percent initial"))

    

    fig.update_layout(

        title_text=title,

        xaxis=dict(showline=True, showgrid=False, showticklabels=True),

        plot_bgcolor='white'

    )

    

    fig.show()
#プレイヤー情報集計用のデータフレームを作成する。

player_info = PlayList.drop_duplicates('PlayerKey')[['PlayerKey','RosterPosition', 'Position']].reset_index()

player_info.drop('index', axis=1, inplace=True)

injury_player_info = InjuryRecord[['PlayerKey']]

player_info = pd.merge(player_info, injury_player_info, on='PlayerKey', how='left', indicator=True)

player_info['inj_flg'] = player_info['_merge'].map({'left_only':'0', 'both':'1'})

player_info.drop('_merge', axis=1, inplace=True)
#ポジション別怪我率を算出する。

RosterPosition_inj_rate_df = make_inj_rate_df(df=player_info, var='RosterPosition')

#ポジション別に人数と怪我率をグラフ化する。

make_graph_number_and_injury_rate(graph_data=RosterPosition_inj_rate_df, 

                                  var='RosterPosition',

                                  title='The number of people and injury rate by position',

                                  yaxis='Number of people')
#試合情報集計用のデータフレームを作成する。

game_info = PlayList.drop_duplicates('GameID')[['GameID', 'StadiumType', 'FieldType', 'Temperature', 'Weather']].reset_index()

game_info.drop('index', axis=1, inplace=True)

injury_game_info = InjuryRecord.drop_duplicates('GameID')[['GameID']].reset_index()

injury_game_info.drop('index', axis=1, inplace=True)

game_info = pd.merge(game_info, injury_game_info, on='GameID', how='left', indicator=True)

game_info['inj_flg'] = game_info['_merge'].map({'left_only':'0', 'both':'1'})

game_info.drop('_merge', axis=1, inplace=True)

game_info = game_info.fillna('Unknown')
#試合情報集計用のデータフレームの変数を加工する。

def Weather_cond(x):

    cloudy = ['Cloudy 50% change of rain', 'Hazy', 'Cloudy.', 'Overcast', 'Mostly Cloudy',

              'Cloudy, fog started developing in 2nd quarter', 'Partly Cloudy',

              'Mostly cloudy', 'Rain Chance 40%',' Partly cloudy', 'Party Cloudy',

              'Rain likely, temps in low 40s', 'Partly Clouidy', 'Cloudy, 50% change of rain','Mostly Coudy',

              '10% Chance of Rain','Cloudy, chance of rain', '30% Chance of Rain', 'Cloudy, light snow accumulating 1-3"',

              'cloudy', 'Coudy', 'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.',

              'Cloudy fog started developing in 2nd quarter', 'Cloudy light snow accumulating 1-3"',

              'Cloudywith periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.',

              'Cloudy 50% change of rain', 'Cloudy and cold','Cloudy and Cool', 'Partly cloudy']

    

    clear = ['Clear, Windy',' Clear to Cloudy', 'Clear, highs to upper 80s',

             'Clear and clear','Partly sunny', 'Clear, Windy', 'Clear skies',

             'Sunny', 'Partly Sunny', 'Mostly Sunny', 'Clear Skies','Sunny Skies',

             'Partly clear', 'Fair', 'Sunny, highs to upper 80s', 'Sun & clouds', 'Mostly sunny','Sunny, Windy',

             'Mostly Sunny Skies', 'Clear and Sunny', 'Clear and sunny','Clear to Partly Cloudy', 'Clear Skies',

             'Clear and cold', 'Clear and warm', 'Clear and Cool', 'Sunny and cold', 'Sunny and warm', 'Sunny and clear']

    

    rainy = ['Rainy', 'Scattered Showers', 'Showers', 'Cloudy Rain', 'Light Rain',

             'Rain shower', 'Rain likely, temps in low 40s.', 'Cloudy, Rain']

    

    snow = ['Heavy lake effect snow']

    

    indoor = ['Indoor', 'Controlled Climate', 'Indoors', 'N/A Indoor', 'N/A (Indoors)']

    

    if x in cloudy:

        return 'Cloudy'

    elif x in indoor:

        return 'Unknown'

    elif x in clear:

        return 'Clear'

    elif x in rainy:

        return 'Rain'

    elif x in snow:

        return 'Snow'

    elif x in ['Heat Index 95', 'Cold']:

        return 'Unknown'

    else:

        return x



def StadiumType_cond(x):

    outdoor = ['Outdoor', 'Outdoors', 'Cloudy', 'Heinz Field', 'Outdor', 'Ourdoor',

               'Outside','Outddors', 'Outdoor Retr Roof-Open', 'Oudoor', 'Bowl']



    indoor_closed = ['Indoors', 'Indoor', 'Indoor, Roof Closed', 'Indoor, Roof Closed',

                     'Retractable Roof', 'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Retr. Roof Closed']



    indoor_open = ['Indoor, Open Roof', 'Open', 'Retr. Roof-Open', 'Retr. Roof - Open']



    dome_closed = ['Dome', 'Domed, closed', 'Closed Dome', 'Domed', 'Dome, closed']



    dome_open = ['Domed, Open', 'Domed, open']

    

    if x in outdoor:

        return 'outdoor'

    elif x in indoor_closed:

        return 'indoor_closed'

    elif x in indoor_open:

        return 'indoor_open'

    elif x in dome_closed:

        return 'dome_closed'

    elif x in dome_open:

        return 'dome_open'

    else:

        return x

    

game_info['Weather'] = game_info['Weather'].apply(Weather_cond)

game_info['StadiumType'] = game_info['StadiumType'].apply(StadiumType_cond)

game_info = game_info.fillna('Unknown')
#フィールドタイプ別怪我率を算出する。

FieldType_inj_rate_df = make_inj_rate_df(df=game_info, var='FieldType')

#フィールドタイプ別に人数と怪我率をグラフ化する。

make_graph_number_and_injury_rate(graph_data=FieldType_inj_rate_df,

                                  var='FieldType', 

                                  title='The number of game and injury rate by FieldType',

                                  yaxis='Number of game')
#スタジアムタイプ別怪我率を算出する。

StadiumType_inj_rate_df = make_inj_rate_df(df=game_info, var='StadiumType')

#スタジアムタイプ別に人数と怪我率をグラフ化する。

make_graph_number_and_injury_rate(graph_data=StadiumType_inj_rate_df,

                                  var='StadiumType', 

                                  title='The number of game and injury rate by StadiumType',

                                  yaxis='Number of game')
#天気別怪我率を算出する。

Weather_inj_rate_df = make_inj_rate_df(df=game_info, var='Weather')

#天気別に人数と怪我率をグラフ化する。

make_graph_number_and_injury_rate(graph_data=Weather_inj_rate_df, 

                                  var='Weather', 

                                  title='The number of game and injury rate by Weather',

                                  yaxis='Number of game')
#取得できた温度情報のみを抽出する。

Temperature_game_info = game_info[game_info['Temperature'] != -999]

Temperature_game_info = Temperature_game_info[['GameID', 'Temperature', 'inj_flg']]

Temperature_game_info['inj_flg'] = Temperature_game_info['inj_flg'].map({'0':'Normal', '1':'Injury'})

#怪我有無別に温度の分布をグラフ化する。

make_graph_distribution_injury_and_not(graph_data=Temperature_game_info,

                                       var='Temperature',

                                       title='Temperature Distribution')
#プレイ情報集計用のデータフレームを作成する。

play_info = PlayList[['PlayKey','PlayType', 'FieldType']].reset_index()

play_info.drop('index', axis=1, inplace=True)

injury_play_info = InjuryRecord[['PlayKey']]

play_info = pd.merge(play_info, injury_play_info, on='PlayKey', how='left', indicator=True)

play_info['inj_flg'] = play_info['_merge'].map({'left_only':'0', 'both':'1'})

play_info.drop('_merge', axis=1, inplace=True)
#プレイ別怪我率を算出する。

PlayType_inj_rate_df = make_inj_rate_df(df=play_info, var='PlayType')

#プレイ別に人数と怪我率をグラフ化する。

make_graph_number_and_injury_rate(graph_data=PlayType_inj_rate_df, 

                                  var='PlayType',

                                  title='The number of play and injury rate by PlayType',

                                  yaxis='Number of play')
#PlayerTrackData.csvをデータフレーム化する。

PlayerTrackData = pd.read_csv('../input/nfl-playing-surface-analytics/PlayerTrackData.csv')[['PlayKey', 'event']]

PlayerTrackData = reduce_mem_usage(PlayerTrackData)
#PlayerTrackDataに怪我フラグとフィールドタイプを追加する。

InjuryRecordNonan = InjuryRecord.dropna(subset=['PlayKey'])[['PlayKey', 'Surface']]

InjuryRecordNonan['inj_flg'] = '1'

play_injury = InjuryRecordNonan[['PlayKey', 'inj_flg']]

play_injury_dict = dict(list(play_injury.values))

PlayerTrackData['inj_flg'] = PlayerTrackData['PlayKey'].map(play_injury_dict)

PlayerTrackData = PlayerTrackData.fillna('0')

play_fieldtype = PlayList[['PlayKey', 'FieldType']]

play_fieldtype_dict = dict(list(play_fieldtype.values))

PlayerTrackData['FieldType'] = PlayerTrackData['PlayKey'].map(play_fieldtype_dict)

PlayerTrackData = PlayerTrackData.fillna('Unknown')

PlayerTrackData = PlayerTrackData[PlayerTrackData['event'] != '0']
#イベント別怪我率を算出する。

event_inj_rate_df = make_inj_rate_df(df=PlayerTrackData, var='event')

event_inj_rate_df = event_inj_rate_df[event_inj_rate_df['inj_sum'] > 0]

#イベント別に人数と怪我率をグラフ化する。

make_graph_number_and_injury_rate(graph_data=event_inj_rate_df, 

                                  var='event', 

                                  title='The number of action and injury rate by event',

                                  yaxis='Number of action')
#フィールドタイプ別に抽出したデータフレームを作成する。

Natural_game_info = game_info[game_info['FieldType'] == 'Natural']

Synthetic_game_info = game_info[game_info['FieldType'] == 'Synthetic']
#フィールドタイプとスタジアムタイプ別の怪我率を算出する。

Natural_StadiumType_inj_rate_df = make_inj_rate_df(df=Natural_game_info, var='StadiumType')

Synthetic_StadiumType_inj_rate_df = make_inj_rate_df(df=Synthetic_game_info, var='StadiumType')

#フィールドタイプとスタジアムタイプ別に人数と怪我率をグラフ化する。

make_graph_fieldtype_number_and_injury_rate(graph_data1=Natural_StadiumType_inj_rate_df,

                                            graph_data2=Synthetic_StadiumType_inj_rate_df,

                                            var='StadiumType',

                                            title='The number of game and injury rate by FieldType and StadiumType',

                                            yaxis='Number of game')
#フィールドタイプと天気別の怪我率を算出する。

Natural_Weather_inj_rate_df = make_inj_rate_df(df=Natural_game_info, var='Weather')

Synthetic_Weather_inj_rate_df = make_inj_rate_df(df=Synthetic_game_info, var='Weather')

#フィールドタイプと天気別に人数と怪我率をグラフ化する。

make_graph_fieldtype_number_and_injury_rate(graph_data1=Natural_Weather_inj_rate_df,

                                            graph_data2=Synthetic_Weather_inj_rate_df,

                                            var='Weather',

                                            title='The number of game and injury rate by FieldType and Weather',

                                            yaxis='Number of game')
#フィールドタイプ別に抽出したデータフレームを作成する。

Natural_play_info = play_info[play_info['FieldType'] == 'Natural']

Synthetic_play_info = play_info[play_info['FieldType'] == 'Synthetic']
#フィールドタイプとプレイ別の怪我率を算出する。

Natural_play_inj_rate_df = make_inj_rate_df(df=Natural_play_info, var='PlayType')

Synthetic_play_inj_rate_df = make_inj_rate_df(df=Synthetic_play_info, var='PlayType')

#フィールドタイプとプレイ別に人数と怪我率をグラフ化する。

make_graph_fieldtype_number_and_injury_rate(graph_data1=Natural_play_inj_rate_df,

                                            graph_data2=Synthetic_play_inj_rate_df,

                                            var='PlayType',

                                            title='The number of play and injury rate by FieldType and PlayType',

                                            yaxis='Number of play')
#メモリを削減する。

del player_info, RosterPosition_inj_rate_df, game_info, FieldType_inj_rate_df, StadiumType_inj_rate_df, Weather_inj_rate_df, Temperature_game_info, play_info, PlayType_inj_rate_df

gc.collect()
#フィールドタイプ別に抽出したデータフレームを作成する。

Natural_PlayerTrackData = PlayerTrackData[PlayerTrackData['FieldType'] == 'Natural']

Synthetic_PlayerTrackData = PlayerTrackData[PlayerTrackData['FieldType'] == 'Synthetic']
#フィールドタイプとイベント別の怪我率を算出する。

Natural_event_inj_rate_df = make_inj_rate_df(df=Natural_PlayerTrackData, var='event')

Natural_event_inj_rate_df = Natural_event_inj_rate_df[Natural_event_inj_rate_df['inj_rate'] != 0]

Synthetic_event_inj_rate_df = make_inj_rate_df(df=Synthetic_PlayerTrackData, var='event')

Synthetic_event_inj_rate_df = Synthetic_event_inj_rate_df[Synthetic_event_inj_rate_df['inj_rate'] != 0]

#フィールドタイプとイベント別に人数と怪我率をグラフ化する。

make_graph_fieldtype_number_and_injury_rate(graph_data1=Natural_event_inj_rate_df,

                                            graph_data2=Synthetic_event_inj_rate_df,

                                            var='event',

                                            title='The number of action and injury rate by FieldType and event',

                                            yaxis='Number of action')
#フィールドタイプ別怪我離脱期間ファネルグラフを作成する。

surface_dm = InjuryRecord.groupby('Surface').sum().reset_index()

surface_dm = surface_dm.rename(columns={'DM_M1':'1-6 days', 'DM_M7':'7-27 days', 'DM_M28':'28-41 days', 'DM_M42':'42- days'})

surface_dm.drop('PlayerKey', axis=1, inplace=True)

Natural_dm = surface_dm[surface_dm['Surface'] == 'Natural'].melt()

Natural_dm = Natural_dm[Natural_dm['value'] != 'Natural']

Synthetic_dm = surface_dm[surface_dm['Surface'] == 'Synthetic'].melt()

Synthetic_dm = Synthetic_dm[Synthetic_dm['value'] != 'Synthetic']

funnel_fieldtype_injury_duration(Natural_dm=Natural_dm, Synthetic_dm=Synthetic_dm, title='All BodyPart')
#フィールドタイプと怪我部位別怪我離脱期間ファネルグラフを作成する。

BodyPartlist = list(InjuryRecord['BodyPart'].drop_duplicates().values)

for var in BodyPartlist:

    InjuryRecord_tmp = InjuryRecord[InjuryRecord['BodyPart'] == var]

    surface_dm = InjuryRecord_tmp.groupby('Surface').sum().reset_index()

    surface_dm = surface_dm.rename(columns={'DM_M1':'1-6 days', 'DM_M7':'7-27 days', 'DM_M28':'28-41 days', 'DM_M42':'42- days'})

    surface_dm.drop('PlayerKey', axis=1, inplace=True)

    Natural_dm = surface_dm[surface_dm['Surface'] == 'Natural'].melt()

    Natural_dm = Natural_dm[Natural_dm['value'] != 'Natural']

    Synthetic_dm = surface_dm[surface_dm['Surface'] == 'Synthetic'].melt()

    Synthetic_dm = Synthetic_dm[Synthetic_dm['value'] != 'Synthetic']

    funnel_fieldtype_injury_duration(Natural_dm=Natural_dm, Synthetic_dm=Synthetic_dm, title=var)