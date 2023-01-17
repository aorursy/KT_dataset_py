import numpy as np # Linear algebra operations
import pandas as pd # Data processing
import random # Random sampling of data
import datetime as dt # Formatting dates and times
def loadAndRemove(s):
    '''
    In this project, we are interested in plays (and therefore injuries) that occur downfield, the times of which lie between 
    the time of the punt and the time of the play completion.
    
    This function reads in a dataset and removes observations in which the time interval between the snap and the punt is
    small. Thus, we keep observations about plays that occur downfield, and discard observations about plays that occur at
    the line of scrimmage or after completion of the play.
    
    Arguments:
        s: A file name, entered as a string.
    
    Returns:
        df: A dataframe which is stripped of any play occurring prior to the punt or after play completion.
    '''    
    # Remove NAs
    df = pd.read_csv(s, low_memory=False).dropna(subset=['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'Time', 'x', 'y', 'dis', 'o', 'dir'])
    
    # Convert Time variable (originally a string) to a datetime object
    df.Time = pd.to_datetime(df.Time)
    # Create GamePlayKey to denote the game ID and play ID of each observation
    df['GamePlayKey'] = df.GameKey.astype(str) + '_' + df.PlayID.astype(str)
    # Create seconds variable, which holds the number of elapsed seconds since January 1, 2015 at 12am
    df['seconds'] = (df['Time'] - dt.datetime(2015,1,1)).dt.total_seconds()
    
    # Create punt_seconds column that holds the time at which the ball was punted
    # We group by GamePlayKey & then merge punt_seconds based on when Event "punt" occurs within the play
    df = pd.merge(df, df.groupby('GamePlayKey').apply(lambda x: x.loc[x.Event == 'punt'].seconds.mean()).dropna().to_frame(name = 'punt_seconds'),  on = ['GamePlayKey'], how ='outer')
    
    # Create play_end_seconds column that holds the num of seconds at which the play is considered completed
    # We group by GamePlayKey and then merge play_end_seconds based on the first time point at which an event occurs that signals the end of the play
    # We considered "play ending events" to be the following:
        # fair_catch, tackle, safety, touchback, out_of_bounds, punt_downed, touchdown, fumble_defense_recovered, pass_outcome_incomplete
    df = pd.merge(df, df.groupby('GamePlayKey').apply(lambda x: x.loc[(x.Event == 'fair_catch') | (x.Event == 'tackle') | (x.Event == 'safety') | (x.Event == 'touchback') | (x.Event == 'out_of_bounds') | (x.Event == 'punt_downed') | (x.Event == 'touchdown') | (x.Event == 'fumble_defense_recovered') | (x.Event == 'pass_outcome_incomplete')].seconds.min()).dropna().to_frame(name = 'play_end_seconds'),  on = ['GamePlayKey'], how ='outer')
    
    # Keep only observations that occur after the ball has been punted
    df = df.loc[df.seconds > df.punt_seconds]
    # Keep only observations that occur before the completion of play
    df = df.loc[df.seconds < df.play_end_seconds]
    
    # Create post_punt_playduration, which denotes the elapsed time (in seconds) between the punt and the completion of play
    df['post_punt_playduration'] = df.play_end_seconds - df.punt_seconds
    # Create post_punt_duration, which denotes the elapsed time (in seconds) between the punt and the occurrence of the observation
    df['post_punt_duration'] = df.seconds - df.punt_seconds

    return df
# Import Next Gen Stats data in a way that saves RAM; load all data into dataframe ngs
ngs = loadAndRemove('../input/NGS-2016-reg-wk13-17.csv')
ngs = ngs.append(loadAndRemove('../input/NGS-2017-reg-wk1-6.csv'))
ngs = ngs.append(loadAndRemove('../input/NGS-2017-reg-wk7-12.csv'))
ngs = ngs.append(loadAndRemove('../input/NGS-2017-pre.csv'))
ngs = ngs.append(loadAndRemove('../input/NGS-2017-reg-wk13-17.csv'))
ngs = ngs.append(loadAndRemove('../input/NGS-2016-pre.csv'))
ngs = ngs.append(loadAndRemove('../input/NGS-2016-reg-wk7-12.csv'))
ngs = ngs.append(loadAndRemove('../input/NGS-2017-post.csv'))
ngs = ngs.append(loadAndRemove('../input/NGS-2016-reg-wk1-6.csv'))
ngs = ngs.drop(['seconds','punt_seconds','play_end_seconds'], axis = 1)

# Import other data sources
df_gamedata = pd.read_csv('../input/game_data.csv')
df_videoreview = pd.read_csv('../input/video_review.csv')
# Create GamePlayKey based on the unique Game ID and the unique Player ID
df_videoreview = df_videoreview.assign(GamePlayKey = df_videoreview.GameKey.astype(str) + '_' + df_videoreview.PlayID.astype(str))
# Create GamePlayTimeKey, which is a unique ID for every time point for which there exists recorded data
ngs['GamePlayTimeKey'] = ngs['GamePlayKey'].astype(str) + '_' + ngs['Time'].astype(str)

#Removing rows with missign values from videoreview
df_videoreview = df_videoreview.dropna()
df_videoreview = df_videoreview.reset_index()
df_videoreview = df_videoreview.drop(['index'], axis = 1)

#adding features for referencing
df_videoreview['PlayPlayerPartnerID_1'] = df_videoreview['PlayID'].astype(str) + '_' + df_videoreview['GSISID'].astype(str) + '_' + df_videoreview['Primary_Partner_GSISID'].astype(str)

df_videoreview['PlayPlayerPartnerID_2'] = df_videoreview['PlayID'].astype(str) + '_' + df_videoreview['Primary_Partner_GSISID'].astype(str) + '_' + df_videoreview['GSISID'].astype(str)
def createData(df_create, contactDist):
    '''
    This function iterates through player data to create a new dataframe. Each row in the new dataframe consists of player data
    and contact partner data, only if the two players are deemed to have made contact with each other during the play.
    
    Args:
        df_create: A pandas dataframe that has the same features as the ngs dataframe.
        contactDist: A value (float or integer) that represents the maximum distance two players can be located from each other
        in order for there to be considered contact between them.
    
    Returns:
        df_output: A pandas dataframe that has double the number of rows as df_create. Features in df_output consist of the same
        features as df_create, as well as new feature variable names for the contact partner that are analogous to the features
        for the primary player, represented with a "_partner" suffix on the column name. Every row in df_output represents an
        instance of contact between the player (represented by the first half of the columns) and his primary partner (represented
        by the second half of the columns).
    '''
    # Instantiate the output dataframe, to which we will continually append rows
    df_output = pd.DataFrame()

    # For every unique combination of Game, Play, and Time in df_create
    for key in (df_create.GamePlayTimeKey.unique()):
        # Filter df_create to include only the current key of interest, saving in a new dataframe called df_temp
        df_temp = df_create.loc[df_create['GamePlayTimeKey'] == key]
        # For every row in df_temp
        for i in range(len(df_temp)):
            # For every subsequent row in df_temp
            for j in range(i+1, len(df_temp)):
                # Compute the distance between the locations of player and contact partner
                # Then determine if this distance is less than contactDist
                if (((df_temp.iloc[i]['x'] - df_temp.iloc[j]['x']) ** 2 +
                    (df_temp.iloc[i]['y'] - df_temp.iloc[j]['y']) ** 2) ** 0.5 <= contactDist):
                    # Merge the df_temp data of the player and the contact partner if they contacted each other into a single row
                    # Add this new row into a new dataframe called df_temp2
                    df_temp2 = pd.merge(df_temp.iloc[i].to_frame().transpose(),
                                       df_temp.iloc[j].to_frame().transpose(),
                                       suffixes = ['','_partner'], on='GamePlayTimeKey') # Add a suffix to indicate contact partner feature
                    
                    # Drop unnecessary features
                    df_temp2 = df_temp2.drop(['Season_Year_partner', 'GameKey_partner', 'PlayID_partner', 'Time_partner', 'Event_partner', 'GamePlayKey_partner', 'post_punt_playduration_partner', 'post_punt_duration_partner'], axis = 1)
                    # Append all rows from df_temp2 to df_output. df_temp2 will then be automatically overwritten during the next iteration
                    df_output = df_output.append(pd.DataFrame(data=df_temp2), ignore_index = True)
    
    # Ensure there is some data stored in df_output
    if len(df_output) > 0:
        # Convert GameKey identifier to an integer
        df_output.GameKey = df_output.GameKey.astype(int)
        # Merge df_output with dataframe df_gamedata based on the GameKey ID
        df_output = df_output.merge(df_gamedata, on = ['GameKey'], suffixes = ['','_gameData']) 
    
        # If there are any duplicate rows in df_output, this means that players contacted each other for a considerable amount of time
        # Drop such observations to avoid redundancy
        df_output = df_output.drop_duplicates(subset = ['GSISID', 'GSISID_partner', 'GamePlayKey'])
        
        # Drop redundant columns, since such columns were created for both the primary player and the contact partner
        df_output = df_output.drop(['Season_Year_gameData', 'Home_Team', 'Visit_Team', "HomeTeamCode","VisitTeamCode", "Stadium","Game_Site","OutdoorWeather", "StadiumType", "GameWeather", "Turf"], axis = 1)
    
    return df_output
# We first take a random sample of data from ngs dataframe
# Set a random seed to ensure reproducibility of results
random.seed(53)

# Define contactDist, which determines how far apart two players can be for contact to be considered
contactDist = 0.5
# Define the number of GamePlayTimeKeys that are to be taken as a sample from ngs dataframe
length = 1500
# Obtain the sample which consists of 1500 (length) unique keys
samp = random.sample(list(ngs.GamePlayTimeKey.unique()), length)
# Filter dataframe ngs based on the samp criteria as defined above
ngs_sample = ngs.loc[ngs.GamePlayTimeKey.isin(samp)]

# Create the data based on the ngs_sample dataframe and the defined contactDist
df = createData(ngs_sample, contactDist)

# Create PlayPlayerPartnerID variable that serves as a key to indicate the two players that make contact with each other
df['PlayPlayerPartnerID'] = df['PlayID'].astype(str) + '_' + df['GSISID'].astype(int).astype(str) + '_' + df['GSISID_partner'].astype(int).astype(str)

# Now we take a sample from plays taking sample from plays in video review
# This dataset consists of identifiable plays that are associated with concussions

# Filter dataframe ngs based on all GamePlayKeys that are in dataframe df_videoreview
ngs_sample = ngs.loc[ngs.GamePlayKey.isin(df_videoreview.GamePlayKey)]
# Create the data based on the new ngs_sample dataframe
df2 = createData(ngs_sample, contactDist)

# Create PlayPlayerPartnerID variable that serves as a key to indicate the two players that make contact with each other
df2['PlayPlayerPartnerID'] = df2['PlayID'].astype(str) + '_' + df2['GSISID'].astype(int).astype(str) + '_' + df2['GSISID_partner'].astype(int).astype(str)

# Remove plays from the created dataframe that were video-reviewed, but were not associated with concussions
# This is to ensure that we do not oversample from the data
df2 = df2.loc[df2.PlayPlayerPartnerID.isin(df_videoreview.PlayPlayerPartnerID_1) | df2.PlayPlayerPartnerID.isin(df_videoreview.PlayPlayerPartnerID_2)]

# Append this newly created dataframe to the dataframe produced from the first run of createData
df = df.append(df2)

# Export data to a CSV file which will be used as input for the Classification notebook
df.to_csv('x.csv', index = False)
# Create y data (i.e. dependent variable data) and export it as a CSV file
y = df.PlayPlayerPartnerID.isin(df_videoreview.PlayPlayerPartnerID_1) | df.PlayPlayerPartnerID.isin(df_videoreview.PlayPlayerPartnerID_2)
y.to_csv('y.csv', index = False)