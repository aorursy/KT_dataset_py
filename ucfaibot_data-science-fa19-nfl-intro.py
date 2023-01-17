import os

import pandas as pd

from kaggle.competitions import nflrush

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from keras import Sequential

from keras.layers import Dense,BatchNormalization,Dropout

from keras.optimizers import RMSprop,Adam

from keras.callbacks import ReduceLROnPlateau

import datetime

import tqdm
# First, let's build the enviroment from the API

env=nflrush.make_env()

df_train=pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv',low_memory=False)

df_train
print('The train dataframe contrains {} rows and {} columns'.format(df_train.shape[0],df_train.shape[1]))

df_train.isna().sum().sort_values(ascending=False)
df_train['WindSpeed'].value_counts()
def windspeed(x):

    x=str(x)

    if x.isdigit():

        return int(x)

    elif (x.isalpha()):

        return 0

    elif (x.isalnum()):

        return int(x.upper().split('M')[0])                             #return 12 incase of 12mp or 12 MPH

    elif '-' in x:

        return int((int(x.split('-')[0])+int(x.split('-')[1]))/2)   # return average windspeed incase of 11 - 20 etc..

    else:

        return 0
# Let's just apply our fix to the messed up values 

df_train['WindSpeed']=df_train['WindSpeed'].apply(windspeed)

# Then, lets just fill the missing values with the average, as we have been doing 

df_train['WindSpeed'].fillna(df_train['WindSpeed'].mean(),inplace=True)
sns.distplot(df_train['WindSpeed'])
df_train['WindDirection'].value_counts()
# So wind direction is a bit confusing in that it is all base on WHERE it comes from

# We can see that it has the same problem as wind speed: multiple wants for saying the same thing.

# So we need to handle these cases indv

# We are going to reduce the number of options a bit 



def clean_wind_direction(wind_direction):

    wd = str(wind_direction).upper()

    if wd == 'N' or 'FROM N' in wd:

        return 'north'

    if wd == 'S' or 'FROM S' in wd:

        return 'south'

    if wd == 'W' or 'FROM W' in wd:

        return 'west'

    if wd == 'E' or 'FROM E' in wd:

        return 'east'

    

    if 'FROM SW' in wd or 'FROM SSW' in wd or 'FROM WSW' in wd:

        return 'south west'

    if 'FROM SE' in wd or 'FROM SSE' in wd or 'FROM ESE' in wd:

        return 'south east'

    if 'FROM NW' in wd or 'FROM NNW' in wd or 'FROM WNW' in wd:

        return 'north west'

    if 'FROM NE' in wd or 'FROM NNE' in wd or 'FROM ENE' in wd:

        return 'north east'

    

    if 'NW' in wd or 'NORTHWEST' in wd:

        return 'north west'

    if 'NE' in wd or 'NORTH EAST' in wd:

        return 'north east'

    if 'SW' in wd or 'SOUTHWEST' in wd:

        return 'south west'

    if 'SE' in wd or 'SOUTHEAST' in wd:

        return 'south east'



    return 'none'



df_train['WindDirection'] = df_train['WindDirection'].apply(clean_wind_direction)
df_train['Humidity'].fillna(method='ffill', inplace=True)

df_train['Temperature'].fillna(method='ffill', inplace=True)

na_map = {

    # What is the average orientation of the playrees

    'Orientation': df_train['Orientation'].mean(),

    # Average direction 

    'Dir': df_train['Dir'].mean(),

    # Average # of defenders in the box (# of defenders directly opposing person with the ball)

    'DefendersInTheBox': np.math.ceil(df_train['DefendersInTheBox'].mean()),

    # What formation the team is using is really important, but often some teams use custom formations

    # In the case we will just say that we don't know

    'OffenseFormation': 'UNKNOWN'

}



df_train.fillna(na_map, inplace=True)

df_train['GameWeather'].value_counts()
def group_game_weather(weather):

    rain = [

        'Rainy', 'Rain Chance 40%', 'Showers',

        'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.',

        'Scattered Showers', 'Cloudy, Rain', 'Rain shower', 'Light Rain', 'Rain'

    ]

    overcast = [

        'Cloudy, light snow accumulating 1-3"', 'Party Cloudy', 'Cloudy, chance of rain',

        'Coudy', 'Cloudy, 50% change of rain', 'Rain likely, temps in low 40s.',

        'Cloudy and cold', 'Cloudy, fog started developing in 2nd quarter',

        'Partly Clouidy', '30% Chance of Rain', 'Mostly Coudy', 'Cloudy and Cool',

        'cloudy', 'Partly cloudy', 'Overcast', 'Hazy', 'Mostly cloudy', 'Mostly Cloudy',

        'Partly Cloudy', 'Cloudy'

    ]

    clear = [

        'Partly clear', 'Sunny and clear', 'Sun & clouds', 'Clear and Sunny',

        'Sunny and cold', 'Sunny Skies', 'Clear and Cool', 'Clear and sunny',

        'Sunny, highs to upper 80s', 'Mostly Sunny Skies', 'Cold',

        'Clear and warm', 'Sunny and warm', 'Clear and cold', 'Mostly sunny',

        'T: 51; H: 55; W: NW 10 mph', 'Clear Skies', 'Clear skies', 'Partly sunny',

        'Fair', 'Partly Sunny', 'Mostly Sunny', 'Clear', 'Sunny'

    ]

    snow  = ['Heavy lake effect snow', 'Snow']

    none  = ['N/A Indoor', 'Indoors', 'Indoor', 'N/A (Indoors)', 'Controlled Climate']

    

    if weather in rain:

        return 'rain'

    elif weather in overcast:

        return 'overcast'

    elif weather in clear:

        return 'clear'

    elif weather in snow:

        return 'snow'

    elif weather in none:

        return 'none'

    

    return 'none'



df_train['GameWeather'] = df_train['GameWeather'].apply(group_game_weather)



df_train['FieldPosition'] = np.where(df_train['YardLine'] == 50, df_train['PossessionTeam'], df_train['FieldPosition'])
df_train
def group_stadium_types(stadium):

    outdoor       = [

        'Outdoor', 'Outdoors', 'Cloudy', 'Heinz Field', 

        'Outdor', 'Ourdoor', 'Outside', 'Outddors', 

        'Outdoor Retr Roof-Open', 'Oudoor', 'Bowl'

    ]

    indoor_closed = [

        'Indoors', 'Indoor', 'Indoor, Roof Closed', 'Indoor, Roof Closed', 

        'Retractable Roof', 'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Retr. Roof Closed',

    ]

    indoor_open   = ['Indoor, Open Roof', 'Open', 'Retr. Roof-Open', 'Retr. Roof - Open']

    dome_closed   = ['Dome', 'Domed, closed', 'Closed Dome', 'Domed', 'Dome, closed']

    dome_open     = ['Domed, Open', 'Domed, open']

    

    if stadium in outdoor:

        return 'outdoor'

    elif stadium in indoor_closed:

        return 'indoor closed'

    elif stadium in indoor_open:

        return 'indoor open'

    elif stadium in dome_closed:

        return 'dome closed'

    elif stadium in dome_open:

        return 'dome open'

    else:

        return 'unknown'

    

df_train['StadiumType'] = df_train['StadiumType'].apply(group_stadium_types)
df_train['TimeHandoff'] = df_train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

df_train['TimeSnap'] = df_train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

df_train['TimeDelta'] = df_train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)

df_train.drop(['TimeSnap','TimeHandoff'],axis=1,inplace=True)
df_train['BirthYear']=df_train['PlayerBirthDate'].apply(lambda x : int(x.split('/')[2]))

df_train['GameHour']=df_train['GameClock'].apply(lambda x : int(x.split(':')[0]))



df_train.drop(['PlayerBirthDate',"GameClock"],axis=1,inplace=True)
df_train['PlayerHeight']=df_train['PlayerHeight'].apply(lambda x : np.mean(list(map(int,x.split('-')))))

#df_train.drop('PlayerHeight',axis=1,inplace=True)                                                       
def process_defense(x):

    num=[]

    num=x.split(',')

    dl=int(num[0].split(' ')[0])

    lb=int(num[1].split(' ')[1])

    db=int(num[2].split(' ')[1])

    if(len(num)>3):

         ol=int(num[3].split(' ')[1])

    else:

         ol=0

    return [dl,lb,db,ol]



values=df_train['DefensePersonnel'].apply(process_defense)

u,v,x,y=list(map(list,zip(*values)))
df_train['DL']=u

df_train['LB']=v

df_train['BL']=x

df_train['OL']=y

df_train.drop(['DefensePersonnel'],axis=1,inplace=True)
df_train.shape
new_obj=[]

for c in df_train.columns:

    if(df_train[c].dtype != int):

            try:

                df_train[c]=df_train[c].astype('float16')

            except:

                new_obj.append(c)
lbdic={}

for c in new_obj:

    lb=LabelEncoder()

    lb=lb.fit(df_train[c].values)

    lbdic[c]=lb

    df_train[c]=lb.transform(df_train[c].values)
columns_drop=['GameId','PlayId','NflId','NflIdRusher']

one=[]

two=[]

more=[]

for col in df_train.drop(columns_drop,axis=1).columns:

    if df_train[col][:22].nunique() <2:

        one.append(col)

    elif df_train[col][:22].nunique() <=2:

        two.append(col)

    else:

        more.append(col)

        
print('The number of attributes for preprocessing =',len(one)+len(two)+len(more))
# We're going to start by appending the variables that have more than two unique values

# Remember, every "example" is actually one of 11 timesteps from two seperate games, 

# For a total of 22 datapoints per player

new_cols=[]

for col in more:

    for i in range(0,11):

        new_cols.append(str(col)+'A'+str(i))

    for i in range(0,11):

         new_cols.append(str(col)+'B'+str(i))

        

        
train=pd.DataFrame()

x=np.tile(np.arange(0,22),14)
# Now we build the targets

out=[]

for c in more:

    for  i in range(0,22):

         out.append(df_train[i:len(df_train):22][c].values)

               

for col in zip(new_cols,np.arange(len(out))):

    train[col]=out[i]

out=np.array(out).transpose()
train=pd.DataFrame(data=out,columns=new_cols)

    
train.head()
df_one=df_train.groupby(['PlayId'])[one].first()

for col in df_one.columns:

    train[col]=df_one[col].values
sns.distplot(train['Yards'].values)
not_object=[]

obj=[]

for col in more+one:

    if df_train[col].dtype != 'object':

        not_object.append(col)

    else:

        obj.append(col)
# Split the features and target variables

X=train.drop('Yards',axis=1)

y=train['Yards']

def create_model():

    model=Sequential()

    model.add(Dense(356,input_shape=[X.shape[1]],activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(.4))

    model.add(Dense(200,activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(.4))

    model.add(Dense(256,activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(.3))

    model.add(Dense(212,activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(.3))

    model.add(Dense(199,activation='sigmoid'))



    optimizer=Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)

    model.compile(optimizer=optimizer,loss=['mse'],metrics=['accuracy'])

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 

                                                patience=3, 

                                                verbose=1, 

                                                factor=0.5, 

                                                min_lr=0.00001)

    return model

model = create_model()
def transform_y(X_train,y_train):

    Y_train=np.zeros((X_train.shape[0],199))

    for i,yard in enumerate(y_train):

        Y_train[i, yard+99:] = np.ones(shape=(1, 100-yard))

    

    return Y_train



from sklearn.model_selection import KFold

kfold=KFold(n_splits=3,shuffle=True)



for train_ind,val in kfold.split(X,y):

    

    x_train,xval = X.iloc[train_ind],X.iloc[val]

    y_train,yval= y.iloc[train_ind],y.iloc[val]

    

    y_train=transform_y(x_train,y_train)

    y_val=transform_y(xval,yval)

    

    model=None

    model=create_model()

    

    history=model.fit(x_train,y_train,epochs=20,validation_data=[xval,y_val],verbose=1)

    print('validation accuracy : {}'.format(np.mean(history.history['val_accuracy'])))
fig,ax=plt.subplots(2,1)

fig.set_size_inches((5,5))

epochs=20

x=range(1,1+epochs)

ax[0].plot(x,history.history['loss'],color='red')

ax[0].plot(x,history.history['val_loss'],color='blue')



ax[1].plot(x,history.history['accuracy'],color='red')

ax[1].plot(x,history.history['val_accuracy'],color='blue')

ax[0].legend(['trainng loss','validation loss'])

ax[1].legend(['trainng acc','validation acc'])

plt.xlabel('Number of epochs')

plt.ylabel('accuracy')
def make_prediction(test,sample,env,model,df_train):

    

    na_map = {

    'Orientation': df_train['Orientation'].mean(),

    'Dir': df_train['Dir'].mean(),

    'DefendersInTheBox': 7.0,

    'OffenseFormation': 'UNKNOWN','WindSpeed':df_train['WindSpeed'].mean()

    }



    test.fillna(na_map, inplace=True)

    test['Temperature'].fillna(61.0,inplace=True)

    test['WindSpeed']=test['WindSpeed'].apply(windspeed)

    #test['WindSpeed'].fillna(df_train['WindSpeed'].mean(),inplace=True)



    test['GameWeather'] = test['GameWeather'].apply(group_game_weather)

    test['FieldPosition'] = np.where(test['YardLine'] == 50, test['PossessionTeam'], test['FieldPosition'])

    test['StadiumType'] = test['StadiumType'].apply(group_stadium_types)

    test['WindDirection'] = test['WindDirection'].apply(clean_wind_direction)

    

    test['TimeHandoff'] = test['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    test['TimeSnap'] = test['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    test['TimeDelta'] = test.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)

    test.drop(['TimeSnap','TimeHandoff'],axis=1,inplace=True)

   

    test['PlayerHeight']=test['PlayerHeight'].apply(lambda x : np.mean(list(map(int,x.split('-')))))









    test['BirthYear']=test['PlayerBirthDate'].apply(lambda x : int(x.split('/')[2]))

    test['GameHour']=test['GameClock'].apply(lambda x : int(x.split(':')[0]))

    test.drop(['PlayerBirthDate',"GameClock"],axis=1,inplace=True)



    values=test['DefensePersonnel'].apply(process_defense)

    u,v,x,y=list(map(list,zip(*values)))

    test['DL']=u

    test['LB']=v

    test['BL']=x

    test['OL']=y

    test.drop(['DefensePersonnel'],axis=1,inplace=True)

    

    new_obj=[]

    for c in test.columns:

        if(test[c].dtype != int):

                try:

                    test[c]=test[c].astype('float16')

                except:

                    new_obj.append(c)



    for c in new_obj:

        try:

            test[c]=lbdic[c].transform(test[c].values)

        except:

            l=LabelEncoder()

            test[c]=l.fit_transform(test[c].values)

            

    

    columns_drop=['GameId','PlayId','NflId','NflIdRusher']

    one=[]

    two=[]

    more=[]

    for col in test.drop(columns_drop,axis=1).columns:

        if test[col][:22].nunique() <2:

            one.append(col)

        elif test[col][:22].nunique() <=2:

            two.append(col)

        else:

            more.append(col)

        



    new_cols=[]

    for col in more:

        for i in range(0,11):

            new_cols.append(str(col)+'A'+str(i))

        for i in range(0,11):

             new_cols.append(str(col)+'B'+str(i))



    



    out=[]

    for c in more:

        out.append(test[c].values)

    

   

    new_out=[]

    for i in out:

        for j in i:

            new_out.append(j)



    new_test=pd.DataFrame(data=[new_out],columns=new_cols)



    df_one=test.groupby(['PlayId'])[one].first()

    for col in df_one.columns:

        new_test[col]=df_one[col].values



    

    new_test.fillna(na_map,inplace=True)

    new_test['Temperature'].fillna(61.0,inplace=True)

        

    y_pred=np.zeros((1,199))

    

    y_pred = model.predict(new_test)

    

        

    for pred in y_pred:

        prev = 0

        for i in range(len(pred)):

            if pred[i]<prev:

                pred[i]=prev

            prev=pred[i]

    

    y_pred[:, -1] = np.ones(shape=(y_pred.shape[0], 1))

    y_pred[:, 0] = np.zeros(shape=(y_pred.shape[0], 1))

  

    pred=pd.DataFrame(data=y_pred,columns=sample.columns)

    env.predict(pred)



    return y_pred



        

for test, sample in tqdm.tqdm(env.iter_test()):

    make_prediction(test,sample,env,model,df_train)

    

env.write_submission_file()