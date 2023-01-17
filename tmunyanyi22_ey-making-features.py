#For data manipulaiton

import pandas as pd



#For linear algebra

import numpy as np



#For plotting

import matplotlib.pyplot as plt



import seaborn as sns

%matplotlib inline



#Not sure what this is used for

from scipy import stats



#Used for specific mathematical functions

import math



#to calculate skewness and kurtosis

from scipy.stats import skew

from scipy.stats import kurtosis

from scipy.stats import moment

from scipy.stats import iqr



# The distance 

from scipy.spatial.distance import euclidean



#Used for tracking progress of for loops

from tqdm import tqdm



#for suppressing warnings

import warnings

warnings.filterwarnings('ignore')



#The paths of the data

#Test_Path = '/content/gdrive/My Drive/Personal/EY Data Science Challange/data/test.csv'

#Train_Path = '/content/gdrive/My Drive/Personal/EY Data Science Challange/data/train.csv'



#Alternative paths

#Train_Path = '/content/gdrive/My Drive/EY Data Science Challange/data/train.csv'

#Test_Path = '/content/gdrive/My Drive/EY Data Science Challange/data/test.csv'



#Kaggle paths

Train_Path = '../input/data_train.csv'

Test_Path = '../input/data_test.csv'
#Training and testing data

df_train = pd.read_csv(Train_Path)

df_test = pd.read_csv(Test_Path)



#Drop the unnecessary index columnd

df_train = df_train.drop("Unnamed: 0", axis= 1)

df_test  = df_test.drop("Unnamed: 0", axis= 1)
#Specifying the city edge limits so that we don't have to type them later on.

x_min =3750901.5068

x_max = 3770901.5068

y_min = -19268905.6133

y_max = -19208905.6133

width = x_max - x_min

height = y_max - y_min
df_train['Is_V_NOT_NULL'] = np.where(df_train['vmean']>=0,1,0)

df_test['Is_V_NOT_NULL'] = np.where(df_test['vmean']>=0,1,0)



df_train['Is_V_minus'] = np.where(df_train['vmean']<0,1,0)

df_test['Is_V_minus'] = np.where(df_test['vmean']<0,1,0)
#Turns the time_entry and time_exit columns to seconds

def HHMMSS_int(t):

  (h, m, s) = t.split(':')

  return int(h) * 3600 + int(m) * 60 + int(s)



#Calculates the distance between points

def Distance(a,b,c,d):

  hor = a-b

  ver = c-d

  return (hor**2+ver**2)**0.5



#Calculates the direction an individual is travelling

def Direction(a,b,c,d):

  pointA = (a,b)

  pointB = (c,d)

  

  lat1 = math.radians(pointA[0])

  lat2 = math.radians(pointB[0])



  diffLong = math.radians(pointB[1] - pointA[1])



  x = math.sin(diffLong) * math.cos(lat2)

  y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)

  * math.cos(lat2) * math.cos(diffLong))



  initial_bearing = math.atan2(x, y)



  # Now we have the initial bearing but math.atan2 return values

  # from -180° to + 180° which is not what we want for a compass bearing

  # The solution is to normalize the initial bearing as shown below

  initial_bearing = math.degrees(initial_bearing)

  compass_bearing = (initial_bearing + 360) % 360



  return compass_bearing



#Checks if an individual is the city

def InCity(a,b):

  a = list(a)

  b = list(b)

  x = []

  for i in range(len(a)):

    cond = ((3750901.5068 <= a[i] <= 3770901.5068) and (-19268905.6133<= b[i]<=-19208905.6133)) #Found the issue in this function. Had the y limits mixed up initially.

    if cond:

      x.append(1)

    else:

      x.append(0)

  return x



print('done')
#Turn time_entry and time_exit to secodns and also making a new variable calculating the between between entry and exit for a trajectory

df_train['time_entry_secs'] = df_train['time_entry'].map(lambda x : HHMMSS_int(x))

df_train['time_exit_secs'] = df_train['time_exit'].map(lambda x : HHMMSS_int(x))

df_train['Time_stayed_secs'] = df_train['time_exit_secs']-df_train['time_entry_secs']



#Make a variable which captures if the individual was in the city at the time of exit

df_train['In_City'] = InCity(df_train['x_exit'],df_train['y_exit'])





#The code below will be used to calculate the Bearings(direction the individual is travelling in) and Distance travelled variables

a = list(df_train['x_entry'])

b = list(df_train['y_entry'])

c = list(df_train['x_exit'])

d = list(df_train['y_exit']) 

Bearings = []

DistanceL = []







for i in tqdm(range(len(a))):

  Bearings.append(Direction(a[i],b[i],c[i],d[i]))

  DistanceL.append(Distance(a[i],c[i],b[i],d[i]))      #Taku: Made a change to the input order. Initially it was a,b,c,d. Changed it to a,c,d,b so that it matches 



Bearings = pd.DataFrame(Bearings, columns = ['Bearings'])

DistanceL = pd.DataFrame(DistanceL, columns = ['Euclidean_Distance'])



df_train = pd.concat([df_train, Bearings, DistanceL], axis=1)
#Turn time_entry and time_exit to secodns and also making a new variable calculating the between between entry and exit for a trajectory

df_test['time_entry_secs'] = df_test['time_entry'].map(lambda x : HHMMSS_int(x))

df_test['time_exit_secs'] = df_test['time_exit'].map(lambda x : HHMMSS_int(x))

df_test['Time_stayed_secs'] = df_test['time_exit_secs']-df_test['time_entry_secs']



#Make a variable which captures if the individual was in the city at the time of exit

df_test['In_City'] = InCity(df_test['x_exit'],df_test['y_exit'])





#The code below will be used to calculate the Bearings(direction the individual is travelling in) and Distance travelled variables

a = list(df_test['x_entry'])

b = list(df_test['y_entry'])

c = list(df_test['x_exit'])

d = list(df_test['y_exit']) 

Bearings = []

DistanceL = []







for i in tqdm(range(len(a))):

  Bearings.append(Direction(a[i],b[i],c[i],d[i]))

  DistanceL.append(Distance(a[i],c[i],b[i],d[i]))      #Taku: Made a change to the input order. Initially it was a,b,c,d. Changed it to a,c,d,b so that it matches 



Bearings = pd.DataFrame(Bearings, columns = ['Bearings'])

DistanceL = pd.DataFrame(DistanceL, columns = ['Euclidean_Distance'])



df_test = pd.concat([df_test, Bearings, DistanceL], axis=1)
#Make a new variable capturing speed

df_train['Euclidean_Speed'] = np.array(df_train['Euclidean_Distance'])/np.array(df_train['Time_stayed_secs'])

df_train['Euclidean_Speed'] = df_train['Euclidean_Speed'].replace(np.inf,0)

df_test['Euclidean_Speed'] = np.array(df_test['Euclidean_Distance'])/np.array(df_test['Time_stayed_secs'])

df_test['Euclidean_Speed'] = df_test['Euclidean_Speed'].replace(np.inf,0)
#Making a variable which contains the hour of day when an individual's time of entry was caputred

df_train['hour_of_day'] = df_train['time_entry'].map(lambda x : int(x.split(':')[0]))

df_test['hour_of_day'] = df_test['time_entry'].map(lambda x : int(x.split(':')[0]))
#Make the last number in the hash number into a factor

df_train['New_Hash_Var_1'] = df_train.hash.map(lambda x:np.float(x[-1]))

df_test['New_Hash_Var_1'] = df_test.hash.map(lambda x:np.float(x[-1]))
###Sperical_Distance for training data



#Make list of potential desired columns

distances = ['x_entry', 'y_entry','x_exit', 'y_exit']

entry = ['x_entry', 'y_entry']

exit = ['x_exit', 'y_exit']



radius = 3963 #In miles



#Store the entry and exisst coordinates in seperata vectors

entry_vector = pd.Series(list(df_train[entry].values)).map(lambda x:np.array(x))

exit_vector = pd.Series(list(df_train[exit].values)).map(lambda x:np.array(x))



distances_vector = list()

for i in tqdm(range(len(entry_vector))):



  #Entry and exit points

  entry = tuple(entry_vector[i])

  exit =  tuple(exit_vector[i])

  

  #Abs change in k

  abs_change_in_k = np.abs(entry[1] - exit[1])

  

  

  #change in sigma

  triangle_sigma = np.arccos(np.sin(entry[0])*np.sin(exit[0]) + np.cos(entry[0])*np.cos(exit[0])*np.cos(abs_change_in_k))

  

  #Calculate spherical distance

  distance = radius*triangle_sigma

  distances_vector.append(distance)

  

#Add in a new column with this type of speed

df_train['Spherical_Distance'] = distances_vector
###Sperical_Distance for testing data



#Make list of potential desired columns

distances = ['x_entry', 'y_entry','x_exit', 'y_exit']

entry = ['x_entry', 'y_entry']

exit = ['x_exit', 'y_exit']



radius = 3963 #In miles



#Store the entry and exisst coordinates in seperata vectors

entry_vector = pd.Series(list(df_test[entry].values)).map(lambda x:np.array(x))

exit_vector = pd.Series(list(df_test[exit].values)).map(lambda x:np.array(x))



#Make a list to store the distances

distances_vector = list()



for i in tqdm(range(len(entry_vector))):



  #Entry and exit points

  entry = tuple(entry_vector[i])

  exit =  tuple(exit_vector[i])

  

  #Abs change in k

  abs_change_in_k = np.abs(entry[1] - exit[1])

  

  

  #change in sigma

  triangle_sigma = np.arccos(np.sin(entry[0])*np.sin(exit[0]) + np.cos(entry[0])*np.cos(exit[0])*np.cos(abs_change_in_k))

  

  #Calculate spherical distance

  distance = radius*triangle_sigma

  distances_vector.append(distance)

  

#Add in a new column with this type of speed

df_test['Spherical_Distance'] = distances_vector
#Make a new variable capturing speed

df_train['Spherical_Speed'] = np.array(df_train['Spherical_Distance'])/np.array(df_train['Time_stayed_secs'])

df_train['Spherical_Speed'] = df_train['Spherical_Speed'].replace(np.inf,0)

df_test['Spherical_Speed'] = np.array(df_test['Spherical_Distance'])/np.array(df_test['Time_stayed_secs'])

df_test['Spherical_Speed'] = df_test['Spherical_Speed'].replace(np.inf,0)
#The city centre

x_centre = (x_max + x_min)/2

y_centre = (y_max + y_min)/2

city_centre = (x_centre,y_centre)
#Spherical Distance From City Centre For TRAIN

entry = ['x_entry', 'y_entry']



radius = 3963 #In miles



#Store the entry and exisst coordinates in seperata vectors

entry_vector = pd.Series(list(df_train[entry].values)).map(lambda x:np.array(x))



#Make a list to store the distances

distances_vector = list()



for i in tqdm(range(len(entry_vector))):



  #Entry and exit points

  entry = tuple(entry_vector[i])

  exit =  city_centre

  

  #Abs change in k

  abs_change_in_k = np.abs(entry[1] - exit[1])

  

  

  #change in sigma

  triangle_sigma = np.arccos(np.sin(entry[0])*np.sin(exit[0]) + np.cos(entry[0])*np.cos(exit[0])*np.cos(abs_change_in_k))

  

  #Calculate spherical distance

  distance = radius*triangle_sigma

  distances_vector.append(distance)

  

#Add in a new column with this type of speed

df_train['SDistance_From_Centre'] = distances_vector

#Spherical Distance From City Centre for TEST

entry = ['x_entry', 'y_entry']



radius = 3963 #In miles



#Store the entry and exisst coordinates in seperata vectors

entry_vector = pd.Series(list(df_test[entry].values)).map(lambda x:np.array(x))



#Make a list to store the distances

distances_vector = list()



for i in tqdm(range(len(entry_vector))):



  #Entry and exit points

  entry = tuple(entry_vector[i])

  exit =  city_centre

  

  #Abs change in k

  abs_change_in_k = np.abs(entry[1] - exit[1])

  

  

  #change in sigma

  triangle_sigma = np.arccos(np.sin(entry[0])*np.sin(exit[0]) + np.cos(entry[0])*np.cos(exit[0])*np.cos(abs_change_in_k))

  

  #Calculate spherical distance

  distance = radius*triangle_sigma

  distances_vector.append(distance)

  

#Add in a new column with this type of speed

df_test['SDistance_From_Centre'] = distances_vector
#Euclidean Distance From City Centre for TRAIN

entry = ['x_entry', 'y_entry']



#Store the entry and exisst coordinates in seperata vectors

entry_vector = pd.Series(list(df_train[entry].values)).map(lambda x:np.array(x))



#Make a list to store the distances

distances_vector = list()



for i in tqdm(range(len(entry_vector))):



  #Entry and exit points

  entry = tuple(entry_vector[i])

  exit =  city_centre

  

  distances_vector.append(euclidean(np.array(entry),np.array(exit)))

  

#Add in a new column with this type of speed

df_train['EDistance_From_Centre'] = distances_vector
#Euclidean Distance From City Centre for TEST

entry = ['x_entry', 'y_entry']



#Store the entry and exisst coordinates in seperata vectors

entry_vector = pd.Series(list(df_test[entry].values)).map(lambda x:np.array(x))



#Make a list to store the distances

distances_vector = list()



for i in tqdm(range(len(entry_vector))):



  #Entry and exit points

  entry = tuple(entry_vector[i])

  exit =  city_centre

  

  distances_vector.append(euclidean(np.array(entry),np.array(exit)))

  

#Add in a new column with this type of speed

df_test['EDistance_From_Centre'] = distances_vector
df_train['In_City_At_Entry'] = InCity(df_train['x_entry'],df_train['y_entry'])

df_test['In_City_At_Entry'] = InCity(df_test['x_entry'],df_test['y_entry'])
#Different parts of the city

north_east_train = (df_train['x_entry'] > x_centre) & (df_train['y_entry'] > y_centre) & (df_train['In_City_At_Entry'] ==0)

north_west_train = (df_train['x_entry'] < x_centre) & (df_train['y_entry'] > y_centre) & (df_train['In_City_At_Entry'] ==0)

south_east_train = (df_train['x_entry'] > x_centre) & (df_train['y_entry'] < y_centre) & (df_train['In_City_At_Entry'] ==0)

south_west_train = (df_train['x_entry'] < x_centre) & (df_train['y_entry'] < y_centre) & (df_train['In_City_At_Entry'] ==0)



#Conditions

conditions_train = [north_east_train,north_west_train,south_east_train,south_west_train]



#Choice List

choices = ['North_East','North_West','South_East','South_West']



#Part of the city

df_train['Area_of_City_Entry'] = np.select(conditions_train,choices,default = 'Already_In_City') 
#Different parts of the city

north_east_test = (df_test['x_entry'] > x_centre) & (df_test['y_entry'] > y_centre) & (df_test['In_City_At_Entry'] ==0)

north_west_test = (df_test['x_entry'] < x_centre) & (df_test['y_entry'] > y_centre) & (df_test['In_City_At_Entry'] ==0)

south_east_test = (df_test['x_entry'] > x_centre) & (df_test['y_entry'] < y_centre) & (df_test['In_City_At_Entry'] ==0)

south_west_test = (df_test['x_entry'] < x_centre) & (df_test['y_entry'] < y_centre) & (df_test['In_City_At_Entry'] ==0)



#Conditions

conditions_test = [north_east_test,north_west_test,south_east_test,south_west_test]



#Choice List

choices = ['North_East','North_West','South_East','South_West']



#Part of the city

df_test['Area_of_City_Entry'] = np.select(conditions_test,choices,default = 'Already_In_City') 
#List of the current variables in the data

current_columns = list(df_train.columns)



#List of new variables in our new datasert

potential_columns = ['mean_x_entry', 'mean_y_entry', 'mean_x_exit', 'mean_y_exit', 'std_x_entry', 'std_y_entry', 'std_x_exit', 'std_y_exit', 'max_x_entry', 'max_y_entry', 'max_x_exit', 'max_y_exit', 'min_x_entry', 'min_y_entry', 'min_x_exit', 'min_y_exit', 'Mean_EDistance_From_Centre', 'Std_EDistance_From_Centre', 'Max_EDistance_From_Centre', 'Min_EDistance_From_Centre', 'IQR_EDistance_From_Centre', 'Mean_SDistance_From_Centre', 'Std_SDistance_From_Centre', 'Max_SDistance_From_Centre', 'Min_SDistance_From_Centre', 'IQR_SDistance_From_Centre', 'Prop_of_time_closer_to_city_E', 'Prop_of_time_closer_to_city_S', 'Favourite_Entry_Area', 'Sum_Time_Stayed', 'Mean_Time_Stayed', 'Std_Time_Stayed', 'Mean_Euclidean_Speed', 'Std_Euclidean_Speed', 'Max_Euclidean_Speed', 'Min_Euclidean_Speed', 'IQR_Euclidean_Speed', 'Mean_Spherical_Speed', 'Std_Spherical_Speed', 'Max_Spherical_Speed', 'Min_Spherical_Speed', 'IQR_Spherical_Speed', 'time_of_first_visit', '#_of_trajectories', 'Starting_Bearing', 'Last_Bearing', 'Proportion of times out of the city', 'Number_of_times_out_of_the_city', 'Mean_Euclidean_Distance', 'Median_Euclidean_Distance', 'Sum_Euclidean_Distance', 'Std_Euclidean_Distance', 'Max_Euclidean_Distance', 'Min_Euclidean_Distance', 'IQR_Euclidean_Distance', 'Mean_Spherical_Distance', 'Median_Spherical_Distance', 'Std_Spherical_Distance', 'Sum_Spherical_Distance', 'Max_Spherical_Distance', 'Min_Spherical_Distance', 'IQR_Spherical_Distance', 'Trajec_id', '#_of_Trajec_ids', 'Mean_VMin', 'Mean_VMax', 'Mean,VAve', 'Std_VMin', 'Std_VMax', 'Std_VAve', '#_of_VNans', 'Proportion_of_#_VNans']

#Combinde the two lists

new_var_list=current_columns + potential_columns
#Make an empty data frame to new the dataset

training_data = pd.DataFrame(index=range(len(df_train['hash'].unique())),columns = new_var_list)



#Make a list of all the hashes

hashes = list(df_train['hash'].unique())



#Pick all the observations belonging to a hash



#Create a dictionary to store results

feaature_dict = dict()



for i in tqdm(range(len(hashes))):





    feaature_dict = dict()



    #The hash value

    j = hashes[i]



    #Select all the observations except the last one

    obs=df_train.loc[df_train['hash']==j]



    #Get some agg statistics

    first_obs = obs[:-1]







    #Get the last observation

    last_obs = obs.iloc[-1]

    training_data.iloc[i,:] = last_obs



    if len(first_obs) <2:

        first_obs = obs



    #Very Basic features

    feaature_dict["mean_x_entry"] = np.mean(first_obs['x_entry'])

    feaature_dict["mean_y_entry"] = np.mean(first_obs['y_entry'])

    feaature_dict["mean_x_exit"] = np.mean(first_obs['x_exit'])

    feaature_dict["mean_y_exit"] = np.mean(first_obs['y_exit'])

    feaature_dict["std_x_entry"] = np.std(first_obs['x_entry'])

    feaature_dict["std_y_entry"] = np.std(first_obs['y_entry'])

    feaature_dict["std_x_exit"] = np.std(first_obs['x_exit'])

    feaature_dict["std_y_exit"] = np.std(first_obs['y_exit'])

    feaature_dict["max_x_entry"] = np.max(first_obs['x_entry'])

    feaature_dict["max_y_entry"] = np.max(first_obs['y_entry'])

    feaature_dict["max_x_exit"] = np.max(first_obs['x_exit'])

    feaature_dict["max_y_exit"] = np.max(first_obs['y_exit'])

    feaature_dict["min_x_entry"] = np.min(first_obs['x_entry'])

    feaature_dict["min_y_entry"] = np.min(first_obs['y_entry'])

    feaature_dict["min_x_exit"] = np.min(first_obs['x_exit'])

    feaature_dict["min_y_exit"] = np.min(first_obs['y_exit'])

      

    feaature_dict['Mean_EDistance_From_Centre'] =  np.mean(first_obs['EDistance_From_Centre'])

    feaature_dict['Std_EDistance_From_Centre'] =  np.std(first_obs['EDistance_From_Centre'])

    feaature_dict['Max_EDistance_From_Centre'] =  np.max(first_obs['EDistance_From_Centre'])

    feaature_dict['Min_EDistance_From_Centre'] =  np.min(first_obs['EDistance_From_Centre'])

    feaature_dict['IQR_EDistance_From_Centre'] =  iqr(first_obs['EDistance_From_Centre'])

    

    feaature_dict['Mean_SDistance_From_Centre'] =  np.mean(first_obs['SDistance_From_Centre'])

    feaature_dict['Std_SDistance_From_Centre'] =  np.std(first_obs['SDistance_From_Centre'])

    feaature_dict['Max_SDistance_From_Centre'] =  np.max(first_obs['SDistance_From_Centre'])

    feaature_dict['Min_SDistance_From_Centre'] =  np.min(first_obs['SDistance_From_Centre'])

    feaature_dict['IQR_SDistance_From_Centre'] =  iqr(first_obs['SDistance_From_Centre'])

    

    #Proportion of times individual got closer to the city

    feaature_dict['Prop_of_time_closer_to_city_E'] = 0

    feaature_dict['Prop_of_time_closer_to_city_S'] = 0

    

    #Most popular area of entry

    feaature_dict['Favourite_Entry_Area'] = first_obs['Area_of_City_Entry'].value_counts().index[0]



    #Time stayed in seconds

    feaature_dict["Sum_Time_Stayed"] = np.sum(first_obs['Time_stayed_secs'])

    feaature_dict["Mean_Time_Stayed"] = np.mean(first_obs['Time_stayed_secs'])

    feaature_dict["Std_Time_Stayed"] = np.std(first_obs['Time_stayed_secs'])



    #Euclidean Speed

    feaature_dict['Mean_Euclidean_Speed'] = np.mean(first_obs['Euclidean_Speed'])

    feaature_dict["Std_Euclidean_Speed"] = np.std(first_obs['Euclidean_Speed'])

    feaature_dict["Max_Euclidean_Speed"] = np.max(first_obs['Euclidean_Speed'])

    feaature_dict["Min_Euclidean_Speed"] = np.min(first_obs['Euclidean_Speed'])

    feaature_dict["IQR_Euclidean_Speed"] = iqr(first_obs['Euclidean_Speed'])

    

    #Spherical Speed

    feaature_dict['Mean_Spherical_Speed'] = np.mean(first_obs['Spherical_Speed'])

    feaature_dict["Std_Spherical_Speed"] = np.std(first_obs['Spherical_Speed'])

    feaature_dict["Max_Spherical_Speed"] = np.max(first_obs['Spherical_Speed'])

    feaature_dict["Min_Spherical_Speed"] = np.min(first_obs['Spherical_Speed'])

    feaature_dict["IQR_Spherical_Speed"] = iqr(first_obs['Spherical_Speed'])    

    

    #Slightly complicated

    feaature_dict["time_of_first_visit"] = first_obs['hour_of_day'].iloc[0]

    feaature_dict["#_of_trajectories"] = len(first_obs)

    

    #Bearing   

    feaature_dict["Starting_Bearing"] = first_obs['Bearings'].iloc[0]

    feaature_dict["Last_Bearing"] = first_obs['Bearings'].iloc[0]

    

    #Previous times in the city

    feaature_dict["Proportion of times out of the city"] = first_obs['In_City'].sum()/len(obs)

    feaature_dict["Number_of_times_out_of_the_city"] = first_obs['In_City'].sum()

    

    #Euclidean Distance

    feaature_dict["Mean_Euclidean_Distance"] = np.mean(first_obs['Euclidean_Distance'])

    feaature_dict["Median_Euclidean_Distance"] = np.median(first_obs['Euclidean_Distance'])

    feaature_dict["Sum_Euclidean_Distance"] = np.sum(first_obs['Euclidean_Distance'])

    feaature_dict["Std_Euclidean_Distance"] = np.std(first_obs['Euclidean_Distance'])

    feaature_dict["Max_Euclidean_Distance"] = np.max(first_obs['Euclidean_Distance'])

    feaature_dict["Min_Euclidean_Distance"] = np.min(first_obs['Euclidean_Distance'])

    feaature_dict["IQR_Euclidean_Distance"] = iqr(first_obs['Euclidean_Distance'])

    

    #Spherical Distance

    feaature_dict["Mean_Spherical_Distance"] = np.mean(first_obs['Spherical_Distance'])

    feaature_dict["Median_Spherical_Distance"] = np.median(first_obs['Spherical_Distance'])

    feaature_dict["Std_Spherical_Distance"] = np.std(first_obs['Spherical_Distance'])

    feaature_dict["Sum_Spherical_Distance"] = np.sum(first_obs['Spherical_Distance'])

    feaature_dict["Max_Spherical_Distance"] = np.max(first_obs['Spherical_Distance'])

    feaature_dict["Min_Spherical_Distance"] = np.min(first_obs['Spherical_Distance'])

    feaature_dict["IQR_Spherical_Distance"] = iqr(first_obs['Spherical_Distance'])    

      

    

    #This depends on if there's more than one observation

    if len(first_obs) >=2:

      

      #Bearing

      feaature_dict["Starting_Bearing"] = first_obs['Bearings'].iloc[0]

      feaature_dict["Last_Bearing"] = first_obs['Bearings'].iloc[-1]

      

      #Previous times in the city

      feaature_dict["Proportion of times out of the city"] = first_obs['In_City'].sum()/len(obs)

      feaature_dict["Number_of_times_out_of_the_city"] = first_obs['In_City'].sum()

      

      #Euclidean Distance

      feaature_dict["Mean_Euclidean_Distance"] = np.mean(first_obs['Euclidean_Distance'])

      feaature_dict["Median_Euclidean_Distance"] = np.median(first_obs['Euclidean_Distance'])

      feaature_dict["Sum_Euclidean_Distance"] = np.sum(first_obs['Euclidean_Distance'])

      feaature_dict["Std_Euclidean_Distance"] = np.std(first_obs['Euclidean_Distance'])

      feaature_dict["Max_Euclidean_Distance"] = np.max(first_obs['Euclidean_Distance'])

      feaature_dict["Min_Euclidean_Distance"] = np.min(first_obs['Euclidean_Distance'])

      feaature_dict["IQR_Euclidean_Distance"] = iqr(first_obs['Euclidean_Distance'])

    

      #Spherical Distance

      feaature_dict["Mean_Spherical_Distance"] = np.mean(first_obs['Spherical_Distance'])

      feaature_dict["Median_Spherical_Distance"] = np.median(first_obs['Spherical_Distance'])

      feaature_dict["Std_Spherical_Distance"] = np.std(first_obs['Spherical_Distance'])

      feaature_dict["Sum_Spherical_Distance"] = np.sum(first_obs['Spherical_Distance'])

      feaature_dict["Max_Spherical_Distance"] = np.max(first_obs['Spherical_Distance'])

      feaature_dict["Min_Spherical_Distance"] = np.min(first_obs['Spherical_Distance'])

      feaature_dict["IQR_Spherical_Distance"] = iqr(first_obs['Spherical_Distance'])  

      

      feaature_dict['Prop_of_time_closer_to_city_E'] = first_obs['EDistance_From_Centre'].diff()[1:].map(lambda x: 1 if x < 0 else 0).sum()/(len(first_obs) -1)

      feaature_dict['Prop_of_time_closer_to_city_S'] = first_obs['SDistance_From_Centre'].diff()[1:].map(lambda x: 1 if x < 0 else 0).sum()/(len(first_obs) - 1)

    



      

    #Trajectory id

    feaature_dict["Trajec_id"] = first_obs['New_Hash_Var_1'].iloc[0]

    feaature_dict['#_of_Trajec_ids'] = len(first_obs)



    #Vmax features

    feaature_dict["Mean_VMin"] = np.mean(first_obs['vmin'])

    feaature_dict["Mean_VMax"] = np.mean(first_obs['vmax'])

    feaature_dict["Mean,VAve"] = np.mean(first_obs['vmean'])

    feaature_dict["Std_VMin"] = np.std(first_obs['vmin'])

    feaature_dict["Std_VMax"] = np.std(first_obs['vmax'])

    feaature_dict["Std_VAve"] = np.std(first_obs['vmean'])

    feaature_dict["#_of_VNans"] = first_obs['vmean'].isna().sum()

    feaature_dict['Proportion_of_#_VNans'] = first_obs['vmean'].isna().sum()/ len(first_obs)



    training_data.loc[i,list(pd.concat([last_obs,pd.Series(feaature_dict)]).index)] = pd.concat([last_obs,pd.Series(feaature_dict)])

#Make an empty data frame to new the dataset

testing_data = pd.DataFrame(index=range(len(df_test['hash'].unique())),columns = new_var_list)



#Make a list of all the hashes

hashes = list(df_test['hash'].unique())



#Pick all the observations belonging to a hash



#Create a dictionary to store results

feaature_dict = dict()



for i in tqdm(range(len(hashes))):





    feaature_dict = dict()



    #The hash value

    j = hashes[i]



    #Select all the observations except the last one

    obs=df_test.loc[df_test['hash']==j]



    #Get some agg statistics

    first_obs = obs[:-1]







    #Get the last observation

    last_obs = obs.iloc[-1]

    testing_data.iloc[i,:] = last_obs



    if len(first_obs) <2:

        first_obs = obs



    #Very Basic features

    feaature_dict["mean_x_entry"] = np.mean(first_obs['x_entry'])

    feaature_dict["mean_y_entry"] = np.mean(first_obs['y_entry'])

    feaature_dict["mean_x_exit"] = np.mean(first_obs['x_exit'])

    feaature_dict["mean_y_exit"] = np.mean(first_obs['y_exit'])

    feaature_dict["std_x_entry"] = np.std(first_obs['x_entry'])

    feaature_dict["std_y_entry"] = np.std(first_obs['y_entry'])

    feaature_dict["std_x_exit"] = np.std(first_obs['x_exit'])

    feaature_dict["std_y_exit"] = np.std(first_obs['y_exit'])

    feaature_dict["max_x_entry"] = np.max(first_obs['x_entry'])

    feaature_dict["max_y_entry"] = np.max(first_obs['y_entry'])

    feaature_dict["max_x_exit"] = np.max(first_obs['x_exit'])

    feaature_dict["max_y_exit"] = np.max(first_obs['y_exit'])

    feaature_dict["min_x_entry"] = np.min(first_obs['x_entry'])

    feaature_dict["min_y_entry"] = np.min(first_obs['y_entry'])

    feaature_dict["min_x_exit"] = np.min(first_obs['x_exit'])

    feaature_dict["min_y_exit"] = np.min(first_obs['y_exit'])

      

    feaature_dict['Mean_EDistance_From_Centre'] =  np.mean(first_obs['EDistance_From_Centre'])

    feaature_dict['Std_EDistance_From_Centre'] =  np.std(first_obs['EDistance_From_Centre'])

    feaature_dict['Max_EDistance_From_Centre'] =  np.max(first_obs['EDistance_From_Centre'])

    feaature_dict['Min_EDistance_From_Centre'] =  np.min(first_obs['EDistance_From_Centre'])

    feaature_dict['IQR_EDistance_From_Centre'] =  iqr(first_obs['EDistance_From_Centre'])

    

    feaature_dict['Mean_SDistance_From_Centre'] =  np.mean(first_obs['SDistance_From_Centre'])

    feaature_dict['Std_SDistance_From_Centre'] =  np.std(first_obs['SDistance_From_Centre'])

    feaature_dict['Max_SDistance_From_Centre'] =  np.max(first_obs['SDistance_From_Centre'])

    feaature_dict['Min_SDistance_From_Centre'] =  np.min(first_obs['SDistance_From_Centre'])

    feaature_dict['IQR_SDistance_From_Centre'] =  iqr(first_obs['SDistance_From_Centre'])

    

    #Proportion of times individual got closer to the city

    feaature_dict['Prop_of_time_closer_to_city_E'] = 0

    feaature_dict['Prop_of_time_closer_to_city_S'] = 0

    

    #Most popular area of entry

    feaature_dict['Favourite_Entry_Area'] = first_obs['Area_of_City_Entry'].value_counts().index[0]



    #Time stayed in seconds

    feaature_dict["Sum_Time_Stayed"] = np.sum(first_obs['Time_stayed_secs'])

    feaature_dict["Mean_Time_Stayed"] = np.mean(first_obs['Time_stayed_secs'])

    feaature_dict["Std_Time_Stayed"] = np.std(first_obs['Time_stayed_secs'])



    #Euclidean Speed

    feaature_dict['Mean_Euclidean_Speed'] = np.mean(first_obs['Euclidean_Speed'])

    feaature_dict["Std_Euclidean_Speed"] = np.std(first_obs['Euclidean_Speed'])

    feaature_dict["Max_Euclidean_Speed"] = np.max(first_obs['Euclidean_Speed'])

    feaature_dict["Min_Euclidean_Speed"] = np.min(first_obs['Euclidean_Speed'])

    feaature_dict["IQR_Euclidean_Speed"] = iqr(first_obs['Euclidean_Speed'])

    

    #Spherical Speed

    feaature_dict['Mean_Spherical_Speed'] = np.mean(first_obs['Spherical_Speed'])

    feaature_dict["Std_Spherical_Speed"] = np.std(first_obs['Spherical_Speed'])

    feaature_dict["Max_Spherical_Speed"] = np.max(first_obs['Spherical_Speed'])

    feaature_dict["Min_Spherical_Speed"] = np.min(first_obs['Spherical_Speed'])

    feaature_dict["IQR_Spherical_Speed"] = iqr(first_obs['Spherical_Speed'])    

    

    #Slightly complicated

    feaature_dict["time_of_first_visit"] = first_obs['hour_of_day'].iloc[0]

    feaature_dict["#_of_trajectories"] = len(first_obs)

    

    #Bearing   

    feaature_dict["Starting_Bearing"] = first_obs['Bearings'].iloc[0]

    feaature_dict["Last_Bearing"] = first_obs['Bearings'].iloc[0]

    

    #Previous times in the city

    feaature_dict["Proportion of times out of the city"] = first_obs['In_City'].sum()/len(obs)

    feaature_dict["Number_of_times_out_of_the_city"] = first_obs['In_City'].sum()

    

    #Euclidean Distance

    feaature_dict["Mean_Euclidean_Distance"] = np.mean(first_obs['Euclidean_Distance'])

    feaature_dict["Median_Euclidean_Distance"] = np.median(first_obs['Euclidean_Distance'])

    feaature_dict["Sum_Euclidean_Distance"] = np.sum(first_obs['Euclidean_Distance'])

    feaature_dict["Std_Euclidean_Distance"] = np.std(first_obs['Euclidean_Distance'])

    feaature_dict["Max_Euclidean_Distance"] = np.max(first_obs['Euclidean_Distance'])

    feaature_dict["Min_Euclidean_Distance"] = np.min(first_obs['Euclidean_Distance'])

    feaature_dict["IQR_Euclidean_Distance"] = iqr(first_obs['Euclidean_Distance'])

    

    #Spherical Distance

    feaature_dict["Mean_Spherical_Distance"] = np.mean(first_obs['Spherical_Distance'])

    feaature_dict["Median_Spherical_Distance"] = np.median(first_obs['Spherical_Distance'])

    feaature_dict["Std_Spherical_Distance"] = np.std(first_obs['Spherical_Distance'])

    feaature_dict["Sum_Spherical_Distance"] = np.sum(first_obs['Spherical_Distance'])

    feaature_dict["Max_Spherical_Distance"] = np.max(first_obs['Spherical_Distance'])

    feaature_dict["Min_Spherical_Distance"] = np.min(first_obs['Spherical_Distance'])

    feaature_dict["IQR_Spherical_Distance"] = iqr(first_obs['Spherical_Distance'])    

      

    

    #This depends on if there's more than one observation

    if len(first_obs) >=2:

      

      #Bearing

      feaature_dict["Starting_Bearing"] = first_obs['Bearings'].iloc[0]

      feaature_dict["Last_Bearing"] = first_obs['Bearings'].iloc[-1]

      

      #Previous times in the city

      feaature_dict["Proportion of times out of the city"] = first_obs['In_City'].sum()/len(obs)

      feaature_dict["Number_of_times_out_of_the_city"] = first_obs['In_City'].sum()

      

      #Euclidean Distance

      feaature_dict["Mean_Euclidean_Distance"] = np.mean(first_obs['Euclidean_Distance'])

      feaature_dict["Median_Euclidean_Distance"] = np.median(first_obs['Euclidean_Distance'])

      feaature_dict["Sum_Euclidean_Distance"] = np.sum(first_obs['Euclidean_Distance'])

      feaature_dict["Std_Euclidean_Distance"] = np.std(first_obs['Euclidean_Distance'])

      feaature_dict["Max_Euclidean_Distance"] = np.max(first_obs['Euclidean_Distance'])

      feaature_dict["Min_Euclidean_Distance"] = np.min(first_obs['Euclidean_Distance'])

      feaature_dict["IQR_Euclidean_Distance"] = iqr(first_obs['Euclidean_Distance'])

    

      #Spherical Distance

      feaature_dict["Mean_Spherical_Distance"] = np.mean(first_obs['Spherical_Distance'])

      feaature_dict["Median_Spherical_Distance"] = np.median(first_obs['Spherical_Distance'])

      feaature_dict["Std_Spherical_Distance"] = np.std(first_obs['Spherical_Distance'])

      feaature_dict["Sum_Spherical_Distance"] = np.sum(first_obs['Spherical_Distance'])

      feaature_dict["Max_Spherical_Distance"] = np.max(first_obs['Spherical_Distance'])

      feaature_dict["Min_Spherical_Distance"] = np.min(first_obs['Spherical_Distance'])

      feaature_dict["IQR_Spherical_Distance"] = iqr(first_obs['Spherical_Distance'])  

      

      feaature_dict['Prop_of_time_closer_to_city_E'] = first_obs['EDistance_From_Centre'].diff()[1:].map(lambda x: 1 if x < 0 else 0).sum()/(len(first_obs) -1)

      feaature_dict['Prop_of_time_closer_to_city_S'] = first_obs['SDistance_From_Centre'].diff()[1:].map(lambda x: 1 if x < 0 else 0).sum()/(len(first_obs) - 1)

    



      

    #Trajectory id

    feaature_dict["Trajec_id"] = first_obs['New_Hash_Var_1'].iloc[0]

    feaature_dict['#_of_Trajec_ids'] = len(first_obs)



    #Vmax features

    feaature_dict["Mean_VMin"] = np.mean(first_obs['vmin'])

    feaature_dict["Mean_VMax"] = np.mean(first_obs['vmax'])

    feaature_dict["Mean,VAve"] = np.mean(first_obs['vmean'])

    feaature_dict["Std_VMin"] = np.std(first_obs['vmin'])

    feaature_dict["Std_VMax"] = np.std(first_obs['vmax'])

    feaature_dict["Std_VAve"] = np.std(first_obs['vmean'])

    feaature_dict["#_of_VNans"] = first_obs['vmean'].isna().sum()

    feaature_dict['Proportion_of_#_VNans'] = first_obs['vmean'].isna().sum()/ len(first_obs)



    testing_data.loc[i,list(pd.concat([last_obs,pd.Series(feaature_dict)]).index)] = pd.concat([last_obs,pd.Series(feaature_dict)])

#Saving the data

training_data.to_pickle("training_data.pkl")

testing_data.to_pickle("testing_data.pkl")