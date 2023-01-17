import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy  as np 

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid")
games = pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")

games = pd.DataFrame(games)

games.columns = [i.lower() for i in games.columns]

# let's drop duplicate rows 

games = games.drop_duplicates()

print("Dataset shape :",games.shape)
# let's rename some columns 

games.rename(columns = {"na_sales":"north_usa_sales","eu_sales":"europe_sales","jp_sales":"japan_sales"},inplace = True)

games.head()
# let's drop row's  if it as  atleast one null value.

games.dropna(how = "any",inplace = True)

print("Dataset shape after droping row's with null valu's :",games.shape)
missing_percent = pd.DataFrame(games.isnull().sum(),columns = ["missing percent"])

missing_percent
# feature transformation

games.year = games.year.astype(int)
max_sold = games.global_sales.max()

print("game got highest global sales : ") 

name = games[games.global_sales == max_sold][["name","global_sales"]]

name
min_sold = games.global_sales.min()

print("Games got least global sales : ") 

games[games.global_sales == min_sold][["name","global_sales"]]
plt.rcParams['figure.figsize'] = (14,10)

year_wise_game_sales  = pd.pivot_table(games ,index = "year" ,values = "global_sales",aggfunc = np.sum)

sns.barplot(year_wise_game_sales["global_sales"],year_wise_game_sales.index,orient = "h")

plt.title("Year wise global game sales :")
plt.rcParams['figure.figsize'] = (8,6)

platform_wise_game_sales  = pd.pivot_table(games ,index = "platform" ,values = "global_sales",aggfunc = np.sum)

platform_wise_game_sales  = platform_wise_game_sales.sort_values(by = "global_sales",ascending  = False).head(10)

sns.barplot(platform_wise_game_sales["global_sales"],platform_wise_game_sales.index,orient = "h")

plt.title("Top 10 Platform wise global game sales :")
plt.rcParams['figure.figsize'] = (8,6)

genre_wise_game_sales  = pd.pivot_table(games ,index = "genre" ,values = "global_sales",aggfunc = np.sum)

genre_wise_game_sales  = genre_wise_game_sales.sort_values(by = "global_sales",ascending  = False).head(10)

sns.barplot(genre_wise_game_sales["global_sales"],genre_wise_game_sales.index,orient = "h",palette = "husl")

plt.title("Top 10 Genre wise global game sales :")
plt.rcParams['figure.figsize'] = (8,6)

publisher_wise_game_sales  = pd.pivot_table(games ,index = "publisher" ,values = "global_sales",aggfunc = np.sum)

publisher_wise_game_sales  = publisher_wise_game_sales.sort_values(by = "global_sales",ascending  = False).head(10)

sns.barplot(publisher_wise_game_sales["global_sales"],publisher_wise_game_sales.index,orient = "h",palette = "viridis")

plt.title("Top 10 Publisher wise global game sales :") 
top_five_action_games = games[games.genre == "Action"][["name","global_sales"]]

top_five_action_games = top_five_action_games.sort_values(by = "global_sales",ascending = False )

top_five_action_games = top_five_action_games.drop_duplicates(["name"]).head(5)

sns.barplot(top_five_action_games["global_sales"],top_five_action_games["name"])

plt.title("Top Five Action Games And Their Sales World Wide : ")

top_five_action_games
top_five_Sports_games = games[games.genre == "Sports"][["name","global_sales"]]

top_five_Sports_games = top_five_Sports_games.sort_values(by = "global_sales",ascending = False )

top_five_Sports_games = top_five_Sports_games.drop_duplicates(["name"]).head(5)

sns.barplot(top_five_Sports_games["global_sales"],top_five_Sports_games["name"])

plt.title("Top Five Sport Games And Their Sales World Wide : ")

top_five_Sports_games
top_five_Shooting_games = games[games.genre == "Shooter"][["name","global_sales"]]

top_five_Shooting_games = top_five_Shooting_games.sort_values(by = "global_sales",ascending = False )

top_five_Shooting_games = top_five_Shooting_games.drop_duplicates(["name"]).head(5)

plt.pie(x = top_five_Shooting_games["global_sales"],labels= list(top_five_Shooting_games.name),shadow=True)

plt.title("Top Five Shooting Games And Their Sales World Wide : ")

top_five_Shooting_games
top_five_Platform_games = games[games.genre == "Platform"][["name","global_sales"]]

top_five_Platform_games = top_five_Platform_games.sort_values(by = "global_sales",ascending = False )

top_five_Platform_games = top_five_Platform_games.drop_duplicates(["name"]).head(5)

sns.barplot(top_five_Platform_games["global_sales"],top_five_Platform_games["name"],palette = 'colorblind')

plt.title("Top Five Platform Games And Their Sales World Wide : ")

top_five_Platform_games
top_five_Role_Playing_games = games[games.genre == "Role-Playing"][["name","global_sales"]]

top_five_Role_Playing_games = top_five_Role_Playing_games.sort_values(by = "global_sales",ascending = False )

top_five_Role_Playing_games = top_five_Role_Playing_games.drop_duplicates(["name"]).head(5)

sns.barplot(top_five_Role_Playing_games["global_sales"],top_five_Role_Playing_games["name"])

plt.title("Top Five Platform Games And Their Sales World Wide : ")

top_five_Role_Playing_games
north_usa_highest_sold_game = games.north_usa_sales.max()

print("highest sold Game in North_USA :")

games[games["north_usa_sales"] == north_usa_highest_sold_game][["name","north_usa_sales"]]
Europe_highest_sold_game = games.europe_sales.max()

print("highest sold Game in EUROPE :")

games[games["europe_sales"] == Europe_highest_sold_game][["name","europe_sales"]]
Japan_highest_sold_game = games.japan_sales.max()

print("highest sold Game in Japan :")

games[games["japan_sales"] == Japan_highest_sold_game][["name","japan_sales"]]
other_sales_highest_sold_game = games.other_sales.max()

print("highest sold Game in Other_Sales :")

games[games["other_sales"] == other_sales_highest_sold_game][["name","other_sales"]]
tot_sales_year_wise = pd.pivot_table(games,index = "year",values = "global_sales",aggfunc= np.sum)

#print(avg_sales_year_wise) #uncomment this to know Total sale's value's with respect each year.

plt.plot(tot_sales_year_wise.index,tot_sales_year_wise["global_sales"],color = 'g',marker = "*")

plt.title("Total Sale's Year Wise")

plt.xlabel("Year's")

plt.ylabel("avg global_sales")
plt.figure(figsize = (14,8))

ax = games.north_usa_sales[:200].plot.kde()

ax = games.europe_sales[:200].plot.kde()

ax = games.japan_sales[:200].plot.kde()

ax = games.other_sales[:200].plot.kde()

ax = games.global_sales[:200].plot.kde()

ax.legend()
# dropping columns

data = games.copy()

data.drop(["north_usa_sales","europe_sales","japan_sales","other_sales","rank"],axis = 1,inplace = True)
def data_encode(x_data):

    for i in x_data.columns:

        x_data[i]=x_data[i].factorize()[0]

        

    return x_data    

    

x_data = data.drop("global_sales",axis = 1)

y_data = data["global_sales"]

x_data = data_encode(x_data)
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x_data,y_data,test_size=.3,random_state=1)
lr_model =LinearRegression()

lr_model.fit(xtrain,ytrain)

ypred=lr_model.predict(xtest)

n = len(xtest)

p = xtest.shape[1]

r2_value = r2_score(ytest,ypred)

adjusted_r2_score = 1 - (((1-r2_value)*(n-1)) /(n-p-1))

print("r2_score for Linear Reg model : ",r2_score(ytest,ypred))

print("adjusted_r2_score Value       : ",adjusted_r2_score)                         

print("MSE for Linear Regression     : ",mean_squared_error(ytest,ypred))
rf_model = RandomForestRegressor(n_estimators=200,min_samples_split=20,random_state=43)

rf_model.fit(xtrain,ytrain)

ypred = rf_model.predict(xtest)

n = len(xtest)

p = xtest.shape[1]

r2_value = r2_score(ytest,ypred)

adjusted_r2_score = 1 - (((1-r2_value)*(n-1)) /(n-p-1))

print("r2_score for Random Forest Reg model : ",r2_score(ytest,ypred))

print("adjusted_r2_score Value              : ",adjusted_r2_score)

print("MSE for Random Forest Regression     : ",mean_squared_error(ytest,ypred))
# random forest seems fine
# My favourite Game is "Tom clancy's Splinter Cell ",

# What's your's comment below.