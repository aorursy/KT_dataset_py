#Вариант 8
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
my_filepath = "../input/steam-data/steam.csv"
steam_data = pd.read_csv(my_filepath)
steam_data.head()
games_valve = steam_data[steam_data.developer == 'Valve']
games = games_valve.groupby('genres').agg(
    **{
    'Average price': pd.NamedAgg(column='price', aggfunc='mean'),
    }
)
print('completed')
games
plt.figure(figsize=(30,10)) 
plt.title("The average price of Valve games")
sns.barplot(x=games.index, y=games['Average price'])
plt.show()