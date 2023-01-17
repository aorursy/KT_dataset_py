import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
my_filepath = "../input/googleplaystore.csv"
google_data = pd.read_csv(my_filepath)
google_data.head()
apps_Free = google_data[google_data.Type == 'Free']
apps_Pay = google_data[google_data.Type != 'Free']
category_Free = apps_Free.groupby('Genres').agg(
    **{
    'Average rating': pd.NamedAgg(column='Rating', aggfunc='mean'),
    }
)
print('completed')
apps_Free


apps_Free=apps_Free.groupby(['Genres'])['Rating','Genres'].mean()
apps_Free.sort_values(by=['Rating'], ascending=False,inplace=True)

apps_Free.reset_index(level=0, inplace=True)
apps_Free = apps_Free.iloc[1:10]
plt.figure(figsize=(20,10))

sns.barplot(x=apps_Free['Genres'], y=apps_Free['Rating'])
category_Pay = apps_Pay.groupby('Genres').agg(
    **{
    'Average rating': pd.NamedAgg(column='Rating', aggfunc='mean'),
    }
)
print('completed')
apps_Pay
apps_Pay=apps_Pay.groupby(['Genres'])['Rating','Genres'].mean()
apps_Pay.sort_values(by=['Rating'], ascending=False,inplace=True)

apps_Pay.reset_index(level=0, inplace=True)
apps_Pay = apps_Pay.iloc[1:10]
plt.figure(figsize=(20,10))

sns.barplot(x=apps_Pay['Genres'], y=apps_Pay['Rating'])