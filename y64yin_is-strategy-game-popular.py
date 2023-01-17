import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
import pandas as pd

appstore_games = pd.read_csv("../input/17k-apple-app-store-strategy-games/appstore_games.csv")

appstore_games
sns.distplot(a=appstore_games['Average User Rating'].dropna(), kde=False)
sns.kdeplot(data=appstore_games['Average User Rating'].dropna(), shade=True)
sns.jointplot(x=appstore_games['Average User Rating'].dropna(), y=appstore_games['User Rating Count'].dropna(), kind="kde",ylim=300000)
