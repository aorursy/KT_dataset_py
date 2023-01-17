import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = 30, 10
df = pd.read_parquet('/kaggle/input/binance-full-history/ETH-BTC.parquet')

df
# window of minutes times hours times days
window = 60 * 24 * 50
df['moving_average'] = df['open'].rolling(window).mean()
df[['open', 'moving_average']].plot(title='ETH BTC', color=['black', 'red', 'green'])
ax = df['volume'].plot(title='ETH BTC', color='black', legend=True)
df['number_of_trades'].plot(title='ETH BTC', color='gold', legend=True, ax=ax)