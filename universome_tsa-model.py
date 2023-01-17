import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.ar_model import AR
from kaggle.competitions import twosigmanews

random.seed(42)
np.random.seed(42)

env = twosigmanews.make_env()
market_train_df, news_train_df = env.get_training_data()
market_train_df.head()
news_train_df.head()
market_train_df.assetCode.unique().shape
market_train_df[market_train_df.assetCode == 'AAPL.O'].plot(x='time', y='returnsOpenNextMktres10', figsize=(10, 5))
plot_acf(market_train_df[market_train_df.assetCode == 'AAPL.O'].returnsOpenNextMktres10, lags=30);
plot_pacf(market_train_df[market_train_df.assetCode == 'AAPL.O'].returnsOpenNextMktres10, lags=30);
from tqdm import tqdm

assets = market_train_df.assetCode.unique()
groups = market_train_df.groupby('assetCode').groups
# df = pd.DataFrame(index=market_train_df.time.unique())
df = {}

for asset in tqdm(assets):
    #df[asset] = market_train_df.iloc[groups[asset]].set_index('time').returnsOpenNextMktres10
    df[asset] = market_train_df.iloc[groups[asset]].returnsOpenNextMktres10.values
for asset in assets:
    df[asset] = df[asset][~np.isnan(df[asset])]
    df[asset] = df[asset][-250:]
lens = [len(df[asset]) for asset in assets]
plt.hist(lens);
min(lens)
import warnings; warnings.filterwarnings("ignore")
# from tqdm import tqdm
# from sklearn.metrics import mean_squared_error

# targets = [df[asset][-1] for asset in assets]
# scores = []
# hps = []

# NUM_ASSETS_FOR_VAL = 100
# assets_to_use = random.sample(list(assets), NUM_ASSETS_FOR_VAL)
# targets_to_use = [t for t,a in zip(targets, assets) if a in set(assets_to_use)]

# for history_size in [250, 100, 50]:
#     for p in [5, 3, 1]:
#         for d in [1, 0]:
#             for q in [3, 1]:
#                 order = (p, d, q)

#                 predictions = []

#                 for asset in tqdm(assets_to_use):
#                     train = df[asset][-history_size:-1]

#                     try:
#                         model = ARIMA(train, order).fit()
#                         pred = model.forecast()[0][0]
#                         if np.isnan(pred): pred = 0.
#                         predictions.append(pred)
#                     except ValueError:
#                         predictions.append(0.)
#                     except np.linalg.LinAlgError:
#                         predictions.append(0.)

#                 scores.append(mean_squared_error(targets_to_use, predictions))
#                 hps.append([history_size, order])
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

targets = [df[asset][-1] for asset in assets]
scores = []
hps = []

for history_size in [500, 250, 100, 50]:
    predictions = []

    for asset in tqdm(assets):
        train = df[asset][-history_size:-1]

        try:
            model = AR(train).fit()
            pred = model.predict()[-1]
            if np.isnan(pred): pred = 0.
            predictions.append(pred)
        except ValueError:
            predictions.append(0.)
        except np.linalg.LinAlgError:
            predictions.append(0.)

    scores.append(mean_squared_error(targets, predictions))
    hps.append(history_size)
plt.figure(figsize=(10, 5))
plt.plot(scores)
print('Best found hp:', hps[np.argmin(scores)])
days = env.get_prediction_days()
# env.predict(predictions.reset_index())
# order = (1, 1, 3)
history_size = 100

for (obs, _, predictions) in tqdm(days):
    predictions.set_index('assetCode', inplace=True)
    obs_dict = {row.assetCode: row.returnsOpenPrevMktres10 for row in obs.itertuples()}

    for asset in obs_dict:
        if asset in df:
            df[asset] = np.concatenate([[obs_dict[asset]], df[asset]])
            df[asset] = df[asset][~np.isnan(df[asset])]
            df[asset] = df[asset][-history_size:]
        else:
            df[asset] = np.array([obs_dict[asset]])

    for asset in obs_dict:
        try:
            #model = ARIMA(df[asset], order).fit(maxiter=50)
            #pred = model.forecast()[0][0]
            model = AR(df[asset]).fit()
            pred = model.predict()[-1]
            if np.isnan(pred): pred = 0.
        except ValueError:
            pred = 0.
        except np.linalg.LinAlgError:
            pred = 0.
        
        predictions.ix[asset, 'confidenceValue'] = pred
    
    #predictions = pd.DataFrame(predictions)
    predictions.confidenceValue.clip(-1, 1, inplace=True)
    env.predict(predictions.reset_index())
env.write_submission_file()