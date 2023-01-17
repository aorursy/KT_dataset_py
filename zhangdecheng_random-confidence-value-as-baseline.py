import random
import pandas as pd
import os
from kaggle.competitions import twosigmanews
# Create the environment
env = twosigmanews.make_env()
def naive_predict(market_obs_df, predictions_template_df):
    market_obs_df = market_obs_df.set_index('assetCode')
    predictions_template_df['confidenceValue'] = -1+2*random.random()
    return predictions_template_df
days = env.get_prediction_days()
for (market_obs_df, _, predictions_template_df) in days:
    predictions_df = naive_predict(market_obs_df, predictions_template_df)
    env.predict(predictions_df)

env.write_submission_file()