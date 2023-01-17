import pickle
from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()

train = env.get_training_data()

test = []
for (market_obs, news_obs, predictions_template) in env.get_prediction_days():
    test.append((market_obs, news_obs, predictions_template))
    predictions_template.confidenceValue = 0.0
    env.predict(predictions_template)

pickle.dump((train, test), open('data.p', 'wb'))
