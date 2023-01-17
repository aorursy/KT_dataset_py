import numpy as np 

import pandas as pd 
test = pd.read_json("../input/test.json")

train = pd.read_json("../input/train.json")
features = pd.DataFrame({'feature': [j for i in test.features.values for j in i]})

features['dummy'] = 1

top_features = features.groupby('feature').count().sort_values('dummy', ascending=False).head(100)



features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price", "num_photos", "num_features", "num_description_words"]



for x in top_features.index:

    fn = x.lower().replace(' ', '_').replace('/', '_').replace('-', '_');

    df[fn] = df['features'].map(lambda a: x in a);

    features_to_use.append(fn)

    

print(features_to_use)
train['num_photos'] = train['photos'].map(len)

train['num_features'] = train['features'].apply(len)

train['num_description_words'] = train['description'].apply(lambda x: len(x.split(' ')))

    

test['num_photos'] = test['photos'].map(len)

test['num_features'] = test['features'].apply(len)

test['num_description_words'] = test['description'].apply(lambda x: len(x.split(' ')))



for x in top_features.index:

    fn = x.lower().replace(' ', '_').replace('/', '_').replace('-', '_')

    train[fn] = train['features'].map(lambda a: x in a)

    test[fn] = test['features'].map(lambda a: x in a)
import xgboost as xgb



def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):

    param = {}

    param['objective'] = 'multi:softprob'

    param['eta'] = 0.1

    param['max_depth'] = 6

    param['silent'] = 1

    param['num_class'] = 3

    param['eval_metric'] = "mlogloss"

    param['min_child_weight'] = 1

    param['subsample'] = 0.7

    param['colsample_bytree'] = 0.7

    param['seed'] = seed_val

    num_rounds = num_rounds



    plst = list(param.items())

    xgtrain = xgb.DMatrix(train_X, label=train_y)



    if test_y is not None:

        xgtest = xgb.DMatrix(test_X, label=test_y)

        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]

        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)

    else:

        xgtest = xgb.DMatrix(test_X)

        model = xgb.train(plst, xgtrain, num_rounds)



    pred_test_y = model.predict(xgtest)

    return pred_test_y, model
features_to_use = set(features_to_use)

print(len(features_to_use))

print(len(set(features_to_use)))

#features_to_use = set(features_to_use)
train_X = train[features_to_use]

test_X = test[features_to_use]



target_num_map = {'high':0, 'medium':1, 'low':2}

train_y = np.array(train['interest_level'].apply(lambda x: target_num_map[x]))

print(train_X.shape, test_X.shape)
train_X
preds, model = runXGB(train_X, train_y, test_X, feature_names=features_to_use, num_rounds=400)

out_df = pd.DataFrame(preds)

out_df.columns = ["high", "medium", "low"]

out_df["listing_id"] = test_df.listing_id.values

out_df.to_csv("xgb_starter2.csv", index=False)