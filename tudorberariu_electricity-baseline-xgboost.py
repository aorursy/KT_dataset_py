## 0. Install required packages



!pip install xgboost pandas python-dateutil
## 1. Reading data into a pandas DataFrame, and inspecting the columns a bit



import pandas as pd

df = pd.read_csv("../input/train_electricity.csv")



print("Dataset has", len(df), "entries.")



print(f"\n\t{'Column':20s} | {'Type':8s} | {'Min':12s} | {'Max':12s}\n")

for col_name in df.columns:

    col = df[col_name]

    print(f"\t{col_name:20s} | {str(col.dtype):8s} | {col.min():12.1f} | {col.max():12.1f}")
## 2. Adding some datetime related features



def add_datetime_features(df):

    features = ["Year", "Week", "Day", "Dayofyear", "Month", "Dayofweek",

                "Is_year_end", "Is_year_start", "Is_month_end", "Is_month_start",

                "Hour", "Minute",]

    one_hot_features = ["Month", "Dayofweek"]



    datetime = pd.to_datetime(df.Date * (10 ** 9))



    df['Datetime'] = datetime  # We won't use this for training, but we'll remove it later



    for feature in features:

        new_column = getattr(datetime.dt, feature.lower())

        if feature in one_hot_features:

            df = pd.concat([df, pd.get_dummies(new_column, prefix=feature)], axis=1)

        else:

            df[feature] = new_column

    return df



df = add_datetime_features(df)

df.columns
## 3. Split data into train / validation (leaving the last six months for validation)



from dateutil.relativedelta import relativedelta



eval_from = df['Datetime'].max() + relativedelta(months=-6)  # Here we set the 6 months threshold

train_df = df[df['Datetime'] < eval_from]

valid_df = df[df['Datetime'] >= eval_from]



print(f"Train data: {train_df['Datetime'].min()} -> {train_df['Datetime'].max()} | {len(train_df)} samples.")

print(f"Valid data: {valid_df['Datetime'].min()} -> {valid_df['Datetime'].max()} | {len(valid_df)} samples.")
## 4. Prepare data for XGBoosting (DataFrame --> DMatrix)



import xgboost as xgb



label_col = "Consumption_MW"  # The target values are in this column

to_drop = [label_col, "Date", "Datetime"]  # Columns we do not need for training



xg_trn_data = xgb.DMatrix(train_df.drop(columns=to_drop), label=train_df[label_col])

xg_vld_data = xgb.DMatrix(valid_df.drop(columns=to_drop), label=valid_df[label_col])
## 5. Train (mostly with default parameters; it overfits like hell)



num_round = 300

xgb_param = {"objective": "reg:squarederror" if xgb.__version__ > '0.82' else 'reg:linear',

            'eta': 0.1, 'booster': 'gbtree', 'max_depth': 5}

watchlist = [(xg_trn_data, "train"), (xg_vld_data, "valid")]



bst = xgb.train(xgb_param, xg_trn_data, num_round, watchlist)
## 6. Read test dataset, use the bst for prediction, save submission csv



test_df = pd.read_csv("../input/test_electricity.csv")

test_df = add_datetime_features(test_df)

xgb_test_data = xgb.DMatrix(test_df.drop(columns=["Date", "Datetime"]))



solution_df = pd.DataFrame(test_df["Date"])

solution_df["Consumption_MW"] = bst.predict(xgb_test_data)

solution_df.to_csv("sample_submission.csv", index=False)

print("Done!")