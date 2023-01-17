import lightgbm as lgb

import numpy as np

import pandas as pd

pd.set_option("max_columns", 500)

pd.set_option("max_rows", 300)



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="white")



from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder



import warnings

warnings.filterwarnings("ignore")



from bokeh.models import Panel, Tabs

from bokeh.io import output_notebook, show

from bokeh.plotting import figure
## defining constants

PATH_TRAIN = "/kaggle/input/covid19-global-forecasting-week-4/train.csv"

PATH_TEST = "/kaggle/input/covid19-global-forecasting-week-4/test.csv"



PATH_SUBMISSION = "submission.csv"

PATH_OUTPUT = "output.csv"



PATH_REGION_METADATA = "/kaggle/input/covid19-forecasting-metadata/region_metadata.csv"

PATH_REGION_DATE_METADATA = "/kaggle/input/covid19-forecasting-metadata/region_date_metadata.csv" # кол-во выздоровевших по датам



VAL_DAYS = 7 # дней валидации

MAD_FACTOR = 0.5 # гиперпараметр второго алгоритма

DAYS_SINCE_CASES = [1, 10, 50, 100, 500, 1000, 5000, 10000] # сколько дней прошло с момента, когда было зарегистрировано N случаев заражения



SEED = 2357



# гиперпараметры для бустинга

LGB_PARAMS = {"objective": "regression", 

              "num_leaves": 5, # максимальное количество листьев

              "learning_rate": 0.013,

              "bagging_fraction": 0.91, # сэмплируем данные

              "feature_fraction": 0.81, # сэмплируем факторы

              "reg_alpha": 0.13, # коэффициент L1 регуляризации

              "reg_lambda": 0.13, # коэффициент L2 регуляризации

              "metric": "rmse", # оптимизируемая метрика

              "seed": SEED}
## reading data

train = pd.read_csv(PATH_TRAIN)

test = pd.read_csv(PATH_TEST)



train["Date"] = pd.to_datetime(train["Date"])

test["Date"] = pd.to_datetime(test["Date"])



region_metadata = pd.read_csv(PATH_REGION_METADATA)

region_date_metadata = pd.read_csv(PATH_REGION_DATE_METADATA)



region_date_metadata["Date"] = pd.to_datetime(region_date_metadata["Date"])
output_notebook()



tab_list = []

for country in ["Italy", "Russia", "Ukraine"]:

    v = figure(plot_width=800, plot_height=400, x_axis_type="datetime", title="Covid-19 Confirmed Cases over time")

    v.line(train[train["Country_Region"] == country]["Date"], train[train["Country_Region"] == country]["ConfirmedCases"], color="green", legend_label="CC")

    v.legend.location = "top_left"

    tab = Panel(child=v, title=country)

    tab_list.append(tab)



tabs = Tabs(tabs=tab_list)

show(tabs)
## preparing data

train = train.merge(test[["ForecastId", "Province_State", "Country_Region", "Date"]], on=["Province_State", "Country_Region", "Date"], how="left")

test = test[~test["Date"].isin(train["Date"].unique())]



df_panel = pd.concat([train, test], sort=False)



# combining state and country into 'geography'

df_panel["geography"] = df_panel["Country_Region"].astype(str) + ": " + df_panel["Province_State"].astype(str)

df_panel.loc[df_panel["Province_State"].isna(), "geography"] = df_panel["Country_Region"]



# fixing data issues with cummax

df_panel["ConfirmedCases"] = df_panel.groupby("geography")["ConfirmedCases"].cummax()

df_panel["Fatalities"] = df_panel.groupby("geography")["Fatalities"].cummax()



# merging external metadata

df_panel = df_panel.merge(region_metadata, on=["Country_Region", "Province_State"])

df_panel = df_panel.merge(region_date_metadata, on=["Country_Region", "Province_State", "Date"], how="left")



# label encoding continent

df_panel["continent"] = LabelEncoder().fit_transform(df_panel["continent"])

df_panel["Date"] = pd.to_datetime(df_panel["Date"], format="%Y-%m-%d")



df_panel.sort_values(["geography", "Date"], inplace=True)
## feature engineering

min_date_train = np.min(df_panel[~df_panel["Id"].isna()]["Date"])

max_date_train = np.max(df_panel[~df_panel["Id"].isna()]["Date"])



min_date_test = np.min(df_panel[~df_panel["ForecastId"].isna()]["Date"])

max_date_test = np.max(df_panel[~df_panel["ForecastId"].isna()]["Date"])



n_dates_test = len(df_panel[~df_panel["ForecastId"].isna()]["Date"].unique())



print("Train date range:", str(min_date_train), " - ", str(max_date_train))

print("Test date range:", str(min_date_test), " - ", str(max_date_test))



# creating lag features

for lag in range(1, 41):

    df_panel[f"lag_{lag}_cc"] = df_panel.groupby("geography")["ConfirmedCases"].shift(lag)

    df_panel[f"lag_{lag}_ft"] = df_panel.groupby("geography")["Fatalities"].shift(lag)

    df_panel[f"lag_{lag}_rc"] = df_panel.groupby("geography")["Recoveries"].shift(lag)



for case in DAYS_SINCE_CASES:

    df_panel = df_panel.merge(df_panel[df_panel["ConfirmedCases"] >= case].groupby("geography")["Date"].min().reset_index().rename(

        columns={"Date": f"case_{case}_date"}), on="geography", how="left")
## function for preparing features

def prepare_features(df, gap):



    df["perc_1_ac"] = np.around((df[f"lag_{gap}_cc"] - df[f"lag_{gap}_ft"] - df[f"lag_{gap}_rc"]) / df[f"lag_{gap}_cc"], 4)

    df["perc_1_cc"] = np.around(df[f"lag_{gap}_cc"] / df.population, 4)



    for i in range(1, 4):

        df[f"diff_{i}_cc"] = df[f"lag_{gap + i - 1}_cc"] - df[f"lag_{gap + i}_cc"]

        df[f"diff_{i}_ft"] = df[f"lag_{gap + i - 1}_ft"] - df[f"lag_{gap + i}_ft"]



    df["diff_123_cc"] = np.around((df[f"lag_{gap}_cc"] - df[f"lag_{gap + 3}_cc"]) / 3, 4)

    df["diff_123_ft"] = np.around((df[f"lag_{gap}_ft"] - df[f"lag_{gap + 3}_ft"]) / 3, 4)



    df["diff_change_1_cc"] = np.around(df["diff_1_cc"] / df["diff_2_cc"], 4)

    df["diff_change_2_cc"] = np.around(df["diff_2_cc"] / df["diff_3_cc"], 4)

    

    df["diff_change_1_ft"] = np.around(df["diff_1_ft"] / df["diff_2_ft"], 4)

    df["diff_change_2_ft"] = np.around(df["diff_2_ft"] / df["diff_3_ft"], 4)



    df["diff_change_12_cc"] = np.around((df["diff_change_1_cc"] + df["diff_change_2_cc"]) / 2, 4)

    df["diff_change_12_ft"] = np.around((df["diff_change_1_ft"] + df["diff_change_2_ft"]) / 2, 4)



    for i in range(1, 4):

        df[f"change_{i}_cc"] = df[f"lag_{gap + i - 1}_cc"] / df[f"lag_{gap + i}_cc"]

        df[f"change_{i}_ft"] = df[f"lag_{gap + i - 1}_ft"] / df[f"lag_{gap + i}_ft"]



    df["change_123_cc"] = np.around(df[f"lag_{gap}_cc"] / df[f"lag_{gap + 3}_cc"], 4)

    df["change_123_ft"] = np.around(df[f"lag_{gap}_ft"] / df[f"lag_{gap + 3}_ft"], 4)



    for case in DAYS_SINCE_CASES:

        df[f"days_since_{case}_case"] = (df["Date"] - df[f"case_{case}_date"]) / np.timedelta64(1, "D")

        df.loc[df[f"days_since_{case}_case"] < gap, f"days_since_{case}_case"] = np.nan



    df["country_flag"] = df["Province_State"].isna().astype(np.int64)

    df["density"] = np.around(df["population"] / df["area"], 4)



    # target variable is log of change from last known value

    df["target_cc"] = np.log1p(df["ConfirmedCases"]) - np.log1p(df[f"lag_{gap}_cc"])

    df["target_ft"] = np.log1p(df["Fatalities"]) - np.log1p(df[f"lag_{gap}_ft"])



    features = [f"lag_{gap}_cc", f"lag_{gap}_ft", f"lag_{gap}_rc",

                "perc_1_ac", "perc_1_cc",

                "diff_1_cc", "diff_2_cc", "diff_3_cc",

                "diff_1_ft", "diff_2_ft", "diff_3_ft",

                "diff_123_cc", "diff_123_ft",

                "diff_change_1_cc", "diff_change_2_cc",

                "diff_change_1_ft", "diff_change_2_ft",

                "diff_change_12_cc", "diff_change_12_ft",

                "change_1_cc", "change_2_cc", "change_3_cc",

                "change_1_ft", "change_2_ft", "change_3_ft",

                "change_123_cc", "change_123_ft",

                "days_since_1_case",

                "days_since_10_case",

                "days_since_50_case",

                "days_since_100_case",

                "days_since_500_case",

                "days_since_1000_case",

                "days_since_5000_case",

                "days_since_10000_case",

                "country_flag",

                "lat",

                "lon",

                "continent",

                "population",

                "area",

                "density",

                "target_cc",

                "target_ft"]



    return df[features]
## function for building and predicting using LGBM model

def build_predict_lgbm(df_train, df_test, gap):



    df_train.dropna(subset=["target_cc", "target_ft", f"lag_{gap}_cc", f"lag_{gap}_ft"], inplace=True)



    target_cc = df_train["target_cc"]

    target_ft = df_train["target_ft"]



    test_lag_cc = df_test[f"lag_{gap}_cc"].values

    test_lag_ft = df_test[f"lag_{gap}_ft"].values

    

    df_train.drop(["target_cc", "target_ft"], axis=1, inplace=True)

    df_test.drop(["target_cc", "target_ft"], axis=1, inplace=True)



    categorical_features = ["continent"]



    dtrain_cc = lgb.Dataset(df_train, label=target_cc, categorical_feature=categorical_features)

    dtrain_ft = lgb.Dataset(df_train, label=target_ft, categorical_feature=categorical_features)



    model_cc = lgb.train(LGB_PARAMS, train_set=dtrain_cc, num_boost_round=200)

    model_ft = lgb.train(LGB_PARAMS, train_set=dtrain_ft, num_boost_round=200)



    # inverse transform from log of change from last known value

    y_pred_cc = np.expm1(model_cc.predict(df_test, num_boost_round=200) + np.log1p(test_lag_cc))

    y_pred_ft = np.expm1(model_ft.predict(df_test, num_boost_round=200) + np.log1p(test_lag_ft))



    return y_pred_cc, y_pred_ft, model_cc, model_ft
## function for predicting moving average decay model

def predict_mad(df_test, gap, val=False):



    df_test["avg_diff_cc"] = np.around((df_test[f"lag_{gap}_cc"] - df_test[f"lag_{gap + 3}_cc"]) / 3, 4)

    df_test["avg_diff_ft"] = np.around((df_test[f"lag_{gap}_ft"] - df_test[f"lag_{gap + 3}_ft"]) / 3, 4)



    if val:

        y_pred_cc = df_test[f"lag_{gap}_cc"] + gap * df_test["avg_diff_cc"] - (1 - MAD_FACTOR) * df_test["avg_diff_cc"] * np.sum(list(range(gap))) / VAL_DAYS

        y_pred_ft = df_test[f"lag_{gap}_ft"] + gap * df_test["avg_diff_ft"] - (1 - MAD_FACTOR) * df_test["avg_diff_ft"] * np.sum(list(range(gap))) / VAL_DAYS

    else:

        y_pred_cc = df_test[f"lag_{gap}_cc"] + gap * df_test["avg_diff_cc"] - (1 - MAD_FACTOR) * df_test["avg_diff_cc"] * np.sum(list(range(gap))) / n_dates_test

        y_pred_ft = df_test[f"lag_{gap}_ft"] + gap * df_test["avg_diff_ft"] - (1 - MAD_FACTOR) * df_test["avg_diff_ft"] * np.sum(list(range(gap))) / n_dates_test



    return y_pred_cc, y_pred_ft
df_test_full = df_panel[~df_panel["ForecastId"].isna()].reset_index(drop=True)

df_train = df_panel[df_panel["ForecastId"].isna()].reset_index(drop=True)



max_date_train = pd.Timestamp("2020-04-15")



print(f"Test shape: {df_test_full.shape}")

print(f"Train shape: {df_train.shape}")
for date in df_test_full["Date"].unique():

    print("Processing date:", date)



    if date in df_train["Date"].values:

        print("already exists")

    else:

        df_test = df_test_full[df_test_full["Date"] == date]



        gap = (pd.Timestamp(date) - max_date_train).days

        print(pd.Timestamp(date).strftime("%Y-%m-%d"), max_date_train.strftime("%Y-%m-%d"), gap)

        break
max_date_train - pd.Timedelta(VAL_DAYS, "D") + pd.Timedelta(gap, "D")
## building lag x-days models

df_train = df_panel[df_panel["Date"] <= max_date_train]

df_test_full = df_panel[(df_panel["Date"] > max_date_train) & (~df_panel["ForecastId"].isna())]



df_preds_val = []

df_preds_test = []



for date in df_test_full["Date"].unique():



    print("Processing date:", date)

    

    # ignore date already present in train data

    if date in df_train["Date"].values:

        df_pred_test = df_test_full.loc[df_test_full["Date"] == date, ["ForecastId", "ConfirmedCases", "Fatalities"]].rename(

            columns={"ConfirmedCases": "ConfirmedCases_test", "Fatalities": "Fatalities_test"})



    else:

        df_test = df_test_full[df_test_full["Date"] == date]

        gap = (pd.Timestamp(date) - max_date_train).days



        if gap <= VAL_DAYS:

            val_date = max_date_train - pd.Timedelta(VAL_DAYS, "D") + pd.Timedelta(gap, "D")



            df_build = df_train[df_train["Date"] < val_date]

            df_val = df_train[df_train["Date"] == val_date]



            X_build = prepare_features(df_build, gap)

            X_val = prepare_features(df_val, gap)



            y_val_cc_lgb, y_val_ft_lgb, _, _ = build_predict_lgbm(X_build, X_val, gap)

            y_val_cc_mad, y_val_ft_mad = predict_mad(df_val, gap, val = True)



            df_pred_val = pd.DataFrame({"Id": df_val["Id"].values,

                                        "ConfirmedCases_val_lgb": y_val_cc_lgb,

                                        "Fatalities_val_lgb": y_val_ft_lgb,

                                        "ConfirmedCases_val_mad": y_val_cc_mad,

                                        "Fatalities_val_mad": y_val_ft_mad})



            df_preds_val.append(df_pred_val)



        X_train = prepare_features(df_train, gap)

        X_test = prepare_features(df_test, gap)



        y_test_cc_lgb, y_test_ft_lgb, model_cc, model_ft = build_predict_lgbm(X_train, X_test, gap)

        y_test_cc_mad, y_test_ft_mad = predict_mad(df_test, gap)

        

        if gap == 1:

            model_1_cc = model_cc

            model_1_ft = model_ft

            features_1 = X_train.columns.values

        elif gap == 14:

            model_14_cc = model_cc

            model_14_ft = model_ft

            features_14 = X_train.columns.values

        elif gap == 28:

            model_28_cc = model_cc

            model_28_ft = model_ft

            features_28 = X_train.columns.values



        df_pred_test = pd.DataFrame({"ForecastId": df_test.ForecastId.values,

                                     "ConfirmedCases_test_lgb": y_test_cc_lgb,

                                     "Fatalities_test_lgb": y_test_ft_lgb,

                                     "ConfirmedCases_test_mad": y_test_cc_mad,

                                     "Fatalities_test_mad": y_test_ft_mad})



    df_preds_test.append(df_pred_test)
## validation score

df_panel = df_panel.merge(pd.concat(df_preds_val, sort=False), on="Id", how="left")

df_panel = df_panel.merge(pd.concat(df_preds_test, sort=False), on="ForecastId", how="left")



rmsle_cc_lgb = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel["ConfirmedCases_val_lgb"].isna()]["ConfirmedCases"]),

                                          np.log1p(df_panel[~df_panel["ConfirmedCases_val_lgb"].isna()]["ConfirmedCases_val_lgb"])))



rmsle_ft_lgb = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel["Fatalities_val_lgb"].isna()]["Fatalities"]),

                                          np.log1p(df_panel[~df_panel["Fatalities_val_lgb"].isna()]["Fatalities_val_lgb"])))



rmsle_cc_mad = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel["ConfirmedCases_val_mad"].isna()]["ConfirmedCases"]),

                                          np.log1p(df_panel[~df_panel["ConfirmedCases_val_mad"].isna()]["ConfirmedCases_val_mad"])))



rmsle_ft_mad = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel["Fatalities_val_mad"].isna()]["Fatalities"]),

                                          np.log1p(df_panel[~df_panel["Fatalities_val_mad"].isna()]["Fatalities_val_mad"])))





print("LGB CC RMSLE Val of", VAL_DAYS, "days for CC:", round(rmsle_cc_lgb, 2))

print("LGB FT RMSLE Val of", VAL_DAYS, "days for FT:", round(rmsle_ft_lgb, 2))

print("LGB Overall RMSLE Val of", VAL_DAYS, "days:", round((rmsle_cc_lgb + rmsle_ft_lgb) / 2, 2))

print("\n")

print("MAD CC RMSLE Val of", VAL_DAYS, "days for CC:", round(rmsle_cc_mad, 2))

print("MAD FT RMSLE Val of", VAL_DAYS, "days for FT:", round(rmsle_ft_mad, 2))

print("MAD Overall RMSLE Val of", VAL_DAYS, "days:", round((rmsle_cc_mad + rmsle_ft_mad) / 2, 2))
# feature importance

from bokeh.io import output_notebook, show

from bokeh.layouts import column

from bokeh.palettes import Spectral3

from bokeh.plotting import figure



output_notebook()



df_fimp_1_cc = pd.DataFrame({"feature": features_1, "importance": model_1_cc.feature_importance(), "model": "m01"})

df_fimp_14_cc = pd.DataFrame({"feature": features_14, "importance": model_14_cc.feature_importance(), "model": "m14"})

df_fimp_28_cc = pd.DataFrame({"feature": features_28, "importance": model_28_cc.feature_importance(), "model": "m28"})



df_fimp_1_cc.sort_values("importance", ascending=False, inplace=True)

df_fimp_14_cc.sort_values("importance", ascending=False, inplace=True)

df_fimp_28_cc.sort_values("importance", ascending=False, inplace=True)



v1 = figure(plot_width=800, plot_height=400, x_range=df_fimp_1_cc["feature"], title="Feature Importance of LGB Model 1")

v1.vbar(x=df_fimp_1_cc["feature"], top=df_fimp_1_cc["importance"], width=1)

v1.xaxis.major_label_orientation = 1.3



v14 = figure(plot_width=800, plot_height=400, x_range=df_fimp_14_cc["feature"], title="Feature Importance of LGB Model 14")

v14.vbar(x=df_fimp_14_cc["feature"], top=df_fimp_14_cc["importance"], width=1)

v14.xaxis.major_label_orientation = 1.3



v28 = figure(plot_width=800, plot_height=400, x_range=df_fimp_28_cc["feature"], title="Feature Importance of LGB Model 28")

v28.vbar(x=df_fimp_28_cc["feature"], top=df_fimp_28_cc["importance"], width=1)

v28.xaxis.major_label_orientation = 1.3



v = column(v1, v14, v28)



show(v)
## visualizing ConfirmedCases

from bokeh.models import Panel, Tabs

from bokeh.io import output_notebook, show

from bokeh.plotting import figure



output_notebook()



tab_list = []

for geography in df_panel["geography"].unique():

    df_geography = df_panel[df_panel["geography"] == geography]

    v = figure(plot_width=800, plot_height=400, x_axis_type="datetime", title="Covid-19 ConfirmedCases over time")



    v.line(df_geography["Date"], df_geography["ConfirmedCases"], color="green", legend_label="CC (Train)")

    v.line(df_geography["Date"], df_geography["ConfirmedCases_val_lgb"], color="blue", legend_label="CC LGB (Val)")

    v.line(df_geography["Date"], df_geography["ConfirmedCases_val_mad"], color="purple", legend_label="CC MAD (Val)")



    v.line(df_geography["Date"][df_geography["Date"] > max_date_train], df_geography["ConfirmedCases_test_lgb"][df_geography["Date"] > max_date_train],

           color="red", legend_label="CC LGB (Test)")



    v.line(df_geography["Date"][df_geography["Date"] > max_date_train], df_geography["ConfirmedCases_test_mad"][df_geography["Date"] > max_date_train],

           color="orange", legend_label="CC MAD (Test)")



    v.legend.location = "top_left"

    tab = Panel(child=v, title=geography)

    tab_list.append(tab)



tabs = Tabs(tabs=tab_list)

show(tabs)
## visualizing Fatalities

from bokeh.models import Panel, Tabs

from bokeh.io import output_notebook, show

from bokeh.plotting import figure



output_notebook()



tab_list = []

for geography in df_panel["geography"].unique():

    df_geography = df_panel[df_panel["geography"] == geography]

    v = figure(plot_width=800, plot_height=400, x_axis_type="datetime", title="Covid-19 Fatalities over time")



    v.line(df_geography["Date"], df_geography["Fatalities"], color="green", legend_label="FT (Train)")

    v.line(df_geography["Date"], df_geography["Fatalities_val_lgb"], color="blue", legend_label="FT LGB (Val)")

    v.line(df_geography["Date"], df_geography["Fatalities_val_mad"], color="purple", legend_label="FT MAD (Val)")



    v.line(df_geography["Date"][df_geography["Date"] > max_date_train], df_geography["Fatalities_test_lgb"][df_geography["Date"] > max_date_train],

           color="red", legend_label="FT LGB (Test)")



    v.line(df_geography["Date"][df_geography["Date"] > max_date_train], df_geography["Fatalities_test_mad"][df_geography["Date"] > max_date_train],

           color="orange", legend_label="FT MAD (Test)")



    v.legend.location = "top_left"

    tab = Panel(child=v, title=geography)

    tab_list.append(tab)



tabs = Tabs(tabs=tab_list)

show(tabs)
## preparing submission file

df_test = df_panel.loc[~df_panel["ForecastId"].isna(), ["ForecastId", "Country_Region", "Province_State", "Date",

                                                        "ConfirmedCases_test_lgb", "ConfirmedCases_test_mad",

                                                        "Fatalities_test_lgb", "Fatalities_test_mad"]].reset_index(drop=True)



df_test["ConfirmedCases"] = 0.15 * df_test["ConfirmedCases_test_lgb"] + 0.85 * df_test["ConfirmedCases_test_mad"]

df_test["Fatalities"] = 0.15 * df_test["Fatalities_test_lgb"] + 0.85 * df_test["Fatalities_test_mad"]



df_submission = df_test[["ForecastId", "ConfirmedCases", "Fatalities"]]

df_submission["ForecastId"] = df_submission["ForecastId"].astype(np.int64)



## writing final submission and complete output

df_submission.to_csv(PATH_SUBMISSION, index=False)

df_test.to_csv(PATH_OUTPUT, index=False)



df_submission.head()
## function for predicting moving average decay model

def predict_mad(df_test, gap, MAD_FACTOR, val=False):



    df_test["avg_diff_cc"] = np.around((df_test[f"lag_{gap}_cc"] - df_test[f"lag_{gap + 3}_cc"]) / 3, 4)

    df_test["avg_diff_ft"] = np.around((df_test[f"lag_{gap}_ft"] - df_test[f"lag_{gap + 3}_ft"]) / 3, 4)



    if val:

        y_pred_cc = df_test[f"lag_{gap}_cc"] + gap * df_test["avg_diff_cc"] - (1 - MAD_FACTOR) * df_test["avg_diff_cc"] * np.sum(list(range(gap))) / VAL_DAYS

        y_pred_ft = df_test[f"lag_{gap}_ft"] + gap * df_test["avg_diff_ft"] - (1 - MAD_FACTOR) * df_test["avg_diff_ft"] * np.sum(list(range(gap))) / VAL_DAYS

    else:

        y_pred_cc = df_test[f"lag_{gap}_cc"] + gap * df_test["avg_diff_cc"] - (1 - MAD_FACTOR) * df_test["avg_diff_cc"] * np.sum(list(range(gap))) / n_dates_test

        y_pred_ft = df_test[f"lag_{gap}_ft"] + gap * df_test["avg_diff_ft"] - (1 - MAD_FACTOR) * df_test["avg_diff_ft"] * np.sum(list(range(gap))) / n_dates_test



    return y_pred_cc, y_pred_ft
import time



def get_script_time(script_time):

    hours = script_time // 3600

    minutes = (script_time % 3600) // 60

    seconds = script_time % 60

    result = (f"{hours}h " if hours > 0 else "") + (f"{minutes}m " if minutes > 0 else "") + f"{seconds}s"

    return result



t_start = time.time()



gap = 13

mad_factors = np.arange(0, 10.01, .01)

scores_mad_df = pd.DataFrame(data={})



for mad_factor in mad_factors:

    predicts_cc_mad, predicts_ft_mad = predict_mad(df_test_full, gap, mad_factor, val=True)

    rmsle_cc_mad = np.around(np.sqrt(mean_squared_error(np.log1p(df_test_full["ConfirmedCases"].values), np.log1p(predicts_cc_mad))), 5)

    rmsle_ft_mad = np.around(np.sqrt(mean_squared_error(np.log1p(df_test_full["Fatalities"].values), np.log1p(predicts_ft_mad))), 5)

    scores_mad_df = pd.concat([scores_mad_df, pd.DataFrame(data={"mad_factor": mad_factor,

                                                                 "rmsle_cc_mad": rmsle_cc_mad,

                                                                 "rmsle_ft_mad": rmsle_ft_mad}, index=[0])], axis=0, ignore_index=True)



scores_mad_df["mean_rmsle"] = scores_mad_df[["rmsle_cc_mad", "rmsle_ft_mad"]].mean(axis=1).round(5)

scores_mad_df = scores_mad_df.sort_values(by="mean_rmsle", ascending=True).reset_index(drop=True)



print(f"Done in {get_script_time(int(time.time() - t_start))}")

scores_mad_df.head()
top_countries = df_test_full[df_test_full["geography"].apply(lambda x: not x.startswith("China"))].groupby("geography")["population"].max().reset_index().sort_values(

    by="population", ascending=False).reset_index(drop=True)

top_countries = top_countries.head(20)["geography"].tolist()



top_countries
t_start = time.time()



scores_by_country_mad_df = pd.DataFrame(data={})



for country in top_countries:

    for mad_factor in mad_factors:

        predicts_cc_mad, predicts_ft_mad = predict_mad(df_test_full[df_test_full["geography"] == country], gap, mad_factor, val=True)

        rmsle_cc_mad = np.around(np.sqrt(mean_squared_error(np.log1p(df_test_full[df_test_full["geography"] == country]["ConfirmedCases"].fillna(-1)), np.log1p(predicts_cc_mad))), 5)

        rmsle_ft_mad = np.around(np.sqrt(mean_squared_error(np.log1p(df_test_full[df_test_full["geography"] == country]["Fatalities"].fillna(-1)), np.log1p(predicts_ft_mad))), 5)

        scores_by_country_mad_df = pd.concat([scores_by_country_mad_df, pd.DataFrame(data={"country": country,

                                                                                           "mad_factor": mad_factor,

                                                                                           "rmsle_cc_mad": rmsle_cc_mad,

                                                                                           "rmsle_ft_mad": rmsle_ft_mad}, index=[0])], axis=0, ignore_index=True)



scores_by_country_mad_df["mean_rmsle"] = scores_by_country_mad_df[["rmsle_cc_mad", "rmsle_ft_mad"]].mean(axis=1).round(5)

scores_by_country_mad_df = scores_by_country_mad_df.sort_values(by=["country", "mean_rmsle"], ascending=[False, True]).reset_index(drop=True)

scores_by_country_mad_df = scores_by_country_mad_df.drop_duplicates(subset=["country"], keep="first").sort_values(by="mean_rmsle", ascending=True).reset_index(drop=True)



print(f"Done in {get_script_time(int(time.time() - t_start))}")

scores_by_country_mad_df
import lightgbm

import shap



def get_shap_values(estimator, X, plot_summary=True):

    explainer = shap.TreeExplainer(estimator)

    shap_values = explainer.shap_values(X)



    if plot_summary:

        shap.summary_plot(shap_values, X, plot_type="bar")

        shap.summary_plot(shap_values, X)



    df_shap = pd.DataFrame(abs(shap_values).mean(axis=0), X.columns.values).reset_index()

    df_shap.sort_values(by=0, ascending=False, inplace=True)

    df_shap.columns = ["feature", "shap_value"]

    return df_shap





for gap in [1, 14, 28]:

    train_data = prepare_features(df_train, gap)



    estimator = lightgbm.LGBMRegressor(**LGB_PARAMS)

    dont_use_cols = [col for col in df_train.columns if "target" in col]

    estimator.fit(train_data.drop(dont_use_cols, axis=1), train_data["target_cc"])



    print(f"Gap: {gap}", "\n")

    get_shap_values(estimator, train_data.drop(dont_use_cols, axis=1), plot_summary=True)

    print("\n" + "".join(["=" for _ in range(200)]) + "\n")
data = prepare_features(df_panel.drop(df_panel.columns.tolist()[-8:], axis=1), gap=1)



data = pd.concat([df_panel[["Id", "geography", "Date", "ConfirmedCases", "Fatalities", "Recoveries"]], data], axis=1)



need_countries = data[(data["Date"] == pd.to_datetime("2020-05-01")) & (data["ConfirmedCases"] > 1000)]["geography"].unique()



data = data[data["geography"].isin(need_countries)].reset_index(drop=True)



# Кол-во выявленных случаев за день

data["cc_by_day"] = 0

data.loc[data["geography"] == data["geography"].shift(), "cc_by_day"] = data["ConfirmedCases"] - data["ConfirmedCases"].shift()



# Максимум выявленных случаев в день по стране

data["cc_by_day_max"] = data.groupby(["geography"])["cc_by_day"].transform(np.max)



# Отношение кол-ва выявленных случаев в текущий день к максимуму выявленных случаев

data["cc_by_day_ratio_to_max"] = np.around(data["cc_by_day"] / data["cc_by_day_max"], 4)

data.loc[data["cc_by_day_max"] == 0, "cc_by_day_ratio_to_max"] = 0



# Среднее кол-во выявленных случаев за последние 3 дня (не включая сегодня)

data["cc_by_day_mean_in_last_3d"] = data.groupby(["geography"])["cc_by_day"].apply(lambda x: x.rolling(3, min_periods=1).mean().round(0)).shift()



# Среднее кол-во выявленных случаев за следующие 3 дня (не включая сегодня)

data = data.sort_values(by=["geography", "Date"], ascending=[True, False]).reset_index(drop=True)

data["cc_by_day_mean_in_next_3d"] = data.groupby(["geography"])["cc_by_day"].apply(lambda x: x.rolling(3, min_periods=1).mean().round(0)).shift()



# Отношение кол-ва выявленных случаев сегодня к среднему кол-ву выявленных случаев за 3 дня до этого

data = data.sort_values(by=["geography", "Date"], ascending=[True, True]).reset_index(drop=True)

data["cc_by_day_dynamic_to_past"] = np.around(data["cc_by_day"] / data["cc_by_day_mean_in_last_3d"], 4)

data.loc[data["cc_by_day_dynamic_to_past"] == np.inf, "cc_by_day_dynamic_to_past"] = 1



# Отношение кол-ва выявленных случаев сегодня к среднему кол-ву выявленных случаев за следующие 3 дня

data["cc_by_day_dynamic_to_future"] = np.around(data["cc_by_day"] / data["cc_by_day_mean_in_next_3d"], 4)

data.loc[data["cc_by_day_dynamic_to_future"] == np.inf, "cc_by_day_dynamic_to_future"] = 1



data["cc_by_day_max_flag"] = (data["cc_by_day_ratio_to_max"] > .95).astype(np.int64)

data["cluster"] = data.groupby(["geography"])["cc_by_day_max_flag"].cumsum()



data["cluster"] = data["cluster"].apply(lambda x: "До пика" if x == 0 else "")

data.loc[(data["cluster"] == "") & (data["cc_by_day_max_flag"] == 1), "cluster"] = "Пик"

data.loc[data["cluster"] == "", "cluster"] = "После пика"



data.head()
nrows, ncols = 2, 4



fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=[8 * ncols, 6 * nrows])



for idx, feature in enumerate(["diff_1_cc", "diff_123_cc", "diff_change_1_cc", "diff_change_12_cc", "change_1_cc", "change_123_cc", "perc_1_ac", "perc_1_cc"]):

    i, j = int(idx / ncols), int(idx % ncols)

    quan1 = np.quantile(data[feature].fillna(1).replace(np.inf, 1), .025)

    quan2 = np.quantile(data[feature].fillna(1).replace(np.inf, 1), .975)

    values = data[(data[feature] > quan1) & (data[feature] < quan2)][feature].apply(lambda x: 1 if not np.isfinite(x) or np.isnan(x) else x)

    sns.distplot(values, bins=30, kde=False, ax=axes[i, j])
data[data["Date"] == pd.to_datetime("2020-05-01")]["cluster"].value_counts(dropna=False).reset_index()
cluster_df = data[data["Date"] == pd.to_datetime("2020-05-01")][["geography", "cluster"]].reset_index(drop=True)



cluster_df
## visualizing ConfirmedCases by day

from bokeh.models import Panel, Tabs

from bokeh.io import output_notebook, show

from bokeh.plotting import figure



output_notebook()



tab_list = []

for geography in data["geography"].unique():

    df_geography = data[data["geography"] == geography]

    v = figure(plot_width=800, plot_height=400, x_axis_type="datetime", title="Covid-19 ConfirmedCases by day")



    cluster = df_geography[data["Date"] == pd.to_datetime("2020-05-01")]["cluster"].values[0]

    v.line(df_geography["Date"], df_geography["cc_by_day"], color="green", legend_label=cluster)



    v.legend.location = "top_left"

    tab = Panel(child=v, title=geography)

    tab_list.append(tab)



tabs = Tabs(tabs=tab_list)

show(tabs)