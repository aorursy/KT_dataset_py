import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option("max_columns", 500)

pd.set_option("max_rows", 500)

from fbprophet import Prophet

import matplotlib.pyplot as plt

import math as math



%matplotlib inline
# Load the data

train = pd.read_csv("../input/web-traffic-time-series-forecasting/train_1.csv.zip", compression="zip")

keys = pd.read_csv("../input/web-traffic-time-series-forecasting/key_1.csv.zip", compression="zip")

ss = pd.read_csv("../input/web-traffic-time-series-forecasting/sample_submission_1.csv.zip", compression="zip")
train.head()
# Check the data

print("Check the number of records")

print("Number of records:", train.shape[0], "\n")



print("Null analysis")

empty_sample = train[train.isnull().any(axis=1)]

print("Number of records contain 1+ null:", empty_sample.shape[0], "\n")
empty_sample.iloc[np.r_[0:10, len(empty_sample)-10:len(empty_sample)]]
# plot 3 the time series

def plot_time_series(df, row_num, start_col =1, ax=None):

    if ax is None:

            fig = plt.figure(facecolor="w", figsize=[10, 6])

            ax = fig.add_subplot(111)

    else:

        fig = ax.get_figure()



    series_title = df.iloc[row_num, 0]

    sample_series = df.iloc[row_num, start_col:]

    sample_series.plot(style=".", ax=ax)

    ax.set_title("Series: %s" % series_title)



fig, axs  = plt.subplots(4,1,figsize=(12,12))

plot_time_series(empty_sample, 1, ax=axs[0])

plot_time_series(empty_sample, 10, ax=axs[1])

plot_time_series(empty_sample, 100, ax=axs[2])

plot_time_series(empty_sample, 1005, ax=axs[3])



plt.tight_layout()
# series with all NaN

empty_sample.iloc[1000:1010]
import re



def breakdown_topic(str):

    m = re.search("(.*)\_(.*).wikipedia.org\_(.*)\_(.*)", str)

    if m is not None:

        return m.group(1), m.group(2), m.group(3), m.group(4)

    else:

        return "", "", "", ""



print(breakdown_topic("Рудова,_Наталья_Александровна_ru.wikipedia.org_all-access_spider"))

print(breakdown_topic("台灣災難列表_zh.wikipedia.org_all-access_spider"))

print(breakdown_topic("File:Memphis_Blues_Tour_2010.jpg_commons.wikimedia.org_mobile-web_all-agents"))
page_details = train.Page.str.extract(r"(?P<topic>.*)\_(?P<lang>.*).wikipedia.org\_(?P<access>.*)\_(?P<type>.*)")



page_details[0:10]
unique_topic = page_details["topic"].unique()

print(unique_topic)

print("Number of distinct topics:", unique_topic.shape[0])
fig, axs = plt.subplots(3, 1, figsize=[12, 12])



page_details["lang"].value_counts().sort_index().plot.bar(ax=axs[0])

axs[0].set_title("Language - distribution")



page_details["access"].value_counts().sort_index().plot.bar(ax=axs[1])

axs[1].set_title("Access - distribution")



page_details["type"].value_counts().sort_index().plot.bar(ax=axs[2])

axs[2].set_title("Type - distribution")



plt.tight_layout()
# Generate train and validate dataset

train_df = pd.concat([page_details, train], axis=1)



def get_train_validate_set(train_df, test_percent):

    train_end = math.floor((train_df.shape[1] - 5) * (1 - test_percent))

    train_ds = train_df.iloc[:, np.r_[0, 1, 2, 3, 4, 5:train_end]]

    test_ds = train_df.iloc[:, np.r_[0, 1, 2, 3, 4, train_end:train_df.shape[1]]]



    return train_ds, test_ds



X_train, y_train = get_train_validate_set(train_df, 0.1)



print("The training set sample:")

print(X_train[0:10])

print("The validation set sample:")

print(y_train[0:10])
def extract_series(df, row_num, start_idx):

    y = df.iloc[row_num, start_idx:]

    df = pd.DataFrame({"ds": y.index, "y": y.values})

    return df
def smape(predict, actual, debug=False):

    """

    predict and actual is a panda series.

    In this implementation I will skip all the datapoint with actual is null

    """

    actual = actual.fillna(0)

    data = pd.concat([predict, actual], axis=1, keys=["predict", "actual"])

    data = data[data["actual"].notnull()]

    if debug:

        print("debug", data)



    evals = abs(data["predict"] - data["actual"]) * 1.0 / (abs(data["predict"]) + abs(data["actual"])) * 2

    evals[evals.isnull()] = 0

    #print(np.sum(evals), len(data), np.sum(evals) * 1.0 / len(data))



    result = np.sum(evals) / len(data) * 100.0



    return result



# create testing series

testing_series_1 = X_train.iloc[0, 5:494]

testing_series_2 = X_train.iloc[0, 5:494].shift(-1)

testing_series_3 = X_train.iloc[1, 5:494]

testing_series_4 = pd.Series([0, 0, 0, 0])
testing_series_1
np.repeat(3, 500)
random_series_1 = pd.Series(np.repeat(3, 500))

random_series_2 = pd.Series(np.random.normal(3, 1, 500))

random_series_3 = pd.Series(np.random.normal(500, 20, 500))

random_series_4 = pd.Series(np.repeat(500, 500))



# testing 1 same series

print("\nSMAPE score to predict a constant array of 3")

print("Score (same series): %.3f" % smape(random_series_1, random_series_1))

print("Score (same series - 1) %.3f" % smape(random_series_1, random_series_1 - 1))

print("Score (same series + 1) %.3f" % smape(random_series_1, random_series_1 + 1))



# testing 2 same series shift by one

print("\nSMAPE score to predict a array of normal distribution around 3")

print("Score (random vs mean) %.3f" % smape(random_series_2, random_series_1))

print("Score (random vs mean-1) %.3f" % smape(random_series_2, random_series_2 - 1))

print("Score (random vs mean+1) %.3f" % smape(random_series_2, random_series_2 + 1))

print("Score (random vs mean*0.9) %.3f" % smape(random_series_2, random_series_2 * 0.9))

print("Score (random vs mean*1.1) %.3f" % smape(random_series_2, random_series_2 * 1.1))



# testing 3 totally different series

print("\nSMAPE score to predict a array of normal distribution around 500")

print("Score (random vs mean) %.3f" % smape(random_series_3, random_series_4))

print("Score (random vs mean-20) %.3f" % smape(random_series_3, random_series_3 - 20))

print("Score (random vs mean+20) %.3f" % smape(random_series_3, random_series_3 + 20))

print("Score (random vs mean*0.9) %.3f" % smape(random_series_3, random_series_3 * 0.9))

print("Score (random vs mean*1.1) %.3f" % smape(random_series_3, random_series_3 * 1.1))
y_true_1 = pd.Series(np.random.normal(1, 1, 500))

y_true_2 = pd.Series(np.random.normal(2, 1, 500))

y_true_3 = pd.Series(np.random.normal(3, 1, 500))

y_pred = pd.Series(np.ones(500))

x = np.linspace(0,10,1000)

res_1 = list([smape(y_true_1, i * y_pred) for i in x])

res_2 = list([smape(y_true_2, i * y_pred) for i in x])

res_3 = list([smape(y_true_3, i * y_pred) for i in x])

plt.plot(x, res_1, color='b')

plt.plot(x, res_2, color='r')

plt.plot(x, res_3, color='g')

plt.axvline(x=1, color='k')

plt.axvline(x=2, color='k')

plt.axvline(x=3, color='k')
def plot_prediction_and_actual_2(train, forecast, actual, xlim=None, ylim=None, figSize=None, title=None):

    fig, ax  = plt.subplots(1,1,figsize=figSize)

    ax.plot(pd.to_datetime(train.index), train.values, "k.")

    ax.plot(pd.to_datetime(actual.index), actual.values, "r.")

    ax.plot(pd.to_datetime(forecast.index), forecast.values, "b-")

    ax.set_title(title)

    plt.show()
def median_model(df_train, df_actual, p, review=False, figSize=[12, 4]):



    def nanmedian_zero(a):

        return np.nan_to_num(np.nanmedian(a))



    df_train["y"] = df_train["y"].astype(np.float64)

    df_actual["y"] = df_actual["y"].astype(np.float64)

    visits = nanmedian_zero(df_train["y"].values[-p:])

    train_series = df_train["y"]

    train_series.index = df_train["ds"]



    idx = np.arange(p) + np.arange(len(df_train) - p + 1)[:, None]

    b = [row[row >= 0] for row in df_train["y"].values[idx]]

    pre_forecast = pd.Series(np.append(([float("nan")] * (p - 1)), list(map(nanmedian_zero, b))))

    pre_forecast.index = df_train["ds"]



    forecast_series = pd.Series(np.repeat(visits, len(df_actual)))

    forecast_series.index = df_actual["ds"]



    forecast_series = pre_forecast.append(forecast_series)



    actual_series = df_actual["y"]

    actual_series.index = df_actual["ds"]



    if(review):

        plot_prediction_and_actual_2(train_series, forecast_series, actual_series, figSize=figSize, title="Median model")



    return smape(forecast_series, actual_series)
# This is to demo the median model

print(train.iloc[[2]])



df_train = extract_series(X_train, 2, 5)

df_actual = extract_series(y_train, 2, 5)

lang = X_train.iloc[2, 1]

score = median_model(df_train.copy(), df_actual.copy(), 15, review=True)

print("The SMAPE score is : %.5f" % score)
# holiday variable

#holiday_en = ['2015-01-01', '2015-01-19', '2015-04-03', '2015-05-04', '2015-05-25', '2015-07-01', '2015-07-03', '2015-09-07', '2015-11-26', '2015-11-27', '2015-12-25', '2015-12-26', '2015-12-28', '2016-01-01', '2016-01-18', '2016-03-25', '2016-05-02', '2016-05-30', '2016-07-01', '2016-07-04', '2016-09-05', '2016-11-11', '2016-11-24', '2016-12-25', '2016-12-26', '2016-12-27', '2017-01-01', '2017-01-02', '2017-01-16', '2017-04-14', '2017-05-01', '2017-05-29', '2017-07-01', '2017-07-03', '2017-07-04', '2017-09-04', '2017-11-10', '2017-11-23', '2017-12-25', '2017-12-26']



holiday_en_us = ['2015-01-01', '2015-01-19', '2015-05-25', '2015-07-03', '2015-09-07', '2015-11-26', '2015-11-27', '2015-12-25', '2016-01-01', '2016-01-18', '2016-05-30', '2016-07-04', '2016-09-05', '2016-11-11', '2016-11-24', '2016-12-26', '2017-01-01', '2017-01-02', '2017-01-16', '2017-05-29', '2017-07-04', '2017-09-04', '2017-11-10', '2017-11-23', '2017-12-25']

holiday_en_uk = ['2015-01-01', '2015-04-03', '2015-05-04', '2015-05-25', '2015-12-25', '2015-12-26', '2015-12-28', '2016-01-01', '2016-03-25', '2016-05-02', '2016-05-30', '2016-12-26', '2016-12-27', '2017-01-01', '2017-04-14', '2017-05-01', '2017-05-29', '2017-12-25', '2017-12-26']

holiday_en_canada = ['2015-01-01', '2015-07-01', '2015-09-07', '2015-12-25', '2016-01-01', '2016-07-01', '2016-09-05', '2016-12-25', '2017-01-01', '2017-07-01', '2017-07-03', '2017-09-04', '2017-12-25']



holiday_ru_russia = ['2015-01-01', '2015-01-02', '2015-01-05', '2015-01-06', '2015-01-07', '2015-01-08', '2015-01-09', '2015-02-23', '2015-03-09', '2015-05-01', '2015-05-04', '2015-05-09', '2015-05-11', '2015-06-12', '2015-11-04', '2016-01-01', '2016-01-04', '2016-01-05', '2016-01-06', '2016-01-07', '2016-02-22', '2016-02-23', '2016-03-08', '2016-05-01', '2016-05-09', '2016-06-12', '2016-06-13', '2016-11-04', '2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07', '2017-02-23', '2017-02-24', '2017-03-08', '2017-05-01', '2017-05-08', '2017-05-09', '2017-06-12', '2017-11-04', '2017-11-06']

#holiday_es = ['2015-01-01', '2015-01-06', '2015-01-12', '2015-02-02', '2015-03-16', '2015-03-23', '2015-04-02', '2015-04-03', '2015-05-01', '2015-05-18', '2015-06-08', '2015-06-15', '2015-06-29', '2015-07-20', '2015-08-07', '2015-08-17', '2015-09-16', '2015-10-12', '2015-11-01', '2015-11-02', '2015-11-16', '2015-12-06', '2015-12-08', '2015-12-12', '2015-12-25', '2016-01-01', '2016-01-06', '2016-01-11', '2016-02-01', '2016-03-21', '2016-03-24', '2016-03-25', '2016-05-01', '2016-05-09', '2016-05-30', '2016-06-06', '2016-07-04', '2016-07-20', '2016-08-07', '2016-08-15', '2016-09-16', '2016-10-12', '2016-10-17', '2016-11-01', '2016-11-02', '2016-11-07', '2016-11-14', '2016-11-21', '2016-12-06', '2016-12-08', '2016-12-12', '2016-12-25', '2016-12-26', '2017-01-01', '2017-01-02', '2017-01-06', '2017-01-09', '2017-02-06', '2017-03-20', '2017-04-13', '2017-04-14', '2017-05-01', '2017-05-29', '2017-06-19', '2017-06-26', '2017-07-03', '2017-07-20', '2017-08-07', '2017-08-15', '2017-09-16', '2017-10-12', '2017-10-16', '2017-11-01', '2017-11-02', '2017-11-06', '2017-11-13', '2017-11-20', '2017-12-06', '2017-12-08', '2017-12-12', '2017-12-25']



holiday_es_mexico = ['2015-01-01', '2015-02-02', '2015-03-16', '2015-04-02', '2015-04-03', '2015-05-01', '2015-09-16', '2015-10-12', '2015-11-02', '2015-11-16', '2015-12-12', '2015-12-25', '2016-01-01', '2016-02-01', '2016-03-21', '2016-03-24', '2016-03-25', '2016-05-01', '2016-09-16', '2016-10-12', '2016-11-02', '2016-11-21', '2016-12-12', '2016-12-25', '2016-12-26', '2017-01-01', '2017-01-02', '2017-02-06', '2017-03-20', '2017-04-13', '2017-04-14', '2017-05-01', '2017-09-16', '2017-10-12', '2017-11-02', '2017-11-20', '2017-12-12', '2017-12-25']

holiday_es_spain = ['2017-01-01', '2017-01-06', '2017-04-14', '2017-05-01', '2017-08-15', '2017-10-12', '2017-11-01', '2017-12-06', '2017-12-08', '2017-12-25', '2016-01-01', '2016-01-06', '2016-03-25', '2016-05-01', '2016-08-15', '2016-10-12', '2016-11-01', '2016-12-06', '2016-12-08', '2016-12-25', '2015-01-01', '2015-01-06', '2015-04-03', '2015-05-01', '2015-10-12', '2015-11-01', '2015-12-06', '2015-12-08', '2015-12-25']

holiday_es_colombia = ['2015-01-01', '2015-01-12', '2015-03-23', '2015-04-02', '2015-04-03', '2015-05-01', '2015-05-18', '2015-06-08', '2015-06-15', '2015-06-29', '2015-07-20', '2015-08-07', '2015-08-17', '2015-10-12', '2015-11-02', '2015-11-16', '2015-12-08', '2015-12-25', '2016-01-01', '2016-01-11', '2016-03-21', '2016-03-24', '2016-03-25', '2016-05-01', '2016-05-09', '2016-05-30', '2016-06-06', '2016-07-04', '2016-07-20', '2016-08-07', '2016-08-15', '2016-10-17', '2016-11-07', '2016-11-14', '2016-12-08', '2016-12-25', '2017-01-01', '2017-01-09', '2017-03-20', '2017-04-13', '2017-04-14', '2017-05-01', '2017-05-29', '2017-06-19', '2017-06-26', '2017-07-03', '2017-07-20', '2017-08-07', '2017-08-15', '2017-10-16', '2017-11-06', '2017-11-13', '2017-12-08', '2017-12-25']



holiday_fr_france = ['2015-01-01', '2015-04-06', '2015-05-01', '2015-05-08', '2015-05-14', '2015-05-25', '2015-07-14', '2015-08-15', '2015-11-01', '2015-11-11', '2015-12-25', '2016-01-01', '2016-03-28', '2016-05-01', '2016-05-05', '2016-05-08', '2016-05-16', '2016-07-14', '2016-08-15', '2016-11-01', '2016-11-11', '2016-12-25', '2017-01-01', '2017-04-17', '2017-05-01', '2017-05-08', '2017-05-25', '2017-06-05', '2017-07-14', '2017-08-15', '2017-11-01', '2017-11-11', '2017-12-25']

holiday_jp_japan = ['2015-01-01', '2015-01-12', '2015-02-11', '2015-03-21', '2015-04-29', '2015-05-03', '2015-05-04', '2015-05-05', '2015-05-06', '2015-07-20', '2015-09-21', '2015-09-22', '2015-09-23', '2015-10-12', '2015-11-03', '2015-11-23', '2015-12-23', '2016-01-01', '2016-01-11', '2016-02-11', '2016-03-21', '2016-04-29', '2016-05-03', '2016-05-04', '2016-05-05', '2016-07-18', '2016-08-11', '2016-09-19', '2016-09-22', '2016-10-10', '2016-11-03', '2016-11-23', '2016-12-23', '2017-01-01', '2017-01-09', '2017-02-11', '2017-03-20', '2017-04-29', '2017-05-03', '2017-05-04', '2017-05-05', '2017-07-17', '2017-08-11', '2017-09-18', '2017-09-22', '2017-10-09', '2017-11-03', '2017-11-23', '2017-12-23']



#holiday_de = ['2015-01-01', '2015-01-06', '2015-04-03', '2015-04-06', '2015-05-01', '2015-05-14', '2015-05-25', '2015-06-04', '2015-08-01', '2015-08-15', '2015-10-03', '2015-10-26', '2015-11-01', '2015-12-08', '2015-12-25', '2015-12-26', '2016-01-01', '2016-01-06', '2016-03-25', '2016-03-28', '2016-05-01', '2016-05-05', '2016-05-16', '2016-05-26', '2016-08-01', '2016-08-15', '2016-10-03', '2016-10-26', '2016-11-01', '2016-12-08', '2016-12-25', '2016-12-26', '2017-01-01', '2017-01-06', '2017-04-14', '2017-04-17', '2017-05-01', '2017-05-25', '2017-06-05', '2017-06-15', '2017-08-01', '2017-08-15', '2017-10-03', '2017-10-26', '2017-10-31', '2017-11-01', '2017-12-08', '2017-12-25', '2017-12-26']



holiday_de_germany = ['2015-01-01', '2015-04-03', '2015-04-06', '2015-05-01', '2015-05-14', '2015-05-14', '2015-05-25', '2015-10-03', '2015-12-25', '2015-12-26', '2016-01-01', '2016-03-25', '2016-03-28', '2016-05-01', '2016-05-05', '2016-05-16', '2016-10-03', '2016-12-25', '2016-12-26', '2017-01-01', '2017-04-14', '2017-04-17', '2017-05-01', '2017-05-25', '2017-06-05', '2017-10-03', '2017-10-31', '2017-12-25', '2017-12-26']

holiday_de_austria = ['2015-01-01', '2015-01-06', '2015-04-06', '2015-05-01', '2015-05-14', '2015-05-25', '2015-06-04', '2015-08-15', '2015-10-26', '2015-11-01', '2015-12-08', '2015-12-25', '2015-12-26', '2016-01-01', '2016-01-06', '2016-03-28', '2016-05-01', '2016-05-05', '2016-05-16', '2016-05-26', '2016-08-15', '2016-10-26', '2016-11-01', '2016-12-08', '2016-12-25', '2016-12-26', '2017-01-01', '2017-01-06', '2017-04-17', '2017-05-01', '2017-05-25', '2017-06-05', '2017-06-15', '2017-08-15', '2017-10-26', '2017-11-01', '2017-12-08', '2017-12-25', '2017-12-26']

holiday_de_switzerland = ['2015-01-01', '2015-04-03', '2015-05-14', '2015-08-01', '2015-12-25', '2016-01-01', '2016-03-25', '2016-05-05', '2016-08-01', '2016-12-25', '2017-01-01', '2017-04-14', '2017-05-25', '2017-08-01', '2017-12-25']



#holiday_zh = ['2015-01-01', '2015-02-18', '2015-02-19', '2015-02-20', '2015-02-21', '2015-02-22', '2015-02-23', '2015-02-27', '2015-04-03', '2015-04-04', '2015-04-05', '2015-04-06', '2015-04-07', '2015-05-01', '2015-05-25', '2015-06-19', '2015-06-20', '2015-07-01', '2015-09-03', '2015-09-28', '2015-10-01', '2015-10-09', '2015-10-10', '2015-10-21', '2015-12-25', '2015-12-26', '2016-01-01', '2016-02-07', '2016-02-08', '2016-02-09', '2016-02-10', '2016-02-11', '2016-02-12', '2016-02-29', '2016-03-25', '2016-03-26', '2016-03-28', '2016-04-04', '2016-04-05', '2016-05-01', '2016-05-02', '2016-05-14', '2016-06-09', '2016-06-10', '2016-07-01', '2016-09-15', '2016-09-16', '2016-09-28', '2016-10-01', '2016-10-10', '2016-12-25', '2016-12-26', '2016-12-27', '2017-01-01', '2017-01-02', '2017-01-27', '2017-01-28', '2017-01-29', '2017-01-30', '2017-01-31', '2017-02-01', '2017-02-27', '2017-02-28', '2017-04-03', '2017-04-04', '2017-04-14', '2017-04-15', '2017-04-17', '2017-05-01', '2017-05-03', '2017-05-29', '2017-05-30', '2017-07-01', '2017-10-01', '2017-10-02', '2017-10-04', '2017-10-05', '2017-10-09', '2017-10-10', '2017-10-28', '2017-12-25', '2017-12-26']



holiday_zh_hongkong = ['2015-01-01', '2015-02-19', '2015-02-20', '2015-04-03', '2015-04-04', '2015-04-05', '2015-04-06', '2015-04-07', '2015-05-01', '2015-05-25', '2015-06-20', '2015-07-01', '2015-09-03', '2015-09-28', '2015-10-01', '2015-10-21', '2015-12-25', '2015-12-26', '2016-01-01', '2016-02-08', '2016-02-09', '2016-02-10', '2016-03-25', '2016-03-26', '2016-03-28', '2016-04-04', '2016-05-01', '2016-05-02', '2016-05-14', '2016-06-09', '2016-07-01', '2016-09-16', '2016-10-01', '2016-10-10', '2016-12-25', '2016-12-26', '2016-12-27', '2017-01-01', '2017-01-02', '2017-01-28', '2017-01-30', '2017-01-31', '2017-04-04', '2017-04-14', '2017-04-15', '2017-04-17', '2017-05-01', '2017-05-03', '2017-05-30', '2017-07-01', '2017-10-01', '2017-10-02', '2017-10-05', '2017-10-28', '2017-12-25', '2017-12-26']

holiday_zh_taiwan = ['2015-01-01', '2015-02-18', '2015-02-19', '2015-02-20', '2015-02-21', '2015-02-22', '2015-02-23', '2015-02-23', '2015-02-27', '2015-04-03', '2015-04-05', '2015-04-06', '2015-06-19', '2015-06-20', '2015-09-28', '2015-10-09', '2015-10-10', '2016-01-01', '2016-02-07', '2016-02-08', '2016-02-09', '2016-02-10', '2016-02-11', '2016-02-12', '2016-02-29', '2016-04-04', '2016-04-05', '2016-06-09', '2016-06-10', '2016-09-15', '2016-09-16', '2016-09-28', '2016-10-10', '2017-01-01', '2017-01-02', '2017-01-27', '2017-01-28', '2017-01-29', '2017-01-30', '2017-01-31', '2017-02-01', '2017-02-27', '2017-02-28', '2017-04-03', '2017-04-04', '2017-05-01', '2017-05-29', '2017-05-30', '2017-10-04', '2017-10-09', '2017-10-10']



holidays_en_us = pd.DataFrame({

  'holiday': 'US public holiday',

  'ds': pd.to_datetime(holiday_en_us),

  'lower_window': 0,

  'upper_window': 0,

})



holidays_en_uk = pd.DataFrame({

  'holiday': 'UK public holiday',

  'ds': pd.to_datetime(holiday_en_uk),

  'lower_window': 0,

  'upper_window': 0,

})



holidays_en_canada = pd.DataFrame({

  'holiday': 'Canada public holiday',

  'ds': pd.to_datetime(holiday_en_canada),

  'lower_window': 0,

  'upper_window': 0,

})



holidays_en = pd.concat((holidays_en_us, holidays_en_uk, holidays_en_canada))



holidays_ru_russia = pd.DataFrame({

  'holiday': 'Russia public holiday',

  'ds': pd.to_datetime(holiday_ru_russia),

  'lower_window': 0,

  'upper_window': 0,

})



holidays_ru = holidays_ru_russia



holidays_es_mexico = pd.DataFrame({

  'holiday': 'Mexico public holiday',

  'ds': pd.to_datetime(holiday_es_mexico),

  'lower_window': 0,

  'upper_window': 0,

})



holidays_es_spain = pd.DataFrame({

  'holiday': 'Spain public holiday',

  'ds': pd.to_datetime(holiday_es_spain),

  'lower_window': 0,

  'upper_window': 0,

})



holidays_es_colombia = pd.DataFrame({

  'holiday': 'Colombia public holiday',

  'ds': pd.to_datetime(holiday_es_colombia),

  'lower_window': 0,

  'upper_window': 0,

})



holidays_es = pd.concat((holidays_es_mexico, holidays_es_spain, holidays_es_colombia))



holidays_fr_france = pd.DataFrame({

  'holiday': 'France public holiday',

  'ds': pd.to_datetime(holiday_fr_france),

  'lower_window': 0,

  'upper_window': 0,

})



holidays_fr = holidays_fr_france



holidays_jp_japan = pd.DataFrame({

  'holiday': 'Japan public holiday',

  'ds': pd.to_datetime(holiday_jp_japan),

  'lower_window': 0,

  'upper_window': 0,

})



holidays_jp = holidays_jp_japan



holidays_de_germany = pd.DataFrame({

  'holiday': 'Germany public holiday',

  'ds': pd.to_datetime(holiday_de_germany),

  'lower_window': 0,

  'upper_window': 0,

})



holidays_de_austria = pd.DataFrame({

  'holiday': 'Austria public holiday',

  'ds': pd.to_datetime(holiday_de_austria),

  'lower_window': 0,

  'upper_window': 0,

})



holidays_de_switzerland = pd.DataFrame({

  'holiday': 'Switzerland public holiday',

  'ds': pd.to_datetime(holiday_de_switzerland),

  'lower_window': 0,

  'upper_window': 0,

})



holidays_de = pd.concat((holidays_de_germany, holidays_de_austria, holidays_de_switzerland))



holidays_zh_hongkong = pd.DataFrame({

  'holiday': 'HK public holiday',

  'ds': pd.to_datetime(holiday_zh_hongkong),

  'lower_window': 0,

  'upper_window': 0,

})



holidays_zh_taiwan = pd.DataFrame({

  'holiday': 'Taiwan public holiday',

  'ds': pd.to_datetime(holiday_zh_taiwan),

  'lower_window': 0,

  'upper_window': 0,

})



holidays_zh = pd.concat((holidays_zh_hongkong, holidays_zh_taiwan))



holidays_dict = {"en": holidays_en, 

                 "ru": holidays_ru, 

                 "es": holidays_es, 

                 "fr": holidays_fr, 

                 "ja": holidays_jp,

                 "de": holidays_de,

                 "zh": holidays_zh}
def median_holiday_model(df_train, df_actual, p, lang, review=False, figSize=[12, 4]):

    # Split the train and actual set

    df_train["ds"] = pd.to_datetime(df_train["ds"])

    df_actual["ds"] = pd.to_datetime(df_actual["ds"])

    train_series = df_train["y"]

    train_series.index = df_train["ds"]



    if isinstance(lang, float) and math.isnan(lang):

        df_train["holiday"] = df_train["ds"].dt.dayofweek >= 5

        df_actual["holiday"] = df_actual["ds"].dt.dayofweek >= 5

    else:

        df_train["holiday"] = (df_train["ds"].dt.dayofweek >= 5) | df_train["ds"].isin(holidays_dict[lang].ds)

        df_actual["holiday"] = (df_actual["ds"].dt.dayofweek >= 5) | df_actual["ds"].isin(holidays_dict[lang].ds)



    # Combine the train and actual set

    predict_holiday = median_holiday_helper(df_train, df_actual[df_actual["holiday"]], p, True)

    predict_non_holiday = median_holiday_helper(df_train, df_actual[~df_actual["holiday"]], p, False)



    forecast_series = predict_non_holiday.combine_first(predict_holiday)



    actual_series = df_actual["y"]

    actual_series.index = df_actual["ds"]



    if review:

        plot_prediction_and_actual_2(train_series, forecast_series, actual_series, figSize=figSize, title="Median model with holiday")



    return smape(forecast_series, actual_series)





def median_holiday_helper(df_train, df_actual, p, holiday):



    def nanmedian_zero(a):

        return np.nan_to_num(np.nanmedian(a))

    

    df_train["y"] = df_train["y"].astype(np.float64).values

    df_actual["y"] = df_actual["y"].astype(np.float64).values



    sample = df_train[-p:]



    if holiday:

        sample = sample[sample["holiday"]]

    else:

        sample = sample[~sample["holiday"]]



    visits = nanmedian_zero(sample["y"])



    idx = np.arange(p) + np.arange(len(df_train) - p + 1)[:, None]

    b = [row[row >= 0] for row in df_train["y"].values[idx]]

    pre_forecast = pd.Series(np.append(([float("nan")] * (p - 1)), list(map(nanmedian_zero, b))))

    pre_forecast.index = df_train["ds"]



    forecast_series = pd.Series(np.repeat(visits, len(df_actual)))

    forecast_series.index = df_actual["ds"]



    forecast_series = pre_forecast.append(forecast_series)



    return forecast_series
# This is to demo the median model - weekday, weekend and 

print(train.iloc[[2]])



df_train = extract_series(X_train, 2, 5)

df_actual = extract_series(y_train, 2, 5)

lang = X_train.iloc[2, 1]

score = median_holiday_model(df_train.copy(), df_actual.copy(), 15, lang, review=True)

print("The SMAPE score is: %.5f" % score)
from statsmodels.tsa.arima_model import ARIMA   

import warnings



def arima_model(df_train, df_actual, p, d, q, figSize=[12, 4], review=False):

    df_train = df_train.fillna(0)

    train_series = df_train["y"]

    train_series.index = df_train["ds"]



    result = None

    with warnings.catch_warnings():

        warnings.filterwarnings("ignore")

        try:

            arima = ARIMA(train_series, [p, d, q])

            result = arima.fit(disp=False)

        except Exception as e:

            print("\tARIMA failed", e)



    #print(result.params)

    start_idx = df_train["ds"][d]

    end_idx = df_actual["ds"].max()

    forecast_series = result.predict(start_idx, end_idx, typ="levels")



    actual_series = df_actual["y"]

    actual_series.index = pd.to_datetime(df_actual["ds"])



    if review:

        plot_prediction_and_actual_2(train_series, forecast_series, actual_series, figSize=figSize, title="ARIMA model")



    return smape(forecast_series, actual_series)
df_train = df_train.fillna(0)

train_series = df_train["y"]

train_series.index = df_train["ds"]



result = None

with warnings.catch_warnings():

    warnings.filterwarnings("ignore")

    try:

        arima = ARIMA(train_series, [4, 1, 4])

        result = arima.fit(disp=False)

    except Exception as e:

        print("\tARIMA failed", e)
print("AR params:", result.arparams, "MA params:", result.maparams)
# This is to demo the ARIMA model

print(train.iloc[[2]])



df_train = extract_series(X_train, 2, 5)

df_actual = extract_series(y_train, 2, 5)

lang = X_train.iloc[2, 1]

score = arima_model(df_train.copy(), df_actual.copy(), 2, 1, 2, review=True)

print("The SMAPE score is : %.5f" % score)
def plot_prediction_and_actual(model, forecast, actual, xlim=None, ylim=None, figSize=None, title=None):

    fig, ax = plt.subplots(1, 1, figsize=figSize)

    ax.set_ylim(ylim)

    ax.plot(pd.to_datetime(actual["ds"]), actual["y"], "r.")

    model.plot(forecast, ax=ax);

    ax.set_title(title)

    plt.show()
start_date = df_actual.ds.min()

end_date = df_actual.ds.max()



actual_series = df_actual.y.copy()

actual_series.index = df_actual.ds



df_train["y"] = df_train["y"].astype(np.float64).values



df_actual["y"] = df_actual["y"].astype(np.float64).values



m = Prophet()

m.fit(df_train)

future = m.make_future_dataframe(periods=60)

forecast = m.predict(future)
# simple linear model

def normal_model(df_train, df_actual, review=False):

    start_date = df_actual.ds.min()

    end_date = df_actual.ds.max()



    actual_series = df_actual["y"].copy()

    actual_series.index = df_actual["ds"]



    df_train["y"] = df_train["y"].astype(np.float64).values

    

    df_actual["y"] = df_actual["y"].astype(np.float64).values



    m = Prophet()

    m.fit(df_train)

    future = m.make_future_dataframe(periods=60)

    forecast = m.predict(future)



    if review:

        ymin = min(df_actual["y"].min(), df_train["y"].min()) - 100

        ymax = max(df_actual["y"].max(), df_train["y"].max()) + 100

    

        plot_prediction_and_actual(m, forecast, df_actual, ylim=[ymin, ymax], figSize=[12, 4], title="Normal model")



    mask = (forecast["ds"] >= start_date) & (forecast["ds"] <= end_date)

    forecast_series = forecast[mask].yhat # filter  predictions

    forecast_series.index = forecast[mask].ds

    forecast_series[forecast_series < 0] = 0 # negative values correction



    return smape(forecast_series, actual_series)



def holiday_model(df_train, df_actual, lang, review=False):

    start_date = df_actual["ds"].min()

    end_date = df_actual["ds"].max()



    actual_series = df_actual["y"].copy()

    actual_series.index = df_actual["ds"]



    df_train["y"] = df_train["y"].astype(np.float64).values



    df_actual["y"] = df_actual["y"].astype(np.float64).values



    if isinstance(lang, float) and math.isnan(lang):

        holidays = None

    else:

        holidays = holidays_dict[lang]



    m = Prophet(holidays=holidays)

    m.fit(df_train)

    future = m.make_future_dataframe(periods=60)

    forecast = m.predict(future)



    if review:

        ymin = min(df_actual["y"].min(), df_train["y"].min()) -100

        ymax = max(df_actual["y"].max(), df_train["y"].max()) +100

        plot_prediction_and_actual(m, forecast, df_actual, ylim=[ymin, ymax], figSize=[12, 4], title="Holiday model")



    mask = (forecast["ds"] >= start_date) & (forecast["ds"] <= end_date)

    forecast_series = forecast[mask].yhat

    forecast_series.index = forecast[mask].ds

    forecast_series[forecast_series < 0] = 0



    return smape(forecast_series, actual_series)



def yearly_model(df_train, df_actual, lang, review=False):

    start_date = df_actual["ds"].min()

    end_date = df_actual["ds"].max()



    actual_series = df_actual["y"].copy()

    actual_series.index = df_actual["ds"]



    df_train["y"] = df_train["y"].astype(np.float64).values



    df_actual["y"] = df_actual["y"].astype(np.float64).values



    if isinstance(lang, float) and math.isnan(lang):

        holidays = None

    else:

        holidays = holidays_dict[lang]



    m = Prophet(holidays=holidays, yearly_seasonality=True)

    m.fit(df_train)

    future = m.make_future_dataframe(periods=60)

    forecast = m.predict(future)

        

    if review:

        ymin = min(df_actual["y"].min(), df_train["y"].min()) - 100

        ymax = max(df_actual["y"].max(), df_train["y"].max()) + 100

        plot_prediction_and_actual(m, forecast, df_actual, ylim=[ymin, ymax], figSize=[12, 4], title="Yealry model")



    mask = (forecast["ds"] >= start_date) & (forecast["ds"] <= end_date)

    forecast_series = forecast[mask].yhat

    forecast_series.index = forecast[mask].ds

    forecast_series[forecast_series < 0] = 0



    return smape(forecast_series, actual_series)
# log model

def normal_model_log(df_train, df_actual, review=False):

    start_date = df_actual["ds"].min()

    end_date = df_actual["ds"].max()



    actual_series = df_actual["y"].copy()

    actual_series.index = df_actual["ds"]



    df_train["y"] = df_train["y"].astype(np.float64).values

    df_train["y"] = np.log1p(df_train["y"])



    df_actual["y"] = df_actual["y"].astype(np.float64).values

    df_actual["y"] = np.log1p(df_actual["y"])



    m = Prophet()

    m.fit(df_train)

    future = m.make_future_dataframe(periods=60)

    forecast = m.predict(future)

    

    if review:

        ymin = min(df_actual["y"].min(), df_train["y"].min()) - 2

        ymax = max(df_actual["y"].max(), df_train["y"].max()) + 2

        plot_prediction_and_actual(m, forecast, df_actual, ylim=[ymin, ymax], figSize=[12, 4], title="Normal model in log")



    mask = (forecast["ds"] >= start_date) & (forecast["ds"] <= end_date)

    forecast_series = np.expm1(forecast[mask].yhat)

    forecast_series.index = forecast[mask].ds

    forecast_series[forecast_series < 0] = 0



    return smape(forecast_series, actual_series)



def holiday_model_log(df_train, df_actual, lang, review=False):

    start_date = df_actual["ds"].min()

    end_date = df_actual["ds"].max()



    actual_series = df_actual["y"].copy()

    actual_series.index = df_actual["ds"]



    df_train["y"] = df_train["y"].astype(np.float64).values

    df_train["y"] = np.log1p(df_train["y"])



    df_actual["y"] = df_actual["y"].astype(np.float64).values

    df_actual["y"] = np.log1p(df_actual["y"])



    if isinstance(lang, float) and math.isnan(lang):

        holidays = None

    else:

        holidays = holidays_dict[lang]



    m = Prophet(holidays=holidays)

    m.fit(df_train)

    future = m.make_future_dataframe(periods=60)

    forecast = m.predict(future)

    

    if review:

        ymin = min(df_actual["y"].min(), df_train["y"].min()) - 2

        ymax = max(df_actual["y"].max(), df_train["y"].max()) + 2

        plot_prediction_and_actual(m, forecast, df_actual, ylim=[ymin, ymax], figSize=[12, 4], title="Holiday model in log")



    mask = (forecast["ds"] >= start_date) & (forecast["ds"] <= end_date)

    forecast_series = np.expm1(forecast[mask].yhat)

    forecast_series.index = forecast[mask].ds

    forecast_series[forecast_series < 0] = 0



    return smape(forecast_series, actual_series)



def yearly_model_log(df_train, df_actual, lang, review=False):

    start_date = df_actual.ds.min()

    end_date = df_actual.ds.max()

    

    actual_series = df_actual.y.copy()

    actual_series.index = df_actual.ds



    df_train["y"] = df_train["y"].astype(np.float64).values

    df_train["y"] = np.log1p(df_train["y"])



    df_actual["y"] = df_actual["y"].astype("float").values

    df_actual["y"] = np.log1p(df_actual["y"])



    if isinstance(lang, float) and math.isnan(lang):

        holidays = None

    else:

        holidays = holidays_dict[lang]

        

    m = Prophet(holidays=holidays, yearly_seasonality=True)

    m.fit(df_train)

    future = m.make_future_dataframe(periods=60)

    forecast = m.predict(future)



    if review:

        ymin = min(df_actual["y"].min(), df_train["y"].min()) - 2

        ymax = max(df_actual["y"].max(), df_train["y"].max()) + 2

        plot_prediction_and_actual(m, forecast, df_actual, ylim=[ymin, ymax], figSize=[12, 4], title="Yearly model in log")



    mask = (forecast["ds"] >= start_date) & (forecast["ds"] <= end_date)

    forecast_series = np.expm1(forecast[mask].yhat)

    forecast_series.index = forecast[mask].ds

    forecast_series[forecast_series < 0] = 0



    return smape(forecast_series, actual_series)
# This is to demo the facebook prophet model

print(train.iloc[[2]])



df_train = extract_series(X_train, 2, 5)

df_actual = extract_series(y_train, 2, 5)

lang = X_train.iloc[2, 1]

score = holiday_model_log(df_train.copy(), df_actual.copy(), lang, review=True)

print("The SMAPE score is : %.5f" % score)
import warnings

warnings.filterwarnings("ignore")
print(train.iloc[[2]])



df_train = extract_series(X_train, 2, 5)

df_actual = extract_series(y_train, 2, 5)

lang = X_train.iloc[2, 1]

score = holiday_model(df_train.copy(), df_actual.copy(), lang, review=True)

print("The SMAPE score is : %.5f" % score)
print(train.iloc[[4464]])



df_train = extract_series(X_train, 4464, 5)

df_actual = extract_series(y_train, 4464, 5)

lang = X_train.iloc[4464, 1]



score = holiday_model(df_train.copy(), df_actual.copy(), lang, review=True)

print("The SMAPE score is : %.5f" % score)



score = holiday_model_log(df_train.copy(), df_actual.copy(), lang, review=True)

print("The SMAPE score is : %.5f" % score)



score = yearly_model(df_train.copy(), df_actual.copy(), lang, review=True)

print("The SMAPE score is : %.5f" % score)



score = yearly_model_log(df_train.copy(), df_actual.copy(), lang, review=True)

print("The SMAPE score is : %.5f" % score)



score = median_model(df_train.copy(), df_actual.copy(), 14, review=True)

print("The SMAPE score is : %.5f" % score)
train.iloc[[6245]]



df_train = extract_series(X_train, 6245, 5)

df_actual = extract_series(y_train, 6245, 5)

lang = X_train.iloc[6245, 1]

score = holiday_model_log(df_train.copy(), df_actual.copy(), lang, review=True)

print("The SMAPE score is : %.5f" % score)



score = yearly_model_log(df_train.copy(), df_actual.copy(), lang, review=True)

print("The SMAPE score is : %.5f" % score)



score = median_model(df_train.copy(), df_actual.copy(), 14, review=True)

print("The SMAPE score is : %.5f" % score)
train.iloc[[80002]]



df_train = extract_series(X_train, 80002, 5)

df_actual = extract_series(y_train, 80002, 5)

lang = X_train.iloc[80002, 1]

title = X_train.iloc[80002, 4]

print(title)



score = holiday_model(df_train.copy(), df_actual.copy(), lang, review=True)

print("The SMAPE score is : %.5f" % score)



score = holiday_model_log(df_train.copy(), df_actual.copy(), lang, review=True)

print("The SMAPE score is : %.5f" % score)



score = yearly_model_log(df_train.copy(), df_actual.copy(), lang, review=True)

print("The SMAPE score is : %.5f" % score)



# Please use this case to check your implementation of SMAPE

score = median_model(df_train.copy(), df_actual.copy(), 14, review=True)

print("The SMAPE score is : %.5f" % score)
train.iloc[[80009]]



df_train = extract_series(X_train, 80009, 5)

df_actual = extract_series(y_train, 80009, 5)

lang = X_train.iloc[80009, 1]

title = X_train.iloc[80009, 4]

print(title)



score = holiday_model(df_train.copy(), df_actual.copy(), review=True, lang=lang)

print("The SMAPE score is : %.5f" % score)



score = holiday_model_log(df_train.copy(), df_actual.copy(), lang, review=True)

print("The SMAPE score is : %.5f" % score)



score = yearly_model_log(df_train.copy(), df_actual.copy(), lang, review=True)

print("The SMAPE score is : %.5f" % score)



score = median_model(df_train.copy(), df_actual.copy(), 14, review=True)

print("The SMAPE score is : %.5f" % score)



score = arima_model(df_train.copy(), df_actual.copy(), 2, 1, 2, review=True)

print("The SMAPE score is : %.5f" % score)
train.iloc[[14211]]



df_train = extract_series(X_train, 14211, 5)

df_actual = extract_series(y_train, 14211, 5)

lang = X_train.iloc[14211, 1]

title = X_train.iloc[14211, 4]

print(title)

score = holiday_model(df_train.copy(), df_actual.copy(), review=True, lang=lang)

print("The SMAPE score is : %.5f" % score)



score = holiday_model_log(df_train.copy(), df_actual.copy(), lang, review=True)

print("The SMAPE score is : %.5f" % score)



score = yearly_model_log(df_train.copy(), df_actual.copy(), lang, review=True)

print("The SMAPE score is : %.5f" % score)



# if there is too many zero, just use normal is OK.

score = median_model(df_train.copy(), df_actual.copy(), 14, review=True)

print("The SMAPE score is : %.5f" % score)



score = arima_model(df_train.copy(), df_actual.copy(), 7, 1, 2, review=True)

print("The SMAPE score is : %.5f" % score)
series_num = 145033

series_num = 145057



print(train.iloc[[series_num]])



df_train = extract_series(X_train, series_num, 5)

df_actual = extract_series(y_train, series_num, 5)



lang = X_train.iloc[series_num, 1]

title = X_train.iloc[series_num, 4]

print(title)



try:

    score = median_model(df_train.copy(), df_actual.copy(), 14, review=True)

    print("The SMAPE score is : %.5f" % score)

except Exception as e:

    print("Error in calculating median model", e)



try:

    score = holiday_model(df_train.copy(), df_actual.copy(), review=True,lang = lang)

    print("The SMAPE score is : %.5f" % score)

except Exception as e:

    print("Error in calculating holiday model", e)



try:

    score = holiday_model_log(df_train.copy(), df_actual.copy(), lang, review=True)

    print("The SMAPE score is : %.5f" % score)

except Exception as e:

    print("Error in calculating holiday model in log", e)



try:

    score = yearly_model_log(df_train.copy(), df_actual.copy(), lang, review=True)

    print("The SMAPE score is : %.5f" % score)

except Exception as e:

    print("Error in calculating yearly model in log", e)



try:

    score = arima_model(df_train.copy(), df_actual.copy(), 7, 1, 2, review=True)

    print("The SMAPE score is : %.5f" % score)

except Exception as e:

    print("Error in calculating arima model", e)
import glob



def read_from_folder(path):

    filenames = glob.glob(path + "/*.csv")



    dfs = []

    for filename in filenames:

        dfs.append(pd.read_csv(filename, index_col=0))

    

    frame = pd.concat(dfs)

    return frame.sort_index()
# TODO: overall validation score in one number.

def validation_score(score_series):

    return score_series.mean()
valid_fn = r"../input/wiktraffictimeseriesforecast/validation_score.csv"

valid_score_data = pd.read_csv(valid_fn, index_col=0)



print(valid_score_data[0:10])
valid_score_data
# Check which model is the best

print("Validation score for median model (7 days) is: %.6f" % validation_score(valid_score_data["median7"]))

print("Validation score for median model (14 days) is: %.6f" % validation_score(valid_score_data["median14"]))

print("Validation score for median model (21 days) is: %.6f" % validation_score(valid_score_data["median21"]))

print("Validation score for median model (28 days) is: %.6f" % validation_score(valid_score_data["median28"]))

print("Validation score for median model (35 days) is: %.6f" % validation_score(valid_score_data["median35"]))

print("Validation score for median model (42 days) is: %.6f" % validation_score(valid_score_data["median42"]))

print("Validation score for median model (49 days) is: %.6f" % validation_score(valid_score_data["median49"]))



fig, axs  = plt.subplots(4, 2, figsize=[12,12])

valid_score_data["median7"].plot.hist(bins=40, ax=axs[0][0])

valid_score_data["median14"].plot.hist(bins=40, ax=axs[0][1])

valid_score_data["median21"].plot.hist(bins=40, ax=axs[1][0])

valid_score_data["median28"].plot.hist(bins=40, ax=axs[1][1])

valid_score_data["median35"].plot.hist(bins=40, ax=axs[2][0])

valid_score_data["median42"].plot.hist(bins=40, ax=axs[2][1])

valid_score_data["median49"].plot.hist(bins=40, ax=axs[3][0])
print("Validation score for median model w/holiday (7 days) is: %.6f" % validation_score(valid_score_data["median7_h"]))

print("Validation score for median model w/holiday (14 days) is: %.6f" % validation_score(valid_score_data["median14_h"]))

print("Validation score for median model w/holiday (21 days) is: %.6f" % validation_score(valid_score_data["median21_h"]))

print("Validation score for median model w/holiday (28 days) is: %.6f" % validation_score(valid_score_data["median28_h"]))

print("Validation score for median model w/holiday (35 days) is: %.6f" % validation_score(valid_score_data["median35_h"]))

print("Validation score for median model w/holiday (42 days) is: %.6f" % validation_score(valid_score_data["median42_h"]))

print("Validation score for median model w/holiday (49 days) is: %.6f" % validation_score(valid_score_data["median49_h"]))



fig, axs  = plt.subplots(4, 2, figsize=[12, 12])

valid_score_data["median7_h"].plot.hist(bins=40, ax=axs[0][0])

valid_score_data["median14_h"].plot.hist(bins=40, ax=axs[0][1])

valid_score_data["median21_h"].plot.hist(bins=40, ax=axs[1][0])

valid_score_data["median28_h"].plot.hist(bins=40, ax=axs[1][1])

valid_score_data["median35_h"].plot.hist(bins=40, ax=axs[2][0])

valid_score_data["median42_h"].plot.hist(bins=40, ax=axs[2][1])

valid_score_data["median49_h"].plot.hist(bins=40, ax=axs[3][0])
print("Validation score for holiday model is: %.6f" % validation_score(valid_score_data["holiday"]))

print("Validation score for holiday model w/log is: %.6f" % validation_score(valid_score_data["holiday_log"]))

print("Validation score for yearly model w/log is: %.6f" % validation_score(valid_score_data["yearly_log"]))



fig, axs  = plt.subplots(3, 1, figsize=[12, 12])

valid_score_data["holiday"].plot.hist(bins=40, ax=axs[0])

axs[0].set_title("Holiday model")

valid_score_data["holiday_log"].plot.hist(bins=40, ax=axs[1])

axs[1].set_title("Holiday model w/log")

valid_score_data["yearly_log"].plot.hist(bins=40, ax=axs[2])

axs[2].set_title("Yearly model w/log")
def model_to_use(median, holiday_log, yearly_log):

    result = median

    if median * 1 > yearly_log:

        result = yearly_log

    elif median * 1 > holiday_log:

        result = holiday_log



    return result



def model_to_use_linear(median, holiday_log, yearly_log):

    result = median

    if median * 1 > yearly_log:

        result = yearly_log

    elif median * 1 > holiday_log:

        result = holiday_log



    return result



model_score = valid_score_data.apply(lambda x: model_to_use(x["median14"], x["holiday_log"], x["yearly_log"]), axis=1)



print("Validation score for a proposed model is: %.6f" % validation_score(model_score))

model_score.plot.hist(bins=40)
model_score_2 = valid_score_data.min(axis=1)

print("Best possible Validation score for a mixed model is: %.6f" % validation_score(model_score_2))



model_score_2.plot.hist(bins=40)
def model_to_use(median, holiday_log, yearly_log):

    result = median

    if median * 1 > yearly_log:

        result = yearly_log

    elif median * 1 > holiday_log:

        result = holiday_log



    return result



def model_to_use_linear(median, holiday_log, yearly_log):

    result = median

    if median * 1 > yearly_log:

        result = yearly_log

    elif median * 1 > holiday_log:

        result = holiday_log



    return result



model_score = valid_score_data.apply(lambda x: model_to_use(x["median14"], x["holiday_log"], x["yearly_log"]), axis=1)



print("Validation score for a proposed model is: %.6f" % validation_score(model_score))
import time

import re

import warnings

warnings.filterwarnings("ignore")

from sklearn import preprocessing

from sklearn import utils

from sklearn.utils import resample

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.feature_selection import RFECV

from sklearn import metrics

from sklearn.metrics import make_scorer

from sklearn.tree import DecisionTreeClassifier

import lightgbm



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="white")



def get_script_time(script_time):

    hours = script_time // 3600

    minutes = (script_time % 3600) // 60

    seconds = script_time % 60

    result = (f"{hours}h " if hours > 0 else "") + (f"{minutes}m " if minutes > 0 else "") + f"{seconds}s"

    return result
t_start = time.time()



try:

    df = pd.read_csv("../input/featuredata/data.csv")



except FileNotFoundError:

    df = valid_score_data[["median21_h", "holiday_log", "yearly_log"]].round(4)



    # Выбираем лучшую модель из трех для каждой статьи

    df["best_model"] = df.idxmin(axis=1)



    # Считаем агрегации для каждой статьи за все дни

    aggregations = ["mean", "median", "min", "max", "std"]

    df = pd.concat([df, X_train.iloc[:, 5:].agg(aggregations, axis=1).round(4)], axis=1)

    df["std_%"] = np.around(df["std"] / df["mean"] * 100, 4)

    df.loc[df["mean"] == 0, "std_%"] = 0



    print(f"Total aggregations done in {get_script_time(int(time.time() - t_start))}")

    print("\n" + "".join(["=" for _ in range(80)]) + "\n")



    # Считаем агрегации за последние n дней

    for p in range(7, 56, 7):

        tmp = X_train.iloc[:, -p:].agg(aggregations, axis=1).round(4)

        tmp.rename(columns={col: f"{col}_{p}d" for col in tmp.columns}, inplace=True)

        df = pd.concat([df, tmp], axis=1)

        df[f"std_%_{p}d"] = np.around(df[f"std_{p}d"] / df[f"mean_{p}d"] * 100, 4)

        df.loc[df[f"mean_{p}d"] == 0, f"std_%_{p}d"] = 0



        if p != 49:

            for i in range(p + 7, 56, 7):

                tmp = X_train.iloc[:, -i:-p].agg(aggregations, axis=1).round(4)

                tmp.rename(columns={col: f"{col}_{p}-{i}d" for col in tmp.columns}, inplace=True)

                df = pd.concat([df, tmp], axis=1)

                df[f"std_%_{p}-{i}d"] = np.around(df[f"std_{p}-{i}d"] / df[f"mean_{p}-{i}d"] * 100, 4)

                df.loc[df[f"mean_{p}-{i}d"] == 0, f"std_%_{p}-{i}d"] = 0



        print(f"{p} days aggregations done in {get_script_time(int(time.time() - t_start))}")



    df.to_csv("data.csv", index=False)

    print("\n" + "".join(["=" for _ in range(80)]) + "\n")





    # Отношения показателей за последние n дней к n + 7 дней и ко всем дням

    for idx, feature in enumerate(df.columns.tolist()):

        days = re.findall("\d+[d]", feature)

        if days and "-" not in feature:

            aggregation = feature[:feature.find(days[0])-1]

            df[f"{feature}_to_total_ratio"] = np.around(df[feature] / df[aggregation], 4)

            df.loc[df[aggregation] == 0, f"{feature}_to_total_ratio"] = 0



            if days[0] != "49d":

                for to_days in range(int(days[0][:-1]) + 7, 56, 7):

                    df[f"{feature}_to_{to_days}d_ratio"] = np.around(df[feature] / df[f"{aggregation}_{to_days}d"], 4)

                    df.loc[df[f"{aggregation}_{to_days}d"] == 0, f"{feature}_to_{to_days}d_ratio"] = 0



        elif "-" in feature:

            days = int(re.findall("\d+[-]", feature)[0][:-1])

            to_days = int(re.findall("\d+[d]", feature)[0][:-1])

            aggregation = feature[:feature.find(str(days))-1]

            df[f"{aggregation}_{days}d_to_{days}-{to_days}d_ratio"] = np.around(df[f"{aggregation}_{days}d"] /

                                                                                df[f"{aggregation}_{days}-{to_days}d"], 4)



            df.loc[df[f"{aggregation}_{days}-{to_days}d"] == 0, f"{aggregation}_{days}d_to_{days}-{to_days}d_ratio"] = 0



            for d in range(days + 7, 49, 7):

                df[f"{aggregation}_{days}-{to_days}d_to_{d}-{d + 7}d_ratio"] = np.around(df[f"{aggregation}_{days}-{to_days}d"] /

                                                                                         df[f"{aggregation}_{d}-{d + 7}d"], 4)

    

                df.loc[df[f"{aggregation}_{d}-{d + 7}d"] == 0, f"{aggregation}_{days}-{to_days}d_to_{d}-{d + 7}d_ratio"] = 0



        if (idx + 1) % 20 == 0:

            print(f"{idx + 1} ratio features calculated in {get_script_time(int(time.time() - t_start))}")



    # Таргет

    df["target"] = preprocessing.LabelEncoder().fit_transform(df["best_model"])

    df.to_csv("data.csv", index=False)





print(df.shape, "\n")

print(f"Done in {get_script_time(int(time.time() - t_start))}", "\n")

df.head(10)
try:

    features_to_use = pd.read_csv("../input/sfs-features-to-use/features_to_use.csv")["feature"].tolist()



except FileNotFoundError:

    t_start = time.time()

    corr_threshold = .98

    corr_matrix = df.drop(dont_use_cols, axis=1).corr().abs()

    corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    drop_columns = [col for col in corr_matrix.columns.tolist() if any(corr_matrix[col] >= corr_threshold)]



    df.drop(drop_columns, axis=1, inplace=True)



    pd.DataFrame(data={"feature": drop_columns}).to_csv("dropped_features_by_big_correlation.csv", index=False)



    print(f"{len(drop_columns)} columns dropped with correlation >= {corr_threshold}", "\n")

    print(f"Done in {get_script_time(int(time.time() - t_start))}", "\n")

    print(df.shape, "\n")

    df.head()
try:

    features_to_use = pd.read_csv("../input/sfs-features-to-use/features_to_use.csv")["feature"].tolist()

    print(len(features_to_use))



except FileNotFoundError:

    t_start = time.time()



    sfs = SFS(estimator=lightgbm.LGBMClassifier(objective="multiclass", n_jobs=-1),

              k_features=(20, 600),

              cv=5,

              scoring="f1_weighted",

              n_jobs=-1,

              verbose=2)



    sfs = sfs.fit(df.drop(dont_use_cols, axis=1), df["target"])



    print("\n" + "".join(["=" for _ in range(100)]) + "\n")

    print(f"Done in {get_script_time(int(time.time() - t_start))}")



    pd.DataFrame(data={"feature": list(sfs.subsets_[24]["feature_names"])}).to_csv("features_to_use.csv", index=False)
features = features_to_use



ncols = 4

nrows = np.ceil(len(features) / ncols).astype(np.int64)



fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=[8 * ncols, 6 * nrows])



for idx, feature in enumerate(features):

    row, col = int(idx / ncols), int(idx % ncols)

    axes[row, col].set_title(feature)



    legend = []

    for ind, label in enumerate(np.sort(df["best_model"].unique())):

        values = df[(df["best_model"] == label) & (df[feature].notnull())][feature].values

        quantile_1 = np.quantile(values, .025)

        quantile_2 = np.quantile(values, .975)

        values = values[(values > max(0, quantile_1)) & (values < quantile_2)]

        sns.distplot(values, bins=1000, kde=True, hist_kws=dict(alpha=.5), ax=axes[row, col])

        legend += [label]



    axes[row, col].legend(legend)

    axes[row, col].plot()



plt.show()
try:

    score_df = pd.read_csv("../input/gridsearch-results/GridSearch_results.csv")

    score_df = score_df.sort_values(by="score", ascending=False).reset_index(drop=True)



except FileNotFoundError:

    start = time.time()

    score_df = pd.DataFrame(data={})



    best_score = 0

    best_num_leaves = 31

    best_min_data_in_leaf = 1

    best_min_split_gain = 0

    best_colsample_by_tree = 1.0

    best_reg_lambda = 0.0

    best_reg_alpha = 0.0

    best_learning_rate = .1

    best_n_estimators = 100



    best_params = {}

    best_params["num_leaves"] = best_num_leaves

    best_params["min_data_in_leaf"] = best_min_data_in_leaf

    best_params["min_split_gain"] = best_min_split_gain

    best_params["colsample_by_tree"] = best_colsample_by_tree

    best_params["reg_lambda"] = best_reg_lambda

    best_params["reg_alpha"] = best_reg_alpha

    best_params["learning_rate"] = best_learning_rate

    best_params["n_estimators"] = best_n_estimators



    def get_cv_score(estimator, data, shuffle_random_state):

        X, y = utils.shuffle(data[features_to_use], data["target"], random_state=shuffle_random_state)

        offset = int(data.shape[0] * .8)

        X_tr, y_tr = X[:offset], y[:offset]

        X_te, y_te = X[offset:], y[offset:]



        estimator.fit(X_tr, y_tr)

        cv_score_train = metrics.f1_score(y_tr, estimator.predict_proba(X_tr).argmax(axis=1), average="weighted")

        cv_score_test = metrics.f1_score(y_te, estimator.predict_proba(X_te).argmax(axis=1), average="weighted")



        return cv_score_train, cv_score_test





    print("Start tuning Num leaves & Min data in leaf...", "\n")



    for num_leaves in [2, 4, 6, 8, 10, 12, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 1000]:

        for min_data_in_leaf in [2, 4, 6, 8, 10, 20, 30, 40, 50, 100, 250, 500, 1000, 1500, 2000]:

            t_start = time.time()

            lgb = lightgbm.LGBMClassifier(objective="multiclass",

                                          n_jobs=-1,

                                          random_state=42,

                                          num_leaves=num_leaves,

                                          min_data_in_leaf=min_data_in_leaf)



            cv_scores_train, cv_scores_test = np.array([]), np.array([])

            for random_state in [42, 61, 87, 104, 121]:

                cv_score_train, cv_score_test = get_cv_score(estimator=lgb, data=df, shuffle_random_state=random_state)

                cv_scores_train = np.append(cv_scores_train, cv_score_train)

                cv_scores_test = np.append(cv_scores_test, cv_score_test)



            cv_score = np.mean(cv_scores_test) - np.std(cv_scores_test)

            cv_score -= abs(np.mean(cv_scores_train) - np.mean(cv_scores_test))



            if cv_score > best_score:

                best_num_leaves = num_leaves

                best_min_data_in_leaf = min_data_in_leaf

                best_score = cv_score

                best_params["num_leaves"] = best_num_leaves

                best_params["min_data_in_leaf"] = best_min_data_in_leaf

                best_params["best_score"] = best_score



            dct = {"num_leaves": num_leaves,

                   "min_data_in_leaf": min_data_in_leaf,

                   "min_split_gain": best_min_split_gain,

                   "colsample_by_tree": best_colsample_by_tree,

                   "reg_lambda": best_reg_lambda,

                   "reg_alpha": best_reg_alpha,

                   "learning_rate": best_learning_rate,

                   "n_estimators": best_n_estimators,

                   "mean_score_train": cv_scores_train.mean(),

                   "mean_score_test": cv_scores_test.mean(),

                   "std_score_train": cv_scores_train.std(),

                   "std_score_test": cv_scores_test.std(),

                   "score": cv_score}



            score_df = pd.concat([score_df, pd.DataFrame(dct, index=[0])], axis=0, ignore_index=True)



            print(f"num_leaves = {num_leaves}; min_data_in_leaf = {min_data_in_leaf}")

            print(f"Best score {np.around(best_score, 4)}. Spent {get_script_time(int(time.time() - t_start))}", "\n")





    print("\n" + "".join(["=" for i in range(80)]) + "\n")

    print("Start tuning Min Split Gain...", "\n")



    for min_split_gain in [.01, .02, .03, .05, .1, .2, .5, .7, .8, .9]:

        t_start = time.time()

        lgb = lightgbm.LGBMClassifier(objective="multiclass",

                                      n_jobs=-1,

                                      random_state=42,

                                      num_leaves=best_num_leaves,

                                      min_data_in_leaf=best_min_data_in_leaf,

                                      min_split_gain=min_split_gain)



        cv_scores_train, cv_scores_test = np.array([]), np.array([])

        for random_state in [42, 61, 87, 104, 121]:

            cv_score_train, cv_score_test = get_cv_score(estimator=lgb, data=df, shuffle_random_state=random_state)

            cv_scores_train = np.append(cv_scores_train, cv_score_train)

            cv_scores_test = np.append(cv_scores_test, cv_score_test)



        cv_score = np.mean(cv_scores_test) - np.std(cv_scores_test)

        cv_score -= abs(np.mean(cv_scores_train) - np.mean(cv_scores_test))



        if cv_score > best_score:

            best_min_split_gain = min_split_gain

            best_score = cv_score

            best_params["min_split_gain"] = min_split_gain

            best_params["best_score"] = best_score



        dct = {"num_leaves": best_num_leaves,

               "min_data_in_leaf": best_min_data_in_leaf,

               "min_split_gain": min_split_gain,

               "colsample_by_tree": best_colsample_by_tree,

               "reg_lambda": best_reg_lambda,

               "reg_alpha": best_reg_alpha,

               "learning_rate": best_learning_rate,

               "n_estimators": best_n_estimators,

               "mean_score_train": cv_scores_train.mean(),

               "mean_score_test": cv_scores_test.mean(),

               "std_score_train": cv_scores_train.std(),

               "std_score_test": cv_scores_test.std(),

               "score": cv_score}



        score_df = pd.concat([score_df, pd.DataFrame(dct, index=[0])], axis=0, ignore_index=True)



        print(f"min_split_gain = {min_split_gain}. Best score {np.around(best_score, 4)}.")

        print(f"Spent {get_script_time(int(time.time() - t_start))}", "\n")





    print("\n" + "".join(["=" for _ in range(80)]) + "\n")

    print("Start tuning Colsample by Tree...", "\n")



    for colsample_by_tree in [.9, .8, .7, .6, .5, .4, .3, .2, .1]:

        t_start = time.time()

        lgb = lightgbm.LGBMClassifier(objective="multiclass",

                                      n_jobs=-1,

                                      random_state=42,

                                      num_leaves=best_num_leaves,

                                      min_data_in_leaf=best_min_data_in_leaf,

                                      min_split_gain=best_min_split_gain,

                                      colsample_by_tree=colsample_by_tree)



        cv_scores_train, cv_scores_test = np.array([]), np.array([])

        for random_state in [42, 61, 87, 104, 121]:

            cv_score_train, cv_score_test = get_cv_score(estimator=lgb, data=df, shuffle_random_state=random_state)

            cv_scores_train = np.append(cv_scores_train, cv_score_train)

            cv_scores_test = np.append(cv_scores_test, cv_score_test)



        cv_score = np.mean(cv_scores_test) - np.std(cv_scores_test)

        cv_score -= abs(np.mean(cv_scores_train) - np.mean(cv_scores_test))



        if cv_score > best_score:

            best_colsample_by_tree = colsample_by_tree

            best_score = cv_score

            best_params["colsample_by_tree"] = colsample_by_tree

            best_params["best_score"] = best_score



        dct = {"num_leaves": best_num_leaves,

               "min_data_in_leaf": best_min_data_in_leaf,

               "min_split_gain": best_min_split_gain,

               "colsample_by_tree": colsample_by_tree,

               "reg_lambda": best_reg_lambda,

               "reg_alpha": best_reg_alpha,

               "learning_rate": best_learning_rate,

               "n_estimators": best_n_estimators,

               "mean_score_train": cv_scores_train.mean(),

               "mean_score_test": cv_scores_test.mean(),

               "std_score_train": cv_scores_train.std(),

               "std_score_test": cv_scores_test.std(),

               "score": cv_score}



        score_df = pd.concat([score_df, pd.DataFrame(dct, index=[0])], axis=0, ignore_index=True)



        print(f"colsample_by_tree = {colsample_by_tree}. Best score {np.around(best_score, 4)}.")

        print(f"Spent {get_script_time(int(time.time() - t_start))}", "\n")





    print("\n" + "".join(["=" for _ in range(80)]) + "\n")

    print("Start tuning Reg Lambda...", "\n")



    for reg_lambda in [.01, .02, .03, .05, .1, .2, .3, .5, 1, 5, 10, 25, 50, 100]:

        t_start = time.time()

        lgb = lightgbm.LGBMClassifier(objective="multiclass",

                                      n_jobs=-1,

                                      random_state=42,

                                      num_leaves=best_num_leaves,

                                      min_data_in_leaf=best_min_data_in_leaf,

                                      min_split_gain=best_min_split_gain,

                                      colsample_by_tree=best_colsample_by_tree,

                                      reg_lambda=reg_lambda)



        cv_scores_train, cv_scores_test = np.array([]), np.array([])

        for random_state in [42, 61, 87, 104, 121]:

            cv_score_train, cv_score_test = get_cv_score(estimator=lgb, data=df, shuffle_random_state=random_state)

            cv_scores_train = np.append(cv_scores_train, cv_score_train)

            cv_scores_test = np.append(cv_scores_test, cv_score_test)



        cv_score = np.mean(cv_scores_test) - np.std(cv_scores_test)

        cv_score -= abs(np.mean(cv_scores_train) - np.mean(cv_scores_test))



        if cv_score > best_score:

            best_reg_lambda = reg_lambda

            best_score = cv_score

            best_params["reg_lambda"] = reg_lambda

            best_params["best_score"] = best_score



        dct = {"num_leaves": best_num_leaves,

               "min_data_in_leaf": best_min_data_in_leaf,

               "min_split_gain": best_min_split_gain,

               "colsample_by_tree": best_colsample_by_tree,

               "reg_lambda": reg_lambda,

               "reg_alpha": best_reg_alpha,

               "learning_rate": best_learning_rate,

               "n_estimators": best_n_estimators,

               "mean_score_train": cv_scores_train.mean(),

               "mean_score_test": cv_scores_test.mean(),

               "std_score_train": cv_scores_train.std(),

               "std_score_test": cv_scores_test.std(),

               "score": cv_score}



        score_df = pd.concat([score_df, pd.DataFrame(dct, index=[0])], axis=0, ignore_index=True)



        print(f"reg_lambda = {reg_lambda}. Best score {np.around(best_score, 4)}.")

        print(f"Spent {get_script_time(int(time.time() - t_start))}", "\n")





    print("\n" + "".join(["=" for _ in range(80)]) + "\n")

    print("Start tuning Reg Alpha...", "\n")



    for reg_alpha in [.01, .02, .03, .05, .1, .5, 1, 5, 10, 25, 50, 100]:

        t_start = time.time()

        lgb = lightgbm.LGBMClassifier(objective="multiclass",

                                      n_jobs=-1,

                                      random_state=42,

                                      num_leaves=best_num_leaves,

                                      min_data_in_leaf=best_min_data_in_leaf,

                                      min_split_gain=best_min_split_gain,

                                      colsample_by_tree=best_colsample_by_tree,

                                      reg_lambda=best_reg_lambda,

                                      reg_alpha=reg_alpha)



        cv_scores_train, cv_scores_test = np.array([]), np.array([])

        for random_state in [42, 61, 87, 104, 121]:

            cv_score_train, cv_score_test = get_cv_score(estimator=lgb, data=df, shuffle_random_state=random_state)

            cv_scores_train = np.append(cv_scores_train, cv_score_train)

            cv_scores_test = np.append(cv_scores_test, cv_score_test)



        cv_score = np.mean(cv_scores_test) - np.std(cv_scores_test)

        cv_score -= abs(np.mean(cv_scores_train) - np.mean(cv_scores_test))



        if cv_score > best_score:

            best_reg_alpha = reg_alpha

            best_score = cv_score

            best_params["reg_alpha"] = reg_alpha

            best_params["best_score"] = best_score



        dct = {"num_leaves": best_num_leaves,

               "min_data_in_leaf": best_min_data_in_leaf,

               "min_split_gain": best_min_split_gain,

               "colsample_by_tree": best_colsample_by_tree,

               "reg_lambda": best_reg_lambda,

               "reg_alpha": reg_alpha,

               "learning_rate": best_learning_rate,

               "n_estimators": best_n_estimators,

               "mean_score_train": cv_scores_train.mean(),

               "mean_score_test": cv_scores_test.mean(),

               "std_score_train": cv_scores_train.std(),

               "std_score_test": cv_scores_test.std(),

               "score": cv_score}



        score_df = pd.concat([score_df, pd.DataFrame(dct, index=[0])], axis=0, ignore_index=True)



        print(f"reg_alpha = {reg_alpha}. Best score {np.around(best_score, 4)}.")

        print(f"Spent {get_script_time(int(time.time() - t_start))}", "\n")





    print("\n" + "".join(["=" for _ in range(80)]) + "\n")

    print("Start tuning Learning Rate & N estimators...", "\n")



    for learning_rate in [.2, .1, .05, .03, .02, .01, .005, .001]:

        for n_estimators in [50, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000]:

            t_start = time.time()

            lgb = lightgbm.LGBMClassifier(objective="multiclass",

                                          n_jobs=-1,

                                          random_state=42,

                                          num_leaves=best_num_leaves,

                                          min_data_in_leaf=best_min_data_in_leaf,

                                          min_split_gain=best_min_split_gain,

                                          colsample_by_tree=best_colsample_by_tree,

                                          reg_lambda=best_reg_lambda,

                                          reg_alpha=best_reg_alpha,

                                          learning_rate=learning_rate,

                                          n_estimators=n_estimators)



            cv_scores_train, cv_scores_test = np.array([]), np.array([])

            for random_state in [42, 61, 87, 104, 121]:

                cv_score_train, cv_score_test = get_cv_score(estimator=lgb, data=df, shuffle_random_state=random_state)

                cv_scores_train = np.append(cv_scores_train, cv_score_train)

                cv_scores_test = np.append(cv_scores_test, cv_score_test)



            cv_score = np.mean(cv_scores_test) - np.std(cv_scores_test)

            cv_score -= abs(np.mean(cv_scores_train) - np.mean(cv_scores_test))



            if cv_score > best_score:

                best_learning_rate = learning_rate

                best_n_estimators = n_estimators

                best_score = cv_score

                best_params["learning_rate"] = best_learning_rate

                best_params["n_estimators"] = n_estimators

                best_params["best_score"] = best_score



            dct = {"num_leaves": best_num_leaves,

                   "min_data_in_leaf": best_min_data_in_leaf,

                   "min_split_gain": best_min_split_gain,

                   "colsample_by_tree": best_colsample_by_tree,

                   "reg_lambda": best_reg_lambda,

                   "reg_alpha": best_reg_alpha,

                   "learning_rate": learning_rate,

                   "n_estimators": n_estimators,

                   "mean_score_train": cv_scores_train.mean(),

                   "mean_score_test": cv_scores_test.mean(),

                   "std_score_train": cv_scores_train.std(),

                   "std_score_test": cv_scores_test.std(),

                   "score": cv_score}



            score_df = pd.concat([score_df, pd.DataFrame(dct, index=[0])], axis=0, ignore_index=True)



            print(f"learning_rate = {learning_rate}; n_estimators = {n_estimators}")

            print(f"Best score {np.around(best_score, 4)}. Spent {get_script_time(int(time.time() - t_start))}", "\n")



    score_df.to_csv("GridSearch_results.csv", index=False)

    print("\n" + "".join(["=" for _ in range(80)]) + "\n")

    print(f"GridSearch done in {get_script_time(int(time.time() - start))}", "\n")



best_params = score_df[score_df.columns.tolist()[:-5]].to_dict("records")[0]

print("LGBMCLassifier best parameters:")

print(best_params)
try:

    df.drop(["model_choice", "smape"], axis=1, inplace=True)

except KeyError:

    pass



lgb = lightgbm.LGBMClassifier(**best_params, random_state=42, objective="multiclass", n_jobs=-1)



lgb.fit(df[features_to_use], df["target"])



probas = lgb.predict_proba(df[features_to_use])



print(metrics.classification_report(df["target"], probas.argmax(axis=1),

                                    target_names=["holiday_log", "median21_h", "yearly_log"]))



dct = df[["target", "best_model"]].drop_duplicates().to_dict("records")



df["model_choice"] = pd.Series(probas.argmax(axis=1)).map({d["target"]: d["best_model"] for d in dct})



df["smape"] = [df[df["model_choice"].iloc[i]].iloc[i] for i in range(df.shape[0])]





print("\n" + "".join(["=" for _ in range(100)]) + "\n")

print(f"Best median model SMAPE score: %.5f" % np.mean(df["median21_h"]))

print(f"Mixed model SMAPE score: %.5f" % np.mean(df["smape"]))