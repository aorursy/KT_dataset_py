import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import json
import gc
from collections import defaultdict
from math import sin, radians
import os
import sys
sys.path.append(("../input/helper-scripts"))
from chart_helper import bar_chart, line_chart, value_counts_barchart, value_counts_linechart, merged_tail_barchart, merged_tail_linechart
import util_helper as utils
COUNT_THRESHOLD = 30
MERGE_THRESHOLD = 0.001
pd.set_option('display.max_columns', 100)
%%time

# Load and parse JSON
# Since parsing takes > 1 min, we will keep a back up copy for reruns

# Load backup copy if it exists 
try:
    train_data = train_back_up.copy()

# If it does not
except NameError:
    # Load data from file
    train_data = utils.load_data(path="../input/ga-customer-revenue-prediction/train.csv")
    
    # Parse JSON columns in data
    train_data = utils.parse_data(data=train_data)
    
    # Create a back up copy, for re-run
    train_back_up = train_data.copy()
%%time

# Load and parse JSON
# Since parsing takes > 1 min, we will keep a back up copy for reruns

# Load backup copy if it exists 
try:
    test_data = test_back_up.copy()

# If it does not
except NameError:
    # Load data from file
    test_data = utils.load_data(path="../input/ga-customer-revenue-prediction/test.csv")
    
    # Parse JSON columns in data
    test_data = utils.parse_data(data=test_data)
    
    # Create a back up copy, for re-run
    test_back_up = test_data.copy()
# Check shape
train_data.shape
test_data.shape
[c for c in train_data.columns if c not in set(test_data.columns)]
train_data[["visitId","visitStartTime"]].head()
(~train_data['visitId']==train_data["visitStartTime"]).sum()
(~test_data['visitId']==test_data["visitStartTime"]).sum()
# Convert VisitStartTime to datetime object
train_data['visitStartTime'] = train_data['visitStartTime'].apply(pd.datetime.fromtimestamp)
train_data['date'] = pd.to_datetime(train_data['date'], format="%Y%m%d")
# Convert VisitStartTime to datetime object
test_data['visitStartTime'] = test_data['visitStartTime'].apply(pd.datetime.fromtimestamp)
test_data['date'] = pd.to_datetime(test_data['date'], format="%Y%m%d")
train_data[["visitId","visitStartTime"]].head()
%%time

# For every column
for col in train_data.columns:
    # Convert ID to string
    if "Id" in col:
        train_data[col] = train_data[col].astype('str')
    
    # Convert to boolean if applicable
    elif '_is' in col and len(train_data[col].unique()) == 2:
            train_data[col] = train_data[col].astype('bool')
    
    # Convert to float if applicable
    else:
        try:
            train_data[col] = train_data[col].astype('float64')
        except ValueError:
            pass
        except TypeError:
            pass
%%time

# For every column
for col in test_data.columns:
    # Convert ID to string
    if "Id" in col:
        test_data[col] = test_data[col].astype('str')
    # Convert to boolean if applicable
    elif '_is' in col and len(test_data[col].unique()) == 2:
            test_data[col] = test_data[col].astype('bool')
    # Convert to float if applicable
    else:
        try:
            test_data[col] = test_data[col].astype('float64')
        except ValueError:
            pass
        except TypeError:
            pass
%%time

# Replace "nan" and "NaN" strings with np.NaN object
train_data.replace(["nan", "NaN"], np.nan, inplace=True)    
%%time

# Replace "nan" and "NaN" strings with np.NaN object
test_data.replace(["nan", "NaN"], np.nan, inplace=True)    
train_data.dtypes
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)
train_missing_values = defaultdict(list)
train_missing_percentage = defaultdict(list)
for col in train_data.columns:
    if train_data[col].isnull().sum() > 0:
        train_missing_values[col] = train_data[col].isnull().sum()
        train_missing_percentage[col] = train_data[col].isnull().sum()/train_data.shape[0]
figure = bar_chart(x_values=(train_missing_values.keys(),), 
                   y_values=(train_missing_values.values(),), 
                   title="Missing Values", 
                   orientation="h")
iplot(figure)
for i, c in enumerate(train_missing_values.keys()):
        print(f"{i+1}. {c}")
        try:
            print("\t", train_data[c].unique())
        except TypeError:
            print("\t", "Cannot parse")
        print("\n")
train_data["totals_transactionRevenue"].fillna(0, inplace=True)
train_data["totals_bounces"].fillna(0, inplace=True)
train_data["totals_newVisits"].fillna(0, inplace=True)
train_data["totals_pageviews"].fillna(0, inplace=True)
train_data["trafficSource_adwordsClickInfo_page"].fillna(0, inplace=True)
test_missing_values = defaultdict(list)
test_missing_percentage = defaultdict(list)
for col in test_data.columns:
    if test_data[col].isnull().sum() > 0:
        test_missing_values[col] = test_data[col].isnull().sum()
        test_missing_percentage[col] = test_data[col].isnull().sum()/test_data.shape[0]
figure = bar_chart(x_values=(test_missing_values.keys(), ), 
                   y_values=(test_missing_values.values(), ), 
                   title="Missing Values", 
                   orientation="h")
iplot(figure)
for i, c in enumerate(test_missing_values.keys()):
        print(f"{i+1}. {c}")
        try:
            print("\t", test_data[c].unique())
        except TypeError:
            print("\t", "Cannot parse")
        print("\n")
test_data["totals_bounces"].fillna(0, inplace=True)
test_data["totals_newVisits"].fillna(0, inplace=True)
test_data["totals_pageviews"].fillna(0, inplace=True)
test_data["trafficSource_adwordsClickInfo_page"].fillna(0, inplace=True)
# Remove columns with more than p% missing values
p = 0.0
for col in train_missing_percentage:
    if train_data[col].isnull().sum() > int(p*train_data.shape[0]):
        try:
            print(col)
            del train_data[col]
            del test_data[col]
        except KeyError:
            pass
print("Columns in train data only are", set(train_data.columns) - set(test_data.columns))
print("Columns in test data only are", set(test_data.columns) - set(train_data.columns))
%%time
# Find numnber of unique values for each column
unique_values_train = {}
for col in train_data.columns:
    unique_values_train[col] = len(train_data[col].unique())
# Create dictionary with ID fields and their unique values
identifier_fields_train = {}
for k, v in unique_values_train.items():
    if "Id" in k and ("visit" in k.lower() or "session" in k.lower()) or k=="visitStartTime":
        identifier_fields_train[k] = v
for k in identifier_fields_train:
    try:
        del unique_values_train[k]
    except KeyError:
        pass
TOO_MANY = 1000
# Create three seprate dictionary to hold number of unique values
moderate_values_train = {}
single_value_train = {}
too_many_values_train = {}
for k, v in unique_values_train.items():
    if v > 1 and v < TOO_MANY:
        moderate_values_train[k] = v
    elif v == 1:
        single_value_train[k] = v
    else:
        too_many_values_train[k] = v
# List columns with a single value
# This columns are useless for ML or analysis
pd.DataFrame({"Column": list(single_value_train.keys()), "Value": [train_data[k][0] for k in single_value_train.keys()]}) 
train_data = train_data[[c for c in train_data.columns if c not in set(single_value_train.keys())]]
# Plot columns moderate number of unique values
figure = bar_chart(x_values=(moderate_values_train.keys(), ), 
                   y_values=(moderate_values_train.values(),), 
                   title="Unique Values (Moderate Range)", 
                   orientation="v")

iplot(figure)
too_many_values_train
figure = bar_chart(x_values=(too_many_values_train.keys(),), 
                   y_values=(too_many_values_train.values(),), 
                   title="Unique Values (High Range)",
                   orientation="h", 
                   height=300)

iplot(figure)
list(train_data["geoNetwork_networkDomain"].unique())[0:20]
identifier_fields_train
train_data[["fullVisitorId", "visitId", "sessionId", "visitStartTime"]].head()
%%time
# Find numnber of unique values for each column
unique_values_test = {}
for col in test_data.columns:
    unique_values_test[col] = len(test_data[col].unique())
# Create dictionary with ID fields and their unique values
identifier_fields_test = {}
for k, v in unique_values_test.items():
    if "Id" in k and ("visit" in k.lower() or "session" in k.lower()) or k=="visitStartTime":
        identifier_fields_test[k] = v
for k in identifier_fields_test:
    try:
        del unique_values_test[k]
    except KeyError:
        pass
# Create three seprate dictionary to hold number of unique values
moderate_values_test = {}
single_value_test = {}
too_many_values_test = {}
for k, v in unique_values_test.items():
    if v > 1 and v < 1000:
        moderate_values_test[k] = v
    elif v == 1:
        single_value_test[k] = v
    else:
        too_many_values_test[k] = v
# List columns with a single value
# This columns are useless for ML or analysis
pd.DataFrame({"Column": list(single_value_test.keys()), "Value": [test_data[k][0] for k in single_value_test.keys()]}) 
set(single_value_train.keys())-(set(single_value_test.keys()))
(set(single_value_test.keys()))-set(single_value_train.keys())
%%time

test_data = test_data[[c for c in test_data.columns if c not in set(single_value_test.keys())]]
# Plot columns moderate number of unique values
figure = bar_chart(x_values=(moderate_values_test.keys(), ), 
                   y_values=(moderate_values_test.values(), ), 
                   title="Unique Values (Moderate Range) Test Data", 
                   orientation="v")

iplot(figure)
# Plot columns moderate number of unique values
figure = bar_chart(x_values=(moderate_values_train.keys(), moderate_values_test.keys()), 
                   y_values=(moderate_values_train.values(), moderate_values_test.values()), 
                   names = ("Train", "Test"),
                   title="Unique Values (Moderate Range)", 
                   orientation="v")

iplot(figure)
too_many_values_test
figure = bar_chart(x_values=(too_many_values_test.keys(), ), 
                   y_values=(too_many_values_test.values(), ), 
                   title="Unique Values (High Range) Test Data",
                   orientation="h", 
                   height=300)

iplot(figure)
identifier_fields_test
test_data[["fullVisitorId", "visitId", "sessionId", "visitStartTime"]].head()
train_data.shape
column_type = {}
for k in train_data.columns:
    if 'Id' in k:
        column_type[k] = "Identifier"
    elif np.issubdtype(train_data[k].dtype, np.number):
        column_type[k] = "Numerical"
    elif train_data[k].dtype == 'object':
        column_type[k] = "Categorical"
    elif np.issubdtype(train_data[k].dtype, np.datetime64):
        column_type[k] = "DateTime"
    elif np.issubdtype(train_data[k].dtype, np.bool_):
        column_type[k] = "Binary"
    else:
        column_type[k] = "Unknown"
column_info = pd.DataFrame({"ColumnName":list(train_data.columns), 
              "ColumnsType": [column_type[k] for k in train_data.columns], 
              "DataType":list(train_data.dtypes)}, index=np.arange(1, len(list(train_data.columns))+1))
column_info
figure, _ = value_counts_barchart(data=train_data, column="channelGrouping", title_suffix="Train Data" )
iplot(figure)
figure, _ = value_counts_barchart(data=test_data, column="channelGrouping", title_suffix="Test Data")
iplot(figure)
column_info.loc[column_info["ColumnName"].str.contains("device_"), :]
figure, counts = value_counts_barchart(data=train_data, column="device_browser", orientation="v", title_suffix="Train Data")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=train_data, column="device_browser", title_suffix="Train Data")

iplot(figure)
figure, counts = value_counts_barchart(data=test_data, column="device_browser", orientation="v", title_suffix="Test Data")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=test_data, column="device_browser", title_suffix="Test Data")
iplot(figure)
figure, _ = value_counts_barchart(data=train_data, column="device_isMobile", title_suffix="Train Data", orientation='h')

iplot(figure)
figure, _ = value_counts_barchart(data=test_data, column="device_isMobile", title_suffix="Test Data", orientation='h')

iplot(figure)
figure, _ = value_counts_barchart(data=train_data, column="device_deviceCategory", title_suffix="Train Data", orientation='h')

iplot(figure)
figure, _ = value_counts_barchart(data=test_data, column="device_deviceCategory", title_suffix="Test Data", orientation='h')

iplot(figure)
# Plot chennle grouping values relevant
figure,  _ = value_counts_barchart(data=train_data, column="device_operatingSystem", orientation='v', title_suffix="Train Data")

iplot(figure)
# Plot chennle grouping values relevant
figure,  _ = value_counts_barchart(data=test_data, column="device_operatingSystem", orientation='v', title_suffix="Test Data")

iplot(figure)
column_info.loc[column_info["ColumnName"].str.contains("geoNetwork_"), :]
# Plot chennle grouping values breakdown
figure, counts = value_counts_barchart(data=train_data, column="geoNetwork_city", orientation="v", title_suffix="Train Data")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=train_data, column="geoNetwork_city", orientation='v', title_suffix="Train Data")
iplot(figure)
# Plot chennle grouping values breakdown
figure, counts = value_counts_barchart(data=test_data, column="geoNetwork_city", orientation="v", title_suffix="Test Data")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=test_data, column="geoNetwork_city", orientation='v', title_suffix="Test Data")
iplot(figure)
# Plot chennle grouping values breakdown
figure, counts = value_counts_barchart(data=train_data, column="geoNetwork_metro", orientation="v", title_suffix="Train Data")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=train_data, column="geoNetwork_metro", orientation='v', title_suffix="Train Data")
iplot(figure)
figure, counts = value_counts_barchart(data=test_data, column="geoNetwork_metro", orientation="v", title_suffix="Test Data")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=train_data, column="geoNetwork_metro", orientation='v', title_suffix="Test Data")
iplot(figure)
figure,  _ = value_counts_barchart(data=train_data, column="geoNetwork_subContinent", title_suffix="Train Data")

iplot(figure)
figure,  _ = value_counts_barchart(data=test_data, column="geoNetwork_subContinent", title_suffix="Test Data")

iplot(figure)
figure,  _ = value_counts_barchart(data=train_data, column="geoNetwork_continent", title_suffix="Train Data")

iplot(figure)
figure,  _ = value_counts_barchart(data=test_data, column="geoNetwork_continent", title_suffix="Test Data")

iplot(figure)
figure, counts = value_counts_barchart(data=train_data, column="geoNetwork_region", orientation="v", title_suffix="Train Data")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=train_data, column="geoNetwork_region", orientation='v', title_suffix="Train Data")
iplot(figure)
figure, counts = value_counts_barchart(data=test_data, column="geoNetwork_region", orientation="v", title_suffix="Test Data")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=test_data, column="geoNetwork_region", orientation='v', title_suffix="Test Data")

iplot(figure)
figure, counts = value_counts_barchart(data=train_data, column="geoNetwork_country", orientation="v", title_suffix="Train Data")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=train_data, column="geoNetwork_country", orientation='v', title_suffix="Train Data")

iplot(figure)
figure, counts = value_counts_barchart(data=test_data, column="geoNetwork_country", orientation="v", title_suffix="Test Data")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=test_data, column="geoNetwork_country", orientation='v', title_suffix="Test Data")

iplot(figure)
column_info.loc[column_info["ColumnName"].str.contains("totals_"), :]
figure, counts = value_counts_barchart(data=train_data, column="totals_newVisits", title_suffix="Train Data", orientation='h')

iplot(figure)
figure, counts = value_counts_barchart(data=test_data, column="totals_newVisits", title_suffix="Test Data", orientation='h')

iplot(figure)
figure, counts = value_counts_barchart(data=train_data, column="totals_bounces", title_suffix="Train Data", orientation='h')

iplot(figure)
figure, counts = value_counts_barchart(data=test_data, column="totals_bounces", title_suffix="Test Data", orientation='h')

iplot(figure)
fig, count = value_counts_linechart(data=train_data, column="totals_pageviews", title_suffix="Train Data")
iplot(fig)
figure, counts = value_counts_linechart(data=test_data, column="totals_pageviews", title_suffix="Test Data")

iplot(figure)
figure, counts = value_counts_linechart(data=train_data, column="totals_hits", title_suffix="Train Data")

iplot(figure)
figure, counts = value_counts_linechart(data=test_data, column="totals_hits", title_suffix="Test Data")

iplot(figure)
column_info.loc[column_info["ColumnName"].str.contains("trafficSource_"), :]
figure, counts = value_counts_barchart(data=train_data, column="trafficSource_source", title_suffix="Train Data", orientation="v")
if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=train_data, column="trafficSource_source", title_suffix="Train Data", orientation="v")

iplot(figure)
figure, counts = value_counts_barchart(data=test_data, column="trafficSource_source", title_suffix="Test Data", orientation="v")

if len(counts) > COUNT_THRESHOLD:
    figure, merged_count = merged_tail_barchart(data=test_data, column="trafficSource_source", title_suffix="Test Data", orientation="v")

iplot(figure)
figure, counts = value_counts_barchart(data=train_data, column="trafficSource_medium",  orientation="v", title_suffix="Train Data")

iplot(figure)
figure, counts = value_counts_barchart(data=test_data, column="trafficSource_medium",  orientation="v", title_suffix="Test Data")

iplot(figure)
figure, counts = value_counts_barchart(data=train_data, column="trafficSource_adwordsClickInfo_isVideoAd", orientation="h", title_suffix="Train Data")

iplot(figure)
figure, counts = value_counts_barchart(data=test_data, column="trafficSource_adwordsClickInfo_isVideoAd", orientation="h", title_suffix="Test Data")

iplot(figure)
figure, count = value_counts_linechart(data=train_data, column="trafficSource_adwordsClickInfo_page", title_suffix="Train Data")
iplot(figure)
figure, count = value_counts_linechart(data=test_data, column="trafficSource_adwordsClickInfo_page", title_suffix="Test Data")
iplot(figure)
train_data["log_revenue"] = np.log1p(np.array(train_data["totals_transactionRevenue"], dtype='float64'))
train_data["isRevenue"] = train_data["log_revenue"]!=0
fig, _ = value_counts_barchart(data=train_data, column="isRevenue", orientation='h')
iplot(fig)
non_zero_values = list(filter(lambda x: x!=0, list(train_data["log_revenue"])))
data = [go.Histogram(x=non_zero_values)]
layout = go.Layout(title="Nonzero Revenue Distribution",xaxis=dict(title="Revenue (log1p)"), yaxis=dict(title="Frequency"))
fig = go.Figure(data=data, layout=layout)
iplot(fig)
train_subset = train_data[["fullVisitorId", "date"]]
visit_counts_train = train_subset.groupby("fullVisitorId").count()
visit_counts_train.rename(columns={"date":"count"}, inplace=True)
counts = dict(visit_counts_train["count"].value_counts())
fig, _ = value_counts_linechart(data=counts, title="Visit Counts")
iplot(fig)
fig, _ = merged_tail_linechart(data=visit_counts_train, column="count")
iplot(fig)
test_subset = test_data[["fullVisitorId", "date"]]
visit_counts_test = test_subset.groupby("fullVisitorId").count()
visit_counts_test.rename(columns={"date":"count"}, inplace=True)
counts = dict(visit_counts_test["count"].value_counts())
fig, _ = value_counts_linechart(data=counts, title="Visit Counts Test Data")
iplot(fig)
fig, _ = merged_tail_linechart(data=visit_counts_test, column="count", title_suffix="Test Data")
iplot(fig)
visit_per_day_train = train_subset.groupby("date", as_index=False).count()
revenue_visit_per_day = train_data.loc[train_data["isRevenue"], ["date", "fullVisitorId"]].groupby("date", as_index=False).count()
visit_per_day_train.rename(columns={"fullVisitorId":"count"}, inplace=True)
revenue_visit_per_day.rename(columns={"fullVisitorId":"count"}, inplace=True)
trace1 = go.Scatter(x=visit_per_day_train["date"], y=visit_per_day_train["count"], name="All visits")
trace2 = go.Scatter(x=revenue_visit_per_day["date"], y=revenue_visit_per_day["count"], name="Visits with revenue", yaxis="y2")
fig = tools.make_subplots(rows=2, cols=1, specs=[[{}], [{}]], shared_xaxes=True, vertical_spacing=0.01)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
iplot(fig)
visit_per_day_test = test_subset.groupby("date", as_index=False).count()
visit_per_day_test.rename(columns={"fullVisitorId":"count"}, inplace=True)
data=[go.Scatter(x=visit_per_day_test["date"], y=visit_per_day_test["count"])]
iplot(data)