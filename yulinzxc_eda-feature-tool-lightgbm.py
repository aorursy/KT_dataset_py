from __future__ import division
import numpy as np
import pandas as pd 
import os
import timeit
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
# link pandas and plotly
# import cufflinks as cf

init_notebook_mode(connected=True)
# cf.set_config_file(offline=True, world_readable=True, theme='ggplot')
%matplotlib inline
from matplotlib import rcParams
rcParams['figure.figsize'] = 22, 14
rcParams['axes.titlesize'] = 24
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['legend.fontsize'] = 14
def sub_boxenplots(x, data, y="totals.transactionRevenue_ln", rot=15):
    order = data[x].unique()
    fig, axes = plt.subplots(ncols=2, nrows=1, squeeze=False, figsize=(22, 8))
    sns.boxenplot(x=x, y=y, data=data, ax=axes[0, 0], order=order)
    axes[0, 0].set_title("All Instances")
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=rot, ha='right')
    sns.boxenplot(x=x, y=y, data=data.loc[data[y] > 0, :], ax=axes[0, 1], order=order)
    axes[0, 1].set_title("Instances With Non-zero Revenue")
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=rot, ha='right')
    fig.tight_layout()
prefix = "../input/ga-customer-revenue-prediction/"
import json
from pandas.io.json import json_normalize
def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    read_start = timeit.default_timer()
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows,
                     parse_dates=['date']
                    )
    read_end = timeit.default_timer()
    print("Finish reading {0}, time usage: {1}".format(csv_path, read_end - read_start))
    
    process_start = timeit.default_timer()
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    process_end = timeit.default_timer()
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}, time usage: ", process_end - process_start)
    return df
# Just for convinence, uncomment before submite the codes
X_train_df = load_df(prefix + "train.csv")
X_test_df = load_df(prefix + "test.csv")

print("Done!")
X_train_df.head(1)
submit_id = X_test_df["fullVisitorId"].unique()
X_train_df.loc[:, "totals.transactionRevenue_ln"] = np.log1p(X_train_df["totals.transactionRevenue"].fillna(0).astype("float"))
labels = ['Zero revenue instance', 'Non-zero revenue instance']
values = [X_train_df.loc[X_train_df["totals.transactionRevenue_ln"] == 0, "totals.transactionRevenue_ln"].count(),
         X_train_df.loc[X_train_df["totals.transactionRevenue_ln"] != 0, "totals.transactionRevenue_ln"].count()]
trace = go.Pie(labels=labels, values=values)
layout = dict(
    title = 'Instance Revenue'
)

fig = dict(data=[trace], layout=layout)
iplot(fig, filename='basic_pie_chart')
sum_target_series = X_train_df.groupby("fullVisitorId")["totals.transactionRevenue_ln"].sum()

labels = ['Zero revenue customer', 'Non-zero customer']
values = [sum_target_series[sum_target_series == 0].count(),
         sum_target_series[sum_target_series != 0].count()]
trace = go.Pie(labels=labels, values=values)
layout = dict(
    title = 'Unique Customer Revenue'
)

fig = dict(data=[trace], layout=layout)
iplot(fig, filename='basic_pie_chart')
trace = go.Box(y=X_train_df.loc[X_train_df["totals.transactionRevenue_ln"] > 0, "totals.transactionRevenue_ln"],
              name='Instance')
layout = dict(
    title = 'Instance With Revenue Greater Than Zero'
)

fig = dict(data=[trace], layout=layout)
iplot(fig, filename='basic_pie_chart')
trace = go.Box(y=sum_target_series[sum_target_series > 0], name="Unique User")
iplot([trace])
to_drop_cols = [c for c in X_train_df.columns if X_train_df[c].nunique(dropna=False)==1 ]

X_train_df.drop(to_drop_cols, axis=1, inplace=True)
X_test_df.drop(to_drop_cols, axis=1, inplace=True)
print("Training set shape: ", X_train_df.shape, " Testing set shape: ", X_test_df.shape)
print("Difference features between two sets:")
for col in X_train_df.columns:
    if col not in X_test_df:
        print(col)
X_train_df["trafficSource.campaignCode"].unique()
X_train_df.drop("trafficSource.campaignCode", axis=1, inplace=True)

X_test_df = pd.concat([X_test_df, 
                       pd.Series(np.nan, name='totals.transactionRevenue'), 
                       pd.Series(np.nan, name='totals.transactionRevenue_ln')], 
                      axis=1)

X_all_df = pd.concat([X_train_df, X_test_df], ignore_index=True, sort=False)
X_all_df.drop(['sessionId', 'visitId'], axis=1, inplace=True)
obj_cols = [column for column in X_all_df.columns if X_all_df[column].dtype == object]
for col in obj_cols:
    try:
        X_all_df.loc[:, col] = pd.to_numeric(X_all_df.loc[:, col], errors='raise')
#         print("Successfully parse column: ", col)
    except ValueError as e:
        # mute the string column that cannot be converted to numeric values.
        pass
#         print("Unable to parse column: ", col)
for col in X_all_df.select_dtypes(include='object'):
    if col == "fullVisitorId":
        continue
    try:
        X_all_df.loc[:, col] = pd.Categorical(X_all_df.loc[:, col])
#         print("Successfully parse column: ", col)
    except ValueError as e:
        print("Unable to parse column: ", col)
bool_cols = []
for col in X_all_df:
    if X_all_df[col].nunique() == 2:
        bool_cols.append(col)
X_all_df.loc[:, bool_cols] = X_all_df.loc[:, bool_cols].astype('bool')
def investigate(col_name, data):
    print("The number of unique category")
    print(col_name, ": ", data[col_name].nunique())
    print("*" * 20)
    print("Value count")
    print(data[col_name].value_counts())
def ratio_nan(data):
    num_data = X_all_df.shape[0]
    null_sum = data.isnull().sum()
    null_val_features = null_sum[null_sum > 0]
    if "totals.transactionRevenue_ln" in null_val_features:
        null_val_features.drop(["totals.transactionRevenue", "totals.transactionRevenue_ln"], inplace=True)
    print(null_val_features/ num_data)
print("The features with NaN value percentage: ")
ratio_nan(X_all_df)
X_all_df.loc[:, "totals.bounces"].fillna(0, inplace=True)
X_all_df.loc[:, "totals.bounces"] = X_all_df.loc[:, "totals.bounces"].astype('bool')
sub_boxenplots(data=X_all_df.loc[X_all_df["totals.transactionRevenue_ln"].notnull(), :], x="totals.bounces")
X_all_df.drop(X_all_df.loc[X_all_df["totals.bounces"], "fullVisitorId"].index, axis=0, inplace=True)

# We don't need this feature anymore
X_all_df.drop("totals.bounces", axis=1, inplace=True)

X_all_df.reset_index(drop=True, inplace=True)
X_all_df.loc[:, "totals.newVisits"].fillna(0, inplace=True)
X_all_df.loc[:, "totals.newVisits"] = X_all_df.loc[:, "totals.newVisits"].astype('bool')
X_all_df.loc[:, "totals.pageviews"].fillna(0, inplace=True)
def plot_cmp_stack(data, col_name):
    
    total_null_num = data[col_name].isnull().sum()
    total_not_null_num = data[col_name].notnull().sum()
    
    null_zero_num =  (data.loc[data[col_name].isnull(), "totals.transactionRevenue_ln"] == 0).sum()
    null_non_zero_num = (data.loc[data[col_name].isnull(), "totals.transactionRevenue_ln"] != 0).sum()
    not_null_zero_num = (data.loc[data[col_name].notnull(), "totals.transactionRevenue_ln"] == 0).sum()
    not_null_non_zero_num = (data.loc[data[col_name].notnull(), "totals.transactionRevenue_ln"] != 0).sum()
    
    trace1 = go.Bar(
        x=['NULL', 'Not NULL'],
        y=[null_zero_num/ total_null_num, not_null_zero_num/ total_not_null_num],
        name='Instances With Zero Revenue',
        marker=dict(
            color='rgb(158,202,225)',
            line=dict(
                color='rgb(8,48,107)',
                width=1.5),
        ),
        opacity=0.6
    )
    
    trace2 = go.Bar(
        x=['NULL', 'Not NULL'],
        y=[null_non_zero_num/ total_null_num, not_null_non_zero_num/ total_not_null_num],
        name='Instances With Non-Zero Revenue',
        marker=dict(
            color='rgb(58,200,225)',
            line=dict(
                color='rgb(8,48,107)',
                width=1.5),
            ),
        opacity=0.6        
    )
    
    trace3 = go.Pie(
        labels = ['Ratio Of Non-zero Revenue With This Feature Is Null', 'Ratio Of Non-zero Revenue With This Feature Is Not NULL'],
        values = [null_non_zero_num, not_null_non_zero_num],
        domain = {"x": [0.5, 1]},
        hole = .4,
        name = 'Instances With Non-zero Revenue',
        text = ['Non-zero Ratio']
    )
    
    data=[trace1, trace2, trace3]
    
    layout=go.Layout(
        barmode='stack',
        title = col_name + ": Revenue Comparison Between NULL and Not NULL Value",
        yaxis = {'title': 'Percentage'},
        xaxis = {
            'domain': [0, 0.5]
        },
        showlegend=False
    )
    
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename=col_name + '_stack_bar')
plot_cmp_stack(data=X_train_df, col_name="trafficSource.adwordsClickInfo.gclId")
plot_cmp_stack(data=X_train_df, col_name="trafficSource.isTrueDirect")
plot_cmp_stack(data=X_train_df, col_name="trafficSource.keyword")
plot_cmp_stack(data=X_train_df, col_name="trafficSource.referralPath")
to_drop_cols = ["trafficSource.adContent", "trafficSource.adwordsClickInfo.adNetworkType", 
                "trafficSource.adwordsClickInfo.gclId", "trafficSource.adwordsClickInfo.isVideoAd",
                "trafficSource.adwordsClickInfo.page", "trafficSource.adwordsClickInfo.slot"]

X_all_df.drop(to_drop_cols, axis=1, inplace=True)

X_all_df.loc[:, "trafficSource.isTrueDirect"] = X_all_df.loc[:, "trafficSource.isTrueDirect"].fillna(0).astype('bool')
ratio_nan(X_all_df)
X_all_df.loc[X_all_df["fullVisitorId"] == "0824839726118485274", "visitNumber"].sort_values().head(3)
X_all_df.groupby("fullVisitorId")["visitNumber"].min().unique()
X_all_df.loc[X_all_df["channelGrouping"] == "Direct", "trafficSource.isTrueDirect"].value_counts()
X_all_df.loc[X_all_df["trafficSource.isTrueDirect"] == 1, "channelGrouping"].value_counts()
X_all_df.loc[X_all_df["trafficSource.isTrueDirect"] == 0, "channelGrouping"].value_counts()
X_all_df.loc[X_all_df["trafficSource.isTrueDirect"] == 1, "trafficSource.medium"].value_counts()
X_all_df.loc[X_all_df["trafficSource.isTrueDirect"] == 0, "trafficSource.medium"].value_counts()
temp_df = X_all_df.loc[X_all_df["trafficSource.isTrueDirect"] == 1, :]
temp_df.loc[ temp_df["trafficSource.medium"] == 'referral', 'channelGrouping'].value_counts()
X_all_df.loc[X_all_df["trafficSource.keyword"].notnull(), "channelGrouping"].value_counts()
X_all_df.loc[X_all_df["trafficSource.keyword"].isnull(), "channelGrouping"].value_counts()
temp_df = X_all_df.loc[X_all_df["trafficSource.keyword"].isnull(), :]
temp_df.loc[(temp_df["channelGrouping"] == "Organic Search") | (temp_df["channelGrouping"] == "Paid Search"), "trafficSource.medium"].value_counts()
X_all_df.loc[temp_df.index, "trafficSource.keyword"] = "(not provided)"
X_all_df.loc[X_all_df["trafficSource.referralPath"].isnull(), "channelGrouping"].value_counts()
temp_df = X_all_df.loc[X_all_df["trafficSource.referralPath"].isnull(), :]
X_all_df.loc[temp_df.loc[(temp_df["channelGrouping"] == "Referral") | (temp_df["channelGrouping"] == "Social")].index, "trafficSource.referralPath"] = X_all_df["trafficSource.referralPath"].value_counts().index[0]
X_all_df.loc[X_all_df["trafficSource.source"] == "(direct)", "channelGrouping"].value_counts()
X_all_df.loc[X_all_df["trafficSource.source"] == "(direct)", "channelGrouping"] = "Direct"
X_all_df["trafficSource.referralPath"] = X_all_df["trafficSource.referralPath"].cat.add_categories(['Not Referral'])

X_all_df.loc[:, "trafficSource.referralPath"] = X_all_df.loc[:, "trafficSource.referralPath"].fillna('Not Referral')
def percentage_counts(data, col_name):
    total = data[col_name].count()
    return data[col_name].value_counts()/ total * 100
def keep_greater_than_percentage_entries(data, col_name, percentage=0.1, new_entry="Others"):
    percentage_counts_series = percentage_counts(temp_df, col_name)
    entries_to_keep = percentage_counts_series[percentage_counts_series >= percentage].index
    data[col_name] = data[col_name].cat.add_categories([new_entry])
    data.loc[~data[col_name].isin(entries_to_keep), col_name] = new_entry
    data[col_name] = data[col_name].cat.remove_unused_categories()
features_to_merge = [
    'device.browser', 'geoNetwork.city', 'geoNetwork.country',
    'geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region',
    'geoNetwork.subContinent', 'trafficSource.campaign', 'trafficSource.keyword',
    'trafficSource.referralPath', 'trafficSource.source', 'device.operatingSystem'
                    ]
for feature in features_to_merge:
    keep_greater_than_percentage_entries(data=X_all_df, col_name=feature)
X_train_df = X_all_df.loc[X_all_df["totals.transactionRevenue_ln"].notnull(), :].reset_index(drop=True)
X_test_df = X_all_df.loc[X_all_df["totals.transactionRevenue_ln"].isnull(), :].reset_index(drop=True)
X_train_df.groupby("date")["totals.transactionRevenue_ln"].agg(['sum', 'count', 'mean']).plot(subplots=True, sharex=True, title="Revenue Base On Date", linewidth=2)
sub_boxenplots(x='channelGrouping', data=X_train_df)
sub_boxenplots(x='device.deviceCategory', data=X_train_df)
sub_boxenplots(x='device.isMobile', data=X_train_df)
sub_boxenplots(x='geoNetwork.continent', data=X_train_df)
sub_boxenplots(x='totals.newVisits', data=X_train_df)
sub_boxenplots(x='trafficSource.isTrueDirect', data=X_train_df)
sub_boxenplots(x='trafficSource.campaign', data=X_train_df, rot=20)
sub_boxenplots(x='trafficSource.medium', data=X_train_df)
sub_boxenplots(x='device.operatingSystem', data=X_train_df, rot=20)
g = sns.jointplot('totals.hits', 'totals.transactionRevenue_ln', data=X_train_df[X_train_df['totals.transactionRevenue_ln'] > 0],
                 kind='reg', height=10)
g = sns.jointplot('visitStartTime', 'totals.transactionRevenue_ln', data=X_train_df[X_train_df['totals.transactionRevenue_ln'] > 0],
                 kind='reg', height=10)
g = sns.jointplot('totals.pageviews', 'totals.transactionRevenue_ln', data=X_train_df[X_train_df['totals.transactionRevenue_ln'] > 0],
                 kind='reg', height=10)
cat_feature_list = ['channelGrouping', 'device.browser', 'device.deviceCategory', 'device.operatingSystem', 
                    'geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country', 'geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region', 'geoNetwork.subContinent',
                    'trafficSource.campaign', 'trafficSource.isTrueDirect', 'trafficSource.keyword', 'trafficSource.medium', 'trafficSource.referralPath', 'trafficSource.source'
                   ]
bool_feature_list = ['device.isMobile', 'totals.newVisits']
id_feature_list = ['fullVisitorId']
num_feature_list = ['visitNumber', 'totals.hits', 'totals.pageviews', 'visitStartTime']
time_feature_list = ['date']
label_list = ['totals.transactionRevenue', 'totals.transactionRevenue_ln']
def check_features():
    found = True
    total_feature_list = id_feature_list + label_list + cat_feature_list + num_feature_list + bool_feature_list + time_feature_list
    for feature in X_train_df.columns:
        if feature not in total_feature_list:
            found = False
        assert found, "You forgot " + feature
        total_feature_list.remove(feature)
    if found:
        if total_feature_list == []:
            print("All the features are found!")
        else:
            print("There are features left: ", total_feature_list)
try:
    check_features()
except AssertionError as e:
    print(e)
X_all_df = X_all_df[id_feature_list + label_list + cat_feature_list + num_feature_list + bool_feature_list + time_feature_list]
from sklearn.preprocessing import LabelEncoder
def cat_feature_encoding(data, cat_feature_list):
    feature_encoder_dict = {}
    with tqdm(cat_feature_list, desc=cat_feature_list[0]) as t:
        for cat_feature in t:
            if cat_feature == "fullVisitorId": # leave the ID as it is 
                continue
            t.set_description_str(cat_feature)
            my_label_encoder = LabelEncoder()
            encoded_col = my_label_encoder.fit_transform(data[cat_feature])
            data.loc[:, cat_feature] = encoded_col
            feature_encoder_dict[cat_feature] = my_label_encoder
    return feature_encoder_dict
X_train_df = X_all_df.loc[X_all_df["totals.transactionRevenue_ln"].notnull(), :].reset_index(drop=True)
X_test_df = X_all_df.loc[X_all_df["totals.transactionRevenue_ln"].isnull(), :].reset_index(drop=True)

train_df = X_train_df
test_df = X_test_df
def printShapes(name1, df1, name2, df2):
    print(name1, df1.shape)
    print(name2, df2.shape)
    
def ratio_nan(data):
    num_data = data.shape[0]
    null_sum = data.isnull().sum()
    null_val_features = null_sum[null_sum > 0]
    print(null_val_features/num_data)
    
def compare_nan(name1, df1, name2, df2):
    print(name1)
    print(ratio_nan(df1))
    print("\n" + name2)
    print(ratio_nan(df2))
    
import datetime as dt
def toPandasTimestamp(val):
    lDate = dt.datetime.fromtimestamp(val)
    lTimestamp = pd.Timestamp(lDate)
    return lTimestamp
    
def printTTShapes():
    printShapes("Train Shape:", train_df, "Test Shape:", test_df)
train_df["visitStartTimestamp"] = train_df["visitStartTime"].apply(toPandasTimestamp)
test_df["visitStartTimestamp"] = test_df["visitStartTime"].apply(toPandasTimestamp)
printTTShapes()
compare_nan("Train", train_df, "Test", test_df)
train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].apply(lambda x: 0.0 if np.isnan(x) else x) 
test_df["totals.transactionRevenue"] = test_df["totals.transactionRevenue"].apply(lambda x: 0.0 if np.isnan(x) else x)
columnsOfInterest = ["fullVisitorId", "visitStartTime", "totals.hits", "totals.pageviews", "visitNumber"]
train_grouped = train_df[columnsOfInterest].groupby("fullVisitorId")
test_grouped = test_df[columnsOfInterest].groupby("fullVisitorId")
train_ds = train_grouped["visitStartTime"].max() - train_grouped["visitStartTime"].min() # ds difference in seconds
test_ds = test_grouped["visitStartTime"].max() - test_grouped["visitStartTime"].min()

train_sh = train_grouped["totals.hits"].sum() # sh - summed hits
train_spv = train_grouped["totals.pageviews"].sum() # spv - summed page views
train_lv = train_grouped["visitNumber"].max() #lv - largestVisit number
test_sh = test_grouped["totals.hits"].sum() # sh - summed hits
test_spv = test_grouped["totals.pageviews"].sum() # spv - summed page views
test_lv = test_grouped["visitNumber"].max() #lv - largestVisit number

train_seconds_per_hit = train_ds / train_sh
train_seconds_per_pageview = train_ds / train_spv 
train_seconds_per_visit = train_ds / train_lv
test_seconds_per_hit = test_ds / test_sh
test_seconds_per_pageview = test_ds / test_spv 
test_seconds_per_visit = test_ds / test_lv

train_nf = pd.concat([train_ds, train_seconds_per_hit, train_seconds_per_pageview, train_seconds_per_visit], 
                    axis = 1, join = "outer", 
                    join_axes = [train_ds.reset_index()["fullVisitorId"]])
test_nf= pd.concat([test_ds, test_seconds_per_hit, test_seconds_per_pageview, test_seconds_per_visit], 
                    axis = 1, join = "outer", 
                    join_axes = [test_ds.reset_index()["fullVisitorId"]])

train_nf.reset_index(inplace = True)
train_nf.columns = ["fullVisitorId", "visitTimeRange", "secondsPerHit", "secondsPerPageview", "secondsPerVisit"]
test_nf.reset_index(inplace = True)
test_nf.columns = ["fullVisitorId", "visitTimeRange", "secondsPerHit", "secondsPerPageview", "secondsPerVisit"]
printShapes("Train New Features:", train_nf, "Test New Features:", test_nf)
compare_nan("Train New Features", train_nf, "Test New Features", test_nf)
train_nf = train_nf.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('.'))
test_nf = test_nf.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('.'))
compare_nan("Train New Features", train_nf, "Test New Features", test_nf)
train_nf.head()
test_nf.head()
fm_columns = ["fullVisitorId", "totals.pageviews", "totals.hits", "visitStartTimestamp", "geoNetwork.city"]
ftrain_df = train_df[fm_columns].reset_index()
ftest_df = test_df[fm_columns].reset_index()
ftrain_df.head()
ftest_df.head()
import featuretools as ft
from featuretools import variable_types as vtype

my_variable_types = {
    "fullVisitorId" : vtype.Id,
    "geoNetwork.city" : vtype.Categorical
}
train_es = ft.EntitySet(id = "train_data")
test_es = ft.EntitySet(id = "test_data")
train_es.entity_from_dataframe(entity_id = "train_log",
                               dataframe = ftrain_df,
                               index = "index",
                               time_index = "visitStartTimestamp",
                               variable_types = my_variable_types)

test_es.entity_from_dataframe(entity_id = "test_log",
                              dataframe = ftest_df,
                              index = "index",
                              time_index = "visitStartTimestamp",
                              variable_types = my_variable_types)

print(train_es, test_es)
train_es.normalize_entity(base_entity_id = "train_log",
                          new_entity_id = "visitors",
                          index = "fullVisitorId")

test_es.normalize_entity(base_entity_id = "test_log",
                         new_entity_id = "visitors",
                         index = "fullVisitorId")

print(train_es, test_es)
my_aggs = ["sum", "max", "min", "mean", "std", "mode"]
my_trans = ["month", "day"]

train_features = ft.dfs(entityset = train_es,
                        target_entity = "visitors",
                        max_depth = 2,
                        agg_primitives = my_aggs,
                        trans_primitives = my_trans,
                        verbose = 1,
                        max_features = 100,
                        n_jobs = 4,
                        features_only = True)

test_features = ft.dfs(entityset = test_es,
                        target_entity = "visitors",
                        max_depth = 2,
                        agg_primitives = my_aggs,
                        trans_primitives = my_trans,
                        verbose = 1,
                        max_features = 100,
                        n_jobs = 4,
                        features_only = True)

print("We're generating %d features for train and %d for test" % (len(train_features), len(test_features)))
for feat in train_features:
    print(feat)
train_feature_matrix, train_features = ft.dfs(entityset = train_es,
                                              target_entity = "visitors",
                                              max_depth = 2,
                                              agg_primitives = my_aggs,
                                              trans_primitives = my_trans,
                                              verbose = 1,
                                              max_features = 100,
                                              n_jobs = 4)

test_feature_matrix, test_features = ft.dfs(entityset = test_es,
                                            target_entity = "visitors",
                                            max_depth = 2,
                                            agg_primitives = my_aggs,
                                            trans_primitives = my_trans,
                                            verbose = 1,
                                            max_features = 100,
                                            n_jobs = 4)
train_feature_matrix.head()
test_feature_matrix.head()
printShapes("Train Feature Matrix:", train_feature_matrix,
            "Test Feature Matrix", test_feature_matrix)
compare_nan("Train FM", train_feature_matrix,
            "Test FM", test_feature_matrix)
train_feature_matrix = train_feature_matrix.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('.'))
test_feature_matrix = test_feature_matrix.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('.'))
compare_nan("Train FM", train_feature_matrix,
            "Test FM", test_feature_matrix)
train_feature_matrix.reset_index(inplace = True)
test_feature_matrix.reset_index(inplace = True)
combined_train = train_nf.merge(train_feature_matrix, how = "outer", on = "fullVisitorId")
combined_test = test_nf.merge(test_feature_matrix, how = "outer", on = "fullVisitorId")
combined_train.head()
combined_test.head()
compare_nan("CTrain", combined_train, "CTest", combined_test)
train_rev = train_df[["fullVisitorId", "totals.transactionRevenue"]].groupby("fullVisitorId").sum().reset_index()
test_rev = test_df[["fullVisitorId", "totals.transactionRevenue"]].groupby("fullVisitorId").sum().reset_index()
combined_train_wrev = combined_train.merge(train_rev, how = "outer", on = "fullVisitorId")
combined_test_wrev = combined_test.merge(test_rev, how = "outer", on = "fullVisitorId")
combined_train_wrev = combined_train_wrev.rename(columns = {"totals.transactionRevenue" : "SUM(transactionRevenue)"})
combined_test_wrev = combined_test_wrev.rename(columns = {"totals.transactionRevenue" : "SUM(transactionRevenue)"})
import lightgbm as lgb
from sklearn.model_selection import KFold
new_train_features_encoding_dict = cat_feature_encoding(data=combined_train_wrev, cat_feature_list=combined_train_wrev.select_dtypes(include='object').columns)
new_test_features_encoding_dict = cat_feature_encoding(data=combined_test_wrev, cat_feature_list=combined_test_wrev.select_dtypes(include='object').columns)
y = np.log1p(combined_train_wrev.iloc[:, -1])

X = combined_train_wrev.iloc[:, 1:-1]

X_pred = combined_test_wrev.iloc[:, 1:-1]
lgb_clf = lgb.LGBMRegressor(learning_rate=0.03, n_estimators=2000, min_child_weight=np.power(10.0, 2), metric='rmse', 
                             num_leaves=128, reg_alpha=np.power(10.0, -3.2454), reg_lambda = np.power(10.0, -4.8571), silent=True, n_jobs=-1,
                             colsample_bytree =  0.6810, min_child_samples = 95,  subsample = 0.2217, min_split_gain=np.power(10.0, -4.9380))
from sklearn.metrics import mean_squared_error
kfold = 5
kf = KFold(n_splits=kfold, shuffle=True)

predicts_result = []
test_result = []
for train_index, test_index in kf.split(X, y):
    print("#"*10)
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    lgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=400, verbose=200, eval_metric='rmse') #eval_metric=f1_macro_evaluation)
#     test_result.append(f1_score(y_pred=lgb_clf.predict(X_val), y_true=y_val, average="macro"))
    y_val_pred = lgb_clf.predict(X_val)
    # Modify the value that is below the zero
    y_val_pred[y_val_pred<0] = 0
    test_result.append(mean_squared_error(y_true=y_val, y_pred=y_val_pred))
    y_pred = lgb_clf.predict(X_pred, lgb_clf.best_iteration_)
    y_pred[y_pred<0] = 0
    predicts_result.append(y_pred)
print("The average test RMSE is: ", np.mean(np.sqrt(test_result)))
def plot_features(col_list, feature_importances, index, most_important=True):
    indices = np.argsort(feature_importances)[::-1]
    indices = indices[:index]

    # Visualise these with a barplot
    plt.subplots(figsize=(20, 15))
    g = sns.barplot(y=col_list[indices], x = lgb_clf.feature_importances_[indices], orient='h')
    g.set_xlabel("Relative importance",fontsize=20)
    g.set_ylabel("Features",fontsize=20)
    g.tick_params(labelsize=15)
    g.set_title("LightGBM feature importance", fontsize=20);
plot_features(col_list=X.columns, feature_importances=lgb_clf.feature_importances_, index=-1)
no_bounce_pred_df = pd.DataFrame({"fullVisitorId": combined_test_wrev.fullVisitorId, "PredictedLogRevenue": np.mean(predicts_result, axis=0)})

submit_df = pd.DataFrame({"fullVisitorId": submit_id})

submit_df = submit_df.merge(no_bounce_pred_df, on="fullVisitorId", how='left').fillna(0)

submit_df.to_csv("submission.csv", index=False)
submit_df[submit_df.PredictedLogRevenue != 0].hist()
labels = ['Zero revenue customer', 'Non-zero customer']
values = [submit_df[submit_df.PredictedLogRevenue == 0].PredictedLogRevenue.count(),
         submit_df[submit_df.PredictedLogRevenue != 0].PredictedLogRevenue.count()]
trace = go.Pie(labels=labels, values=values)
layout = dict(
    title = 'Unique Customer Revenue Prediction'
)

fig = dict(data=[trace], layout=layout)
iplot(fig, filename='basic_pie_chart')
