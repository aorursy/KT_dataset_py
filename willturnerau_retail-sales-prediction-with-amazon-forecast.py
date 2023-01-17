import sys
import os
import json
import time

import pandas as pd
import boto3
import dateutil.parser
import matplotlib.pyplot as plt

# importing forecast notebook utility
sys.path.insert( 0, os.path.abspath("../../common") )
import util
# Assign the S3 bucket name and region name - must create S3 bucket first
#text_widget_bucket = util.create_text_widget( "bucket_name", "retailsalesprediction" )
#text_widget_region = util.create_text_widget( "region", "ap-southeast-2", default_value="us-west-2" )

#bucket_name = text_widget_bucket.value
#assert bucket_name, "bucket_name not set."

region = "ap-southeast-2"
assert region, "region not set."
#validate that your account can communicate with Amazon Forecast
session = boto3.Session(region_name=region) 
forecast = session.client(service_name='forecast') 
forecastquery = session.client(service_name='forecastquery')
# Read in the data
df = pd.read_csv("../input/target-time-series/target_time_series (1).csv", dtype = object, names=['timestamp','store','item_id','demand'])
# Seperate training and test dataframes

# Select training set
feb10_to_feb12 = df[(df['timestamp'] >= '2010-02-05') & (df['timestamp'] <= '2012-01-31')]

# Select test set
remaining_df = df[(df['timestamp'] >= '2012-02-01') & (df['timestamp'] <= '2012-10-26')]
# Export them to the 'data' folder
feb10_to_feb12.to_csv("data/item-demand-time-train.csv", header=False, index=False)
remaining_df.to_csv("data/item-demand-time-validation.csv", header=False, index=False)
# Upload the main dataset to S3
# I already uploaded it manually in this case

#key="elec_data/item-demand-time-train.csv"

#boto3.Session().resource('s3').Bucket(bucket_name).Object(key).upload_file("data/item-demand-time-train.csv")
# Set the configuration of the dataset
DATASET_FREQUENCY = "W" 
TIMESTAMP_FORMAT = "yyyy-MM-dd"
project = 'retail_sales_forecasting'
datasetName= project+'_ds'
datasetGroupName= project +'_dsg'
s3DataPath = "s3://kaggle-sales-forecasting/target_time_series.csv"

# Save it
%store project
# Create the Dataset Group
create_dataset_group_response = forecast.create_dataset_group(DatasetGroupName=datasetGroupName,
                                                              Domain="RETAIL",
                                                             )
datasetGroupArn = create_dataset_group_response['DatasetGroupArn']

forecast.describe_dataset_group(DatasetGroupArn=datasetGroupArn)
# Specify the schema, order must match the data. data must not have headers
schema ={
   "Attributes":[
      {
         "AttributeName":"timestamp",
         "AttributeType":"timestamp"
      },
      {
         "AttributeName":"item_id",
         "AttributeType":"string"
      },
       {
         "AttributeName":"target_value",
         "AttributeType":"float"
      }
      
   ]
}
# Create the Dataset
response=forecast.create_dataset(
                    Domain="RETAIL",
                    DatasetType='TARGET_TIME_SERIES',
                    DatasetName=datasetName,
                    DataFrequency=DATASET_FREQUENCY, 
                    Schema = schema
)

datasetArn = response['DatasetArn']
forecast.describe_dataset(DatasetArn=datasetArn)
# Add Dataset to Dataset Group
forecast.update_dataset_group(DatasetGroupArn=datasetGroupArn, DatasetArns=[datasetArn])
# Create the role to provide to Amazon Forecast.
role_name = "retail_sales_forecast"
role_arn = util.get_or_create_iam_role( role_name = role_name )
# Import the data from S3 into Amazon Forecast
datasetImportJobName = 'EP_DSIMPORT_JOB_TARGET'
ds_import_job_response=forecast.create_dataset_import_job(DatasetImportJobName=datasetImportJobName,
                                                          DatasetArn=datasetArn,
                                                          DataSource= {
                                                              "S3Config" : {
                                                                 "Path":s3DataPath,
                                                                 "RoleArn": role_arn
                                                              } 
                                                          },
                                                          TimestampFormat=TIMESTAMP_FORMAT
                                                         )

ds_import_job_arn=ds_import_job_response['DatasetImportJobArn']
print(ds_import_job_arn)

status_indicator = util.StatusIndicator()

while True:
    status = forecast.describe_dataset_import_job(DatasetImportJobArn=ds_import_job_arn)['Status']
    status_indicator.update(status)
    if status in ('ACTIVE', 'CREATE_FAILED'): break
    time.sleep(10)

status_indicator.end()
# Describe the imported dataset
forecast.describe_dataset_import_job(DatasetImportJobArn=ds_import_job_arn)
# Configure the predictor
predictorName= project+'_deeparp_algo' # Set a Predictor Name
forecastHorizon = 26 # Half a year
algorithmArn = 'arn:aws:forecast:::algorithm/Deep_AR_Plus' # Choose an algorithm to use - Deep AR in this case. Could try ARIMA and CNN later
# Create the predictor
create_predictor_response=forecast.create_predictor(PredictorName=predictorName, 
                                                  AlgorithmArn=algorithmArn,
                                                  ForecastHorizon=forecastHorizon,
                                                  PerformAutoML= False,
                                                  PerformHPO=False,
                                                  EvaluationParameters= {"NumberOfBacktestWindows": 1, 
                                                                         "BackTestWindowOffset": 24}, 
                                                      InputDataConfig= {"DatasetGroupArn": datasetGroupArn},
                                                          FeaturizationConfig= {"ForecastFrequency": "W", 
                                                                                    "Featurizations": 
                                                                                    [
                                                                                      {"AttributeName": "target_value", 
                                                                                       "FeaturizationPipeline": 
                                                                                        [
                                                                                          {"FeaturizationMethodName": "filling", 
                                                                                           "FeaturizationMethodParameters": 
                                                                                            {"frontfill": "none", 
                                                                                             "middlefill": "zero", 
                                                                                             "backfill": "zero"}
                                                                                          }
                                                                                        ]
                                                                                      }
                                                                                    ]
                                                                                   }
                                                 )
# Create the resource
predictor_arn=create_predictor_response['PredictorArn']

# Track the progress of the predictor
status_indicator = util.StatusIndicator()

while True:
    status = forecast.describe_predictor(PredictorArn=predictor_arn)['Status']
    status_indicator.update(status)
    if status in ('ACTIVE', 'CREATE_FAILED'): break
    time.sleep(10)

status_indicator.end()
forecast.get_accuracy_metrics(PredictorArn=predictor_arn)
# Create the forecast using the data + forecast ARN
forecastName= project+'_deeparp_algo_forecast'
create_forecast_response=forecast.create_forecast(ForecastName=forecastName,
                                                  PredictorArn=predictor_arn)
forecast_arn = create_forecast_response['ForecastArn']

# Track the progress
status_indicator = util.StatusIndicator()

while True:
    status = forecast.describe_forecast(ForecastArn=forecast_arn)['Status']
    status_indicator.update(status)
    if status in ('ACTIVE', 'CREATE_FAILED'): break
    time.sleep(10)

status_indicator.end()
# Retrieve the forecast
print(forecast_arn)
print()
forecastResponse = forecastquery.query_forecast(ForecastArn=forecast_arn)
print(forecastResponse)
# Query the predictor
forecastResponse = forecastquery.query_forecast(ForecastArn=forecast_arn)
# Retrieve the Actuals data
actual_df = pd.read_csv("data/item-demand-time-validation.csv", names=['timestamp','store','item_id','demand'])
# Generate DF for each confidence level
prediction_df_p10 = pd.DataFrame.from_dict(forecastResponse['Forecast']['Predictions']['p10'])
prediction_df_p50 = pd.DataFrame.from_dict(forecastResponse['Forecast']['Predictions']['p50'])
prediction_df_p90 = pd.DataFrame.from_dict(forecastResponse['Forecast']['Predictions']['p90'])
# Create an empty df to add the actual & predicted values into
results_df = pd.DataFrame(columns=['timestamp', 'value', 'source'])

# Append the actual values into the df
for index, row in actual_df.iterrows():
    clean_timestamp = dateutil.parser.parse(row['timestamp'])
    results_df = results_df.append({'timestamp' : clean_timestamp , 'value' : row['value'], 'source': 'actual'} , ignore_index=True)
    
# Now add the P10, P50, and P90 Values
for index, row in prediction_df_p10.iterrows():
    clean_timestamp = dateutil.parser.parse(row['Timestamp'])
    results_df = results_df.append({'timestamp' : clean_timestamp , 'value' : row['Value'], 'source': 'p10'} , ignore_index=True)
for index, row in prediction_df_p50.iterrows():
    clean_timestamp = dateutil.parser.parse(row['Timestamp'])
    results_df = results_df.append({'timestamp' : clean_timestamp , 'value' : row['Value'], 'source': 'p50'} , ignore_index=True)
for index, row in prediction_df_p90.iterrows():
    clean_timestamp = dateutil.parser.parse(row['Timestamp'])
    results_df = results_df.append({'timestamp' : clean_timestamp , 'value' : row['Value'], 'source': 'p90'} , ignore_index=True)
# Pivot the dataframe into a better format for plotting
pivot_df = results_df.pivot(columns='source', values='value', index="timestamp")
# Convert the timestamp value to a datetime64 type
actual_df['timestamp'] = actual_df['timestamp'].astype('datetime64')
# Plot the Actual Values, the Predicted Values, and the p90 to p10 error range
plt.figure(figsize=(15,7))
plt.plot(pivot_df.index, pivot_df['actual'], label="Actual")
plt.plot(pivot_df.index, pivot_df['p50'], label="p50")
plt.legend()
plt.fill_between(pivot_df.index, pivot_df['p90'], pivot_df['p10'],
                 color='gray', alpha=0.2)
pivot_df.plot()
# Delete the Forecast:
util.wait_till_delete(lambda: forecast.delete_forecast(ForecastArn=forecast_arn))
# Delete the Predictor:
util.wait_till_delete(lambda: forecast.delete_predictor(PredictorArn=predictor_arn))
# Delete the Import:
util.wait_till_delete(lambda: forecast.delete_dataset_import_job(DatasetImportJobArn=ds_import_job_arn))
# Delete the Dataset:
util.wait_till_delete(lambda: forecast.delete_dataset(DatasetArn=datasetArn))
# Delete the DatasetGroup:
util.wait_till_delete(lambda: forecast.delete_dataset_group(DatasetGroupArn=datasetGroupArn))
# Delete the IAM role
util.delete_iam_role( role_name )