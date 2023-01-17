import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
df_temperature_records = pd.read_csv("../input/temperature-timeseries-for-some-brazilian-cities/station_sao_paulo.csv", )



df_temperature_records.head(10)
df_temperature_records.set_index("YEAR", inplace=True)



df_temperature_records
df_temperature_records.replace(999.90, np.nan, inplace=True)



df_temperature_records
columns_for_mean_records = ["metANN"]

df_temperature_records_months = df_temperature_records.loc[:, :"DEC"]  # slicing up to "DEC"

df_temperature_records_mean = df_temperature_records[columns_for_mean_records]
df_temperature_records_months
df_temperature_records_mean
plt.figure(figsize=(10, 6))

sns.lineplot(data=df_temperature_records_mean, legend=False)

plt.ylabel("Mean temperature (degC)")

plt.show()
df_selected_temperature_year = df_temperature_records_months[df_temperature_records_months.index == 2015]



df_selected_temperature_year
df_selected_temperature_year = df_selected_temperature_year.T



df_selected_temperature_year
plt.figure(figsize=(10, 6))

sns.lineplot(data=df_selected_temperature_year, legend=False)

plt.ylabel("Temperature (degC)")

plt.title(u"São Paulo temperature records - 2015")

plt.show()
df_selected_temperature_2016 = df_temperature_records_months[df_temperature_records_months.index == 2016]

df_selected_temperature_2016 = df_selected_temperature_2016.T



df_selected_temperature_2016
plt.figure(figsize=(10, 6))

sns.lineplot(data=df_selected_temperature_2016, legend=False)

plt.ylabel("Temperature (degC)")

plt.title(u"São Paulo temperature records - 2016")

plt.show()
list_of_df_from_2008_to_2018 = list()



for year in range(2008, 2019):

    df_selected_temperature = df_temperature_records_months[df_temperature_records_months.index == year]

    df_selected_temperature = df_selected_temperature.T

    list_of_df_from_2008_to_2018.append(df_selected_temperature)

    

df_from_2008_to_2018 = pd.concat(list_of_df_from_2008_to_2018, axis=1)



df_from_2008_to_2018
plt.figure(figsize=(10, 6))

sns.lineplot(data=df_from_2008_to_2018, dashes=False)

plt.ylabel("Temperature (degC)")

plt.title(u"São Paulo temperature records from 2008 to 2018")

plt.show()
months = list()

temperature_record = list()



first_temperature_record = df_temperature_records_mean.index[0]

last_temperature_record = df_temperature_records_mean.index[-1]

for year in range(first_temperature_record, last_temperature_record):

    df_selected_temperature = df_temperature_records_months[df_temperature_records_months.index == year]

    for month in df_selected_temperature:

        # This inner loop can be improved, for sure

        current_date = f"{month}-{year}"

        temperature_record.append(df_selected_temperature.loc[:, month].values[0])

        months.append(current_date)

        

list_temperature_per_months = list(zip(months, temperature_record))

df_temperature_per_months = pd.DataFrame(list_temperature_per_months, columns=["Time", "Temperature"])
plt.figure(figsize=(26, 6))

sns.lineplot(x="Time", y="Temperature", data=df_temperature_per_months, sort=False)

plt.ylabel("Temperature (degC)")

plt.title(u"São Paulo temperature records from 1946 to 2019")

plt.xticks(rotation=90)

plt.xticks([])

plt.show()