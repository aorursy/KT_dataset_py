# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
query = """ SELECT unit, COUNT(*) as cnt
FROM `bigquery-public-data.openaq.global_air_quality`
GROUP BY unit
""" 

units = open_aq.query_to_pandas_safe(query)

units.unit.tolist()
query = """ SELECT DISTINCT Country 
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit != 'ppm' 
"""

country = open_aq.query_to_pandas_safe(query)

# we will present the country list by converting the result to a list... 
country.Country.tolist()
country.Country.count()
query = """ SELECT DISTINCT 
    latitude, 
    longitude,
    CASE WHEN unit = 'ppm' THEN 'red' ELSE 'green' END as unit
FROM `bigquery-public-data.openaq.global_air_quality`
""" 

ppm = open_aq.query_to_pandas_safe(query)
import matplotlib.pyplot as plot
import seaborn as sns

sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'white'})
ax = plot.scatter(ppm['longitude'].values,ppm['latitude'].values,color=ppm['unit'],s=0.8,alpha=0.8)
ax.axes.set_title('Distribution of pollution units')
ax.figure.set_size_inches(15,10)
plot.grid(False)
#plot.ylim(lat)
#plot.xlim(long)
plot.show()
query = """ SELECT DISTINCT pollutant 
FROM `bigquery-public-data.openaq.global_air_quality`
"""

have_zero = open_aq.query_to_pandas_safe(query)
have_zero.pollutant.tolist()
query = """ SELECT pollutant, 
            MAX(CASE WHEN value = 0 THEN 1 ELSE 0 END) AS has_zero_value
FROM `bigquery-public-data.openaq.global_air_quality`
GROUP BY pollutant
"""

have_zero = open_aq.query_to_pandas_safe(query)
have_zero

query = """ SELECT pollutant, 
            MAX(value) AS max_value
FROM `bigquery-public-data.openaq.global_air_quality`
GROUP BY pollutant
"""

maxval = open_aq.query_to_pandas_safe(query)
maxval
import matplotlib.pyplot as plot

query = """ SELECT pollutant, 
            SUM(CASE WHEN value = 0 THEN 1 ELSE 0 END) AS zero_value,
            COUNT(*) tot_meassures
FROM `bigquery-public-data.openaq.global_air_quality`
GROUP BY pollutant
"""

sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
zero_meassures = open_aq.query_to_pandas_safe(query)
zero_meassures

p1 = plot.bar(zero_meassures.pollutant, zero_meassures.tot_meassures, 0.4)
p2 = plot.bar(zero_meassures.pollutant, zero_meassures.zero_value, 0.4)

plot.ylabel('# of meassures')
plot.title('Pollution Meassures')
#plt.xticks(, ('G1', 'G2', 'G3', 'G4', 'G5'))
#plt.yticks(np.indarange(0, 81, 10))
plot.legend((p1[0], p2[0]), ('Total # of measures','# of zero measures'))

plot.show()
