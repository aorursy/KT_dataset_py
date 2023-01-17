# Boilerplate from the original notebook
import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
open_aq.list_tables()

# For visualization
import seaborn as sns
import matplotlib.pyplot as plt
# First, to get which units exist in the data and how they're encoded
query = """SELECT DISTINCT unit
            FROM `bigquery-public-data.openaq.global_air_quality`
        """
unique_units = open_aq.query_to_pandas_safe(query)
unique_units.unit
# The following countries use a unit that's not ppm to measure pollution
query = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit <> 'ppm'
        """
unit_not_ppm_countries = open_aq.query_to_pandas_safe(query)
unit_not_ppm_countries.country
# That doesn't say much, so let's discover as well how many countries are in the dataset
query = """SELECT COUNT(DISTINCT country) AS num_countries
            FROM `bigquery-public-data.openaq.global_air_quality`
        """
unique_countries = open_aq.query_to_pandas_safe(query)
unique_countries.num_countries
# It seams that was every single country... So let's answer a more interesting question:
# For each country, what is the share of units used for measuring? To answer this, a quick
# visualization works better
query = """SELECT country, SUM(IF(unit = 'ppm', 1, 0))/COUNT(*) AS ppmShare
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY country
            ORDER BY ppmShare DESC
        """
country_unit_share = open_aq.query_to_pandas_safe(query)

plt.subplots(figsize=(11.7, 8.27))
ax = sns.barplot(
    x="country",
    y="ppmShare",
    data=country_unit_share
)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
plt.tight_layout()
plt.title("Share of measurements using the unit 'ppm' per country")
plt.show()
# The following pollutants have **at least** ony value of exactly 0
query = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
pollutants_with_zero = open_aq.query_to_pandas_safe(query)
pollutants_with_zero.pollutant
# Again, it would be good to know how many and which are the different pollutants in the dataset
query = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
        """
unique_pollutants = open_aq.query_to_pandas_safe(query)
unique_pollutants.pollutant
# Again, it seems to be every single one of them... 
# Let's then analyze the data further
query = """SELECT pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
        """
pollutant_values = open_aq.query_to_pandas_safe(query)

# One liner from: https://stackoverflow.com/a/47254621
pollutant_values.groupby('pollutant').describe().unstack(1)
# There are some negative values which doesn't make any sense. They could be encoding missing values...
# So let's filter
filtered = pollutant_values[(pollutant_values.value >= 0)]
filtered.groupby('pollutant').describe().unstack(1)