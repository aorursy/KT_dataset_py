!pip show google-cloud-bigquery

!pip show google-cloud-core

!pip show google-api-core
# !pip install google-cloud-bigquery==1.12.1

!pip install google-api-core==1.11.1 google-auth==1.6.3 google-api-python-client==1.7.9 google-cloud-bigquery==1.12.1 google-cloud-core==1.0.2 google-resumable-media==0.3.2 googleapis-common-protos==1.6.0 rsa==4.0
from google.cloud import bigquery



# Create client object to access database

client = bigquery.Client()



query = """

        SELECT taxi_id,

            trip_start_timestamp,

            trip_end_timestamp,

            trip_seconds,

            AVG(trip_seconds) 

                OVER (

                      PARTITION BY taxi_id

                      ORDER BY trip_start_timestamp

                     ) AS trip_seconds_avg

        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

        WHERE trip_start_timestamp BETWEEN '2017-05-01' AND '2017-05-02'

        """



result = client.query(query).result().to_dataframe()

result.head()