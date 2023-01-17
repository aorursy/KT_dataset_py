import time

from google.cloud import bigquery

from kaggle.gcp import KaggleKernelCredentials



chicago_taxi = bigquery.Client(project='bigquerytestdefault')



creae_model_sql = """

CREATE MODEL IF NOT EXISTS 

  `bigquerytestdefault.vimota.taxi_trips`

OPTIONS

  ( model_type='linear_reg',

    ls_init_learn_rate=.15,

    l1_reg=1,

    max_iterations=5 ) AS

SELECT

    fare label,

    trip_seconds,

    trip_miles

FROM

  `bigquery-public-data.chicago_taxi_trips.taxi_trips` WHERE fare is not null

"""

res = chicago_taxi.query(creae_model_sql)

while not res.done():

    time.sleep(0.1)

res.result()



sql = """

    SELECT

        *

    FROM

        ML.TRAINING_INFO(MODEL `bigquerytestdefault.vimota.taxi_trips`)

"""

df = chicago_taxi.query(sql).to_dataframe()

print(df)