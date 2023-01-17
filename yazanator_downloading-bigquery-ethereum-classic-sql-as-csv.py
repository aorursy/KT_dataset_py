import numpy as np

import pandas as pd

from google.cloud import bigquery
client = bigquery.Client()

sql = """

SELECT miner, 

    DATE(timestamp) as date,

    COUNT(miner) as total_block_reward

FROM `bigquery-public-data.crypto_ethereum_classic.blocks` 

GROUP BY miner, date

HAVING COUNT(miner) > 0

ORDER BY date, miner, total_block_reward DESC

"""



# Run a Standard SQL query using the environment's default project

df = client.query(sql).to_dataframe()

df
df.to_csv("etc.csv")