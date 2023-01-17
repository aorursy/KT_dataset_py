import pandas as pd

import sqlite3

con = sqlite3.connect('../input/pubsdataset/pubs.sqlite')
# Authors

pd.read_sql_query('SELECT * FROM authors', con)
pd.read_sql_query('SELECT AU_FNAME,AU_LNAME FROM authors', con)
pd.read_sql_query('SELECT AU_FNAME,AU_LNAME FROM authors WHERE STATE ="KS"', con)
pd.read_sql_query('SELECT STATE AS ESTADOS, COUNT(AU_ID) AS "TOTAL AUTHORS" FROM authors GROUP BY STATE', con)
pd.read_sql_query('SELECT STATE AS ESTADOS, COUNT(AU_ID) AS "TOTAL AUTHORS" '

                  'FROM authors '

                  'GROUP BY STATE '

                  'HAVING COUNT(AU_ID)>2', con)
pd.read_sql_query('SELECT * '

                  'FROM PUBLISHERS, TITLES ', con)
pd.read_sql_query('SELECT TITLE,PUB_NAME '

                  'FROM PUBLISHERS, TITLES '

                  'WHERE PUBLISHERS.PUB_ID=TITLES.PUB_ID', con)