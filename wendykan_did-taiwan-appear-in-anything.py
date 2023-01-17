# The data comes both as CSV files and a SQLite database

import pandas as pd
import sqlite3

# You can read in the SQLite datbase like this
import sqlite3
con = sqlite3.connect('../input/database.sqlite')
sample = pd.read_sql_query("""
SELECT *
FROM Emails e
where e.RawText like '%Taiwan%'
""", con)
print(sample)

# It's yours to take from here!
sample.RawText[13]
