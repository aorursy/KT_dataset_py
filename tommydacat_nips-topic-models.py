# The data comes as the raw data files, a transformed CSV file, and a SQLite database

import pandas as pd
import sqlite3

# You can read in the SQLite datbase like this
import sqlite3
con = sqlite3.connect('../input/database.sqlite')
sample = pd.read_sql_query("""
SELECT *
FROM Papers
LIMIT 10""", con)
print(sample)

# You can read a CSV file like this
authors = pd.read_csv("../input/Authors.csv")
print(authors)

# It's yours to take from here!
import GPy
