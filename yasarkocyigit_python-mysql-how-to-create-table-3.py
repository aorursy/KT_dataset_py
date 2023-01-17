"""
i will upload everything about mysql as part once you learn previous part you will be ok for next one!

i uploaded how to connect mysql to python as previous project.

"""

import mysql.connector



mydb = mysql.connector.connect(
    host = "localhost",
    user ="root",
    passwd = "gs163264",
    database="testdb"
)

"""

CREATE TABLE table_name (
    column_1 data_type constraints,
    column_2 data_type constraints,
    ...
    column_n data_type constraints
);


"""

mycursor = mydb.cursor()

#CREATE A TABLE
#my_cursor.execute(CREATE TABLE table_name)

#SHOW TABLE
#my_cursor.execute("SHOW TABLES")

#SHOW DATABASE
#my_cursor.execute("SHOW DATABASES")


mycursor.execute("CREATE TABLE IF NOT EXISTS students (name VARCHAR(255), age INTEGER(10))")