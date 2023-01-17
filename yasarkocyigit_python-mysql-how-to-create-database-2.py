"""
i will upload everything about mysql as part once you learn previous part you will be ok for next one!

i uploaded how to connect mysql to python as previous project.

"""

import mysql.connector



mydb = mysql.connector.connect(
    host = "localhost",
    user ="root",
    passwd = "gs163264"
)

# on this lesson we will learn how create a database


mycursor = mydb.cursor()

#CREATE A TABLE
#my_cursor.execute(CREATE TABLE table_name)

#SHOW TABLE
#my_cursor.execute("SHOW TABLES")

#SHOW DATABASE
#my_cursor.execute("SHOW DATABASES")

mycursor.execute("CREATE DATABASE IF NOT EXISTS test1")
# when you check mysql application you will see there is a "matrixdb" databes

# IF YOU DONT USE " IF NOT EXISTS " once you run code you will get syntax error next time
# its because once you execute code you will have that named database and you cant have 2 same named database
# and also you wont get syntax error if you use " IF NOT EXISTS "