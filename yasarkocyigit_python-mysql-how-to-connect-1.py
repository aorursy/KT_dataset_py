"""
firs thing you need to do guys that is install mysql-connector

open terminal and copy paste this: pip install mysql-connector-python-rf

"""

#now we can import "mysql-connector

import mysql.connector



mydb = mysql.connector.connect(
    host = "localhost",
    user ="root",
    passwd = "use your own password here that you created"
)



# UNTIL HERE "host" "user" must be same if you did not change manually.
# There is only one thing will be different that is password. you need to create your own one.
# And then that's all

print(mydb)

"""
and we got this output that means connection successful!

<mysql.connector.connection_cext.CMySQLConnection object at 0x1029c5208>

"""




