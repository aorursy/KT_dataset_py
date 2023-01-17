import bq_helper

import numpy as np

import pandas as pd

import sqlite3 as sql 
db = sql.connect("bookshelf.sqlite")

cursor = db.cursor()
cursor.execute("PRAGMA busy_timeout = 30000")

cursor.execute("CREATE TABLE IF NOT EXISTS book_information (name,author,readornot,rate)")
book1 = "INSERT INTO book_information VALUES ('Crime and Punishment','Dostoyevski','Yes','*****')"

book2 = "INSERT INTO book_information VALUES ('White Fang','Jack London','Yes','***')"

cursor.execute(book1)

cursor.execute(book2)

db.commit()

db.close()
import os 



database = "bookshelf.sqlite"

folder_exists = os.path.exists(database)
db = sql.connect("bookshelf.sqlite")

cursor = db.cursor()
cursor.execute("SELECT * FROM book_information")

books = cursor.fetchall() # books is a list.

print(books)

for i in books:

    print(i)

    #for k in i:

    #    print(k,end=" ")

    #print("")
cursor.execute("INSERT INTO book_information VALUES ('Greek Mythology','Anna and Louie','Yes','****')")

db.commit()
cursor.execute("SELECT * FROM book_information")

books = cursor.fetchall()

print(books)

for i in books:

    print(i)
cursor.execute("UPDATE book_information SET rate='****' WHERE rate='***'") # where rate is 3 stars

                                                                        # makes them 4 stars

db.commit()

cursor.execute("UPDATE book_information SET readornot='No' WHERE rate ='****'")#if rate equals 3 stars 

                                                                        #make read status no.

db.commit()

cursor.execute("SELECT * FROM book_information")

books = cursor.fetchall()

for i in books:

    print(i)
cursor.execute("DELETE FROM book_information WHERE rate='****'")

db.commit()

cursor.execute("SELECT * FROM book_information")

books = cursor.fetchall()

for i in books:

    print(i)
cursor.execute("CREATE TABLE IF NOT EXISTS special (book_id INTEGER PRIMARY KEY  AUTOINCREMENT,book_name,author,readornot,rate)")

cursor.execute("INSERT INTO special (book_name,author,readornot,rate) VALUES ('Greek Mythology','Anna and Louie','Yes','****')")

cursor.execute("INSERT INTO special (book_name,author,readornot,rate) VALUES ('White Fang','Jack London','Yes','***')")

db.commit()

cursor.execute("SELECT * FROM special")

books = cursor.fetchall()

for i in books:

    print(i)



db.close()
db = sql.connect("bookshelf.sqlite")

cursor = db.cursor()

db.rollback()

cursor.execute("SELECT * FROM book_information")

cursor.fetchall()
cursor.execute("INSERT INTO special (book_name,author,readornot,rate) VALUES ('Greek Mythology','Anna and Louie','Yes','****')")

cursor.execute("INSERT INTO special (book_name,author,readornot,rate) VALUES ('White Fang','Jack London','Yes','***')")

cursor.execute("SELECT * FROM special ORDER BY book_id DESC")

cursor.fetchall()
cursor.execute("SELECT DISTINCT book_name,author,readornot,rate FROM special")

cursor.fetchall()
cursor.execute("SELECT book_name,author,readornot,rate FROM special LIMIT 3")

cursor.fetchall()
cursor.execute("SELECT book_name,author,readornot,rate FROM special WHERE book_id BETWEEN 1 AND 4 ")

cursor.fetchall()

#as you see it contains 1 and 4 
#cursor.execute("SELECT book_name,author,readornot,rate FROM special WHERE book_id IN ()")

# Nothing :)

cursor.execute("SELECT book_name,author,readornot,rate FROM special WHERE book_id IN (1,2,3,4,5,6,7)")

cursor.fetchall()

cursor.execute("SELECT book_name,author,readornot,rate FROM special WHERE author LIKE 'Anna%'")

cursor.fetchall()
cursor.execute("SELECT book_name,author,readornot,rate FROM special WHERE author GLOB '*A*'")

cursor.fetchall()
cursor.execute("SELECT MAX(book_id) book_name,author,readornot,rate FROM special GROUP BY book_name")

cursor.fetchall()
cursor.execute("SELECT book_id,book_name,author,readornot,rate FROM special GROUP BY book_id HAVING book_id=3")

cursor.fetchall()
cursor.execute("SELECT book_id FROM special UNION ALL SELECT book_id FROM special")

cursor.fetchall()
cursor.execute("SELECT book_id FROM special WHERE book_id GLOB '[1-5]' EXCEPT SELECT book_id FROM special WHERE book_id GLOB '[3-8]'")

cursor.fetchall()
cursor.execute("SELECT * ,CASE readornot WHEN 'Yes' THEN 'Read' ELSE 'NotRead' END 'ReadedoRnot' FROM special")

cursor.fetchall()
cursor.execute("REPLACE INTO special (book_name,author,readornot,rate) VALUES ('ABC','ABCD','No','*' )")

cursor.execute("SELECT * FROM special")

cursor.fetchall()