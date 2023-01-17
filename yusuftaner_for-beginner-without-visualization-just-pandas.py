import numpy as np 

import pandas as pd 







import os

print(os.listdir("../input"))

df = pd.read_csv('../input/books.csv',error_bad_lines=False)



df.shape

#for shape

df.columns

#for looking all columns
df=df[["bookID","title","authors","language_code","# num_pages","ratings_count"]]

df

#the columns we want to use 
JKRowling=df[:][df["authors"]=="J.K. Rowling-Mary GrandPré"]

JKRowling

#J.K Rowling's books

df["authors"].value_counts().head(10)

#hard-working authors
data=df.sort_values(by='ratings_count', ascending=False).head(20)

data

#top 20 books for ratings_count
data["authors"].value_counts()

#authors of the top 20 books
data["# num_pages"].describe()

#about top 20 books pages()