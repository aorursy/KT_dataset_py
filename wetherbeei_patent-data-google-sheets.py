from google.cloud import bigquery

client = bigquery.Client()
! pip install gspread
import pandas as pd

import gspread

from oauth2client.service_account import ServiceAccountCredentials



scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]



# Make a new Google Cloud Platform project, then go to IAM > Service Accounts.

# Create a new service account with no permissions, then from the options menu select "Create key".

# Upload the key as a *private* dataset.

# Enable the Google Sheets and Google Drive APIs for your new GCP project.

creds = ServiceAccountCredentials.from_json_keyfile_name("../input/gcpserviceaccount/wetherbeei-backups-181403dab3c0.json", scope)

gc = gspread.authorize(creds)



# Share the sheet publicly or to your service account email address.

sheet_url = 'https://docs.google.com/spreadsheets/d/1T-Iz3ED6kOhlyx3ZDXiJ7F59ShmbFqL0APsymjtIVKM/edit?usp=sharing'



wb = gc.open_by_url(sheet_url)



sheet = wb.worksheet('Sheet1')



data = sheet.get_all_values()



df = pd.DataFrame(data)

display(df)

df.columns = df.iloc[0]

df = df.iloc[1:]



# Match a list of messy numbers to their DOCDB format publication number to join with BigQuery.

import requests



docdb_numbers = []

for row, messy_num in df.itertuples():

    docdb_numbers.append(requests.get(f"https://patents.google.com/api/match?pubnum={messy_num}").text)

print(docdb_numbers)



publications = str(

    docdb_numbers).replace("[", "(").replace("]", ")")
query = r"""

  SELECT

    pubs.publication_number, pubs.publication_date, res.title, pubs.cpc, 

    res.top_terms, res.embedding_v1 as embedding

  FROM 

    `patents-public-data.patents.publications` AS pubs

      INNER JOIN `patents-public-data.google_patents_research.publications` AS res ON

        pubs.publication_number = res.publication_number

  WHERE 

    pubs.publication_number in {}

""".format(publications)

df = client.query(query).to_dataframe()

df