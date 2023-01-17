# Import the Python package 

from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Access the "hacker_news" dataset

# Construct a reference to the "hacker_news" dataset

dataset_ref = client.dataset("github_repos", project="bigquery-public-data")

# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# List all the tables in the "Hacker-news" dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset

for table in tables:

    print(table.table_id)
# Fetching the table - Construct a table reference

table_ref = dataset_ref.table("licenses")

# API request - fetch the table

table = client.get_table(table_ref)

table.schema
# Preview the table header

client.list_rows(table, 

                 #selected_fields=table.schema[:1],

                 max_results=2).to_dataframe()
# Fetching the table - Construct a table reference

table_ref = dataset_ref.table("sample_files")

# API request - fetch the table

table = client.get_table(table_ref)

table.schema

client.list_rows(table,

                max_results=2).to_dataframe()
# Which Hacker News comments generated the most discussion?



old_query = """

        SELECT parent, COUNT(1) As NumPosts

        FROM `bigquery-public-data.hacker_news.comments`

        GROUP BY parent

        HAVING COUNT(id) > 10

        """



# Query to determine the number of files per license, sorted by number of files

query = """

        SELECT L.license, COUNT(1) AS number_of_files

        FROM `bigquery-public-data.github_repos.sample_files` AS sf

        INNER JOIN `bigquery-public-data.github_repos.licenses` AS L 

            ON sf.repo_name = L.repo_name

        GROUP BY L.license

        ORDER BY number_of_files DESC

        """



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 10 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

file_count_by_license = query_job.to_dataframe()

file_count_by_license.head()