# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.list_tables()
github.head('sample_commits')
github.head('sample_files')
query="""SELECT COUNT(sc.commit) AS number_of_commits, sf.path
         FROM `bigquery-public-data.github_repos.sample_files` AS sf
         --WHERE sf.path LIKE '%.py'--
         INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS sc
              ON sf.repo_name=sc.repo_name
         GROUP BY sf.path
         HAVING sf.path LIKE '%.py'
         ORDER BY number_of_commits
         """
         
         
github_commits=github.query_to_pandas(query)

github_commits
