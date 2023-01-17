import bq_helper
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# You can use two dashes (--) to add comments in SQL
query = ("""
        -- Select all the columns we want in our joined table
        SELECT L.license, COUNT(sf.path) AS number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.licenses` as L 
            ON sf.repo_name = L.repo_name -- what columns should we join on?
        GROUP BY L.license
        ORDER BY number_of_files DESC
        """)
github.estimate_query_size(query)
#file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)
# print out all the returned results
#print(file_count_by_license)
query = """WITH python_repo AS(
               SELECT DISTINCT repo_name
               FROM `bigquery-public-data.github_repos.sample_files`
               WHERE path LIKE '%.py'
               )
            SELECT COUNT(c.commit) AS num_commits,
                c.repo_name
            FROM `bigquery-public-data.github_repos.sample_commits` AS c
            INNER JOIN python_repo
            ON c.repo_name=python_repo.repo_name
            GROUP BY c.repo_name
            ORDER BY num_commits DESC
        """
github.estimate_query_size(query)
commits_python_repo=github.query_to_pandas_safe(query,6)
commits_python_repo.head(10)