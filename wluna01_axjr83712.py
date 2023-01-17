# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#load data and sql library
import pandasql as psql
project_usage = pd.read_csv('../input/user-data/Table 1 - Usage Per Project Per Day.csv')
user_plans = pd.read_csv('../input/user-data/Table 2 - User per plan and project.csv')
correlation_data = pd.read_csv('../input/case-data/Sanity Case Data - Sheet1.csv')
#same as above but assumes first date after Jan 1 for project is start date, filters on project table only
#Joining project_usage table on itself twice
t1 = psql.sqldf( 
    """
    WITH data AS (  
       WITH project_dates AS
            (SELECT MIN(date) as startDate, projectid
            FROM project_usage
            GROUP BY projectid
            HAVING startDate > '2019-01-01') 
        SELECT pu.projectid, pd.startDate, 
            pu.date, pu2.date as yesterday, 
            pu.usage_bandwidth as bandwidth, 
            pu.usage_apiRequests as api, 
            pu.usage_users as users, pu2.usage_users as yesterday_users,
            pu.usage_documents as documents, pu2.usage_documents as yesterday_documents
        FROM project_usage pu
        INNER JOIN project_usage pu2 ON pu.date = date(pu2.date, '+1 day') AND pu.projectid = pu2.projectid
        JOIN project_dates pd ON pu.projectid = pd.projectid
    )
    SELECT *, 
        CASE api
             WHEN 0 THEN FALSE
        ELSE TRUE 
        END api_usage, 
        CASE bandwidth
            WHEN 0 THEN FALSE
        ELSE TRUE
        END bandwidth_usage, 
        CASE documents - yesterday_documents
            WHEN 0 THEN FALSE
        ELSE TRUE
        END document_change, 
        CASE users - yesterday_users
            WHEN 0 THEN FALSE
        ELSE TRUE
        END user_change
    FROM data
    WHERE (JULIANDAY(date) - JULIANDAY(startDate)) <= 7
    """
)

#identical query but changed time in final where clause
t2 = psql.sqldf( 
    """
    WITH data AS (  
       WITH project_dates AS
            (SELECT MIN(date) as startDate, projectid
            FROM project_usage
            GROUP BY projectid
            HAVING startDate > '2019-01-01') 
        SELECT pu.projectid, pd.startDate, 
            pu.date, pu2.date as yesterday, 
            pu.usage_bandwidth as bandwidth, 
            pu.usage_apiRequests as api, 
            pu.usage_users as users, pu2.usage_users as yesterday_users,
            pu.usage_documents as documents, pu2.usage_documents as yesterday_documents
        FROM project_usage pu
        INNER JOIN project_usage pu2 ON pu.date = date(pu2.date, '+1 day') AND pu.projectid = pu2.projectid
        JOIN project_dates pd ON pu.projectid = pd.projectid
    )
    SELECT *, 
        CASE api
             WHEN 0 THEN FALSE
        ELSE TRUE 
        END api_usage, 
        CASE bandwidth
            WHEN 0 THEN FALSE
        ELSE TRUE
        END bandwidth_usage, 
        CASE documents - yesterday_documents
            WHEN 0 THEN FALSE
        ELSE TRUE
        END document_change, 
        CASE users - yesterday_users
            WHEN 0 THEN FALSE
        ELSE TRUE
        END user_change
    FROM data
    WHERE (JULIANDAY(date) - JULIANDAY(startDate)) >= 60
    AND (JULIANDAY(date) - JULIANDAY(startDate)) <= 120
    """
)

#sum cases for t1
t3 = psql.sqldf( 
    """
    SELECT t1.projectid, SUM(t1.api_usage) as  api_usage_before_7, SUM(t1.bandwidth_usage) as bandwidth_usage_before_7, 
        SUM(t1.document_change) as document_change_before_7, SUM(t1.user_change) as user_change_before_7
    FROM t1
    GROUP BY projectid
    """
)

#same query as above for t2
t4 = psql.sqldf( 
    """
    SELECT projectid,
        SUM(t2.api_usage) as  api_usage_after_60, SUM(t2.bandwidth_usage) as bandwidth_usage_after_60, 
        SUM(t2.document_change) as document_change_after_60, SUM(t2.user_change) as user_change_after_60
    FROM t2
    GROUP BY t2.projectid
    """
)
#join queries together
t5 = psql.sqldf(
    """
    SELECT *
    FROM t3
    LEFT JOIN t4 ON t3.projectid = t4.projectid
    """
)
t5
t5.corr()
#same query with time intervals changed slightly
t1 = psql.sqldf( 
    """
    WITH data AS (  
       WITH project_dates AS
            (SELECT MIN(date) as startDate, projectid
            FROM project_usage
            GROUP BY projectid
            HAVING startDate > '2019-01-01') 
        SELECT pu.projectid, pd.startDate, 
            pu.date, pu2.date as yesterday, 
            pu.usage_bandwidth as bandwidth, 
            pu.usage_apiRequests as api, 
            pu.usage_users as users, pu2.usage_users as yesterday_users,
            pu.usage_documents as documents, pu2.usage_documents as yesterday_documents
        FROM project_usage pu
        INNER JOIN project_usage pu2 ON pu.date = date(pu2.date, '+1 day') AND pu.projectid = pu2.projectid
        JOIN project_dates pd ON pu.projectid = pd.projectid
    )
    SELECT *, 
        CASE api
             WHEN 0 THEN FALSE
        ELSE TRUE 
        END api_usage, 
        CASE bandwidth
            WHEN 0 THEN FALSE
        ELSE TRUE
        END bandwidth_usage, 
        CASE documents - yesterday_documents
            WHEN 0 THEN FALSE
        ELSE TRUE
        END document_change, 
        CASE users - yesterday_users
            WHEN 0 THEN FALSE
        ELSE TRUE
        END user_change
    FROM data
    WHERE (JULIANDAY(date) - JULIANDAY(startDate)) <= 7
    """
)

#identical query but changed time in final where clause
t2 = psql.sqldf( 
    """
    WITH data AS (  
       WITH project_dates AS
            (SELECT MIN(date) as startDate, projectid
            FROM project_usage
            GROUP BY projectid
            HAVING startDate > '2019-01-01') 
        SELECT pu.projectid, pd.startDate, 
            pu.date, pu2.date as yesterday, 
            pu.usage_bandwidth as bandwidth, 
            pu.usage_apiRequests as api, 
            pu.usage_users as users, pu2.usage_users as yesterday_users,
            pu.usage_documents as documents, pu2.usage_documents as yesterday_documents
        FROM project_usage pu
        INNER JOIN project_usage pu2 ON pu.date = date(pu2.date, '+1 day') AND pu.projectid = pu2.projectid
        JOIN project_dates pd ON pu.projectid = pd.projectid
    )
    SELECT *, 
        CASE api
             WHEN 0 THEN FALSE
        ELSE TRUE 
        END api_usage, 
        CASE bandwidth
            WHEN 0 THEN FALSE
        ELSE TRUE
        END bandwidth_usage, 
        CASE documents - yesterday_documents
            WHEN 0 THEN FALSE
        ELSE TRUE
        END document_change, 
        CASE users - yesterday_users
            WHEN 0 THEN FALSE
        ELSE TRUE
        END user_change
    FROM data
    WHERE (JULIANDAY(date) - JULIANDAY(startDate)) >= 90
    AND (JULIANDAY(date) - JULIANDAY(startDate)) <= 96
    """
)

#sum cases for t1
t3 = psql.sqldf( 
    """
    SELECT t1.projectid, SUM(t1.api_usage) as  api_usage_before_7, 
        SUM(t1.document_change) as document_change_before_7
    FROM t1
    GROUP BY projectid
    """
)

#same query as above for t2
t4 = psql.sqldf( 
    """
    SELECT projectid,
        SUM(t2.api_usage) as  api_usage_after_60, SUM(t2.bandwidth_usage) as bandwidth_usage_after_60, 
        SUM(t2.document_change) as document_change_after_60, SUM(t2.user_change) as user_change_after_60
    FROM t2
    GROUP BY t2.projectid
    """
)
#join queries together
t5 = psql.sqldf(
    """
    SELECT *
    FROM t3
    LEFT JOIN t4 ON t3.projectid = t4.projectid
    """
)
activated_projects = psql.sqldf(
    """
    SELECT projectid,
    (CASE 
        WHEN api_usage_before_7 > 0 AND document_change_before_7 > 0
        THEN 1 ELSE 0 END) as activated
    FROM t3
    """
)
activated_users = psql.sqldf(
    """
    SELECT sanityUserId, MAX(activated) as activated, COUNT(*)
    FROM user_plans up
    JOIN activated_projects ap ON up.projectid = ap.projectid
    GROUP BY sanityUserId
    """
)
activated_users
first_project = psql.sqldf(
    """
    SELECT MIN(SUBSTR((startedAt), 0, 11)) AS firstProject, sanityUserId
    FROM user_plans
    GROUP BY sanityUserId
    """
)
all_projects = psql.sqldf(
    """
    SELECT projectid, SUBSTR((startedAt), 0, 11) as startedAt, fp.sanityUserId, firstProject
    FROM user_plans up
    LEFT JOIN first_project fp ON up.sanityUserId = fp.sanityUserId
    """
)
relevant_projects = psql.sqldf(
    """
    SELECT *
    FROM all_projects
    WHERE 0 <= (JULIANDAY(startedAt) - JULIANDAY(firstProject)) 
    AND (JULIANDAY(startedAt) - JULIANDAY(firstProject))<= 7
    """
)
num_projects = psql.sqldf(
    """
    SELECT COUNT(*) as numProjects, sanityUserId
    FROM relevant_projects
    WHERE sanityUserId != "service"
    GROUP BY sanityUserId
    HAVING numProjects > 1
    """
)
final_query = psql.sqldf(
    """
    SELECT *
    FROM activated_users au
    JOIN num_projects np ON au.sanityUserId = np.sanityUserId
    """
)
final_query
correlation_data.corr()
project_usage.head(2)
user_plans.head(2)
#how far back does the data go?
t1 = psql.sqldf(
    "SELECT MIN(startedAt) as earliest_singup, MAX(endedAt) as latest_churn FROM user_plans"
)
t1
#how many unique users are there?
t1 = psql.sqldf(
    "SELECT COUNT(DISTINCT(sanityUserId)) as num_unique_users FROM user_plans"
)
t1
#how many projects have start date in middle of data (implying their start date)?
#Result: Under twenty projects have at least three months of data with an implied start date in project_usage
t1 = psql.sqldf(
    """
    SELECT MIN(date), projectid
    FROM project_usage
    GROUP BY projectid
    HAVING MIN(date) > '2019-01-01'
    """
)
t1
#how many unique projects are there (project table)?
t1 = psql.sqldf(
    "SELECT COUNT(DISTINCT(projectid)) as num_unique_projects FROM project_usage"
)
t1
#how many unique projects are there (user table)?
t1 = psql.sqldf(
    "SELECT COUNT(DISTINCT(projectid)) as num_unique_projects FROM user_plans"
)
t1
#how many projects exist in both tables?
t1 = psql.sqldf(
    """
    SELECT COUNT(DISTINCT(user_plans.projectid)) as projects_both_tables
    FROM user_plans
    JOIN project_usage ON user_plans.projectid = project_usage.projectid
    """
)
t1
#Which plans started in the first three months of 2019?
t1 = psql.sqldf( 
    """
    SELECT SUBSTR(MIN(startedAt), 0, 11) as startDate, projectid
    FROM user_plans
    GROUP BY projectid
    HAVING startDate < '2019-04-01'
    AND startDate > '2018-12-31'
    """
)
t1
#Which plans started in the first three months of 2019 and exist in both tables?
t1 = psql.sqldf( 
    """
    WITH project_dates AS
        (SELECT SUBSTR(MIN(startedAt), 0, 11) as startDate, projectid
        FROM user_plans
        GROUP BY projectid
        HAVING startDate < '2019-04-01'
        AND startDate > '2018-12-31')
    SELECT *
    FROM project_dates pd
    JOIN project_usage pu
    ON pd.projectid = pu.projectid
    """
)
t1
#self-join query test to analyze DoD growth , usage_bandwidth, usage_apiRequests, usage_users, usage_documents
t1 = psql.sqldf( 
    """
    WITH data AS (  
       WITH project_dates AS
            (SELECT SUBSTR(MIN(startedAt), 0, 11) as startDate, projectid
            FROM user_plans
            GROUP BY projectid
            HAVING startDate < '2019-04-01'
            AND startDate > '2018-12-31')
        SELECT pu.projectid, pd.startDate, pu.date, pu2.date as yesterday, pu.usage_bandwidth as bandwidth, pu2.usage_bandwidth as yesterday_bandwidth, 
            pu.usage_apiRequests as api, pu2.usage_apiRequests as yesterday_api, pu.usage_users as users, pu2.usage_users as yesterday_users,
            pu.usage_documents as documents, pu2.usage_documents as yesterday_documents
        FROM project_usage pu
        INNER JOIN project_usage pu2 ON pu.date = date(pu2.date, '+1 day') AND pu.projectid = pu2.projectid
        JOIN project_dates pd ON pu.projectid = pd.projectid
    )
    SELECT *
    FROM data
    """
)
t1
#self-join query test to analyze DoD growth , usage_bandwidth, usage_apiRequests, usage_users, usage_documents
#removable Juliandate statement to filter on days
t1 = psql.sqldf( 
    """
    WITH data AS (  
       WITH project_dates AS
            (SELECT SUBSTR(MIN(startedAt), 0, 11) as startDate, projectid
            FROM user_plans
            GROUP BY projectid
            HAVING startDate < '2019-04-01'
            AND startDate > '2018-12-31')
        SELECT pu.projectid, pd.startDate, 
            pu.date, pu2.date as yesterday, 
            pu.usage_bandwidth as bandwidth, pu2.usage_bandwidth as yesterday_bandwidth, 
            pu.usage_apiRequests as api, pu2.usage_apiRequests as yesterday_api, 
            pu.usage_users as users, pu2.usage_users as yesterday_users,
            pu.usage_documents as documents, pu2.usage_documents as yesterday_documents
        FROM project_usage pu
        INNER JOIN project_usage pu2 ON pu.date = date(pu2.date, '+1 day') AND pu.projectid = pu2.projectid
        JOIN project_dates pd ON pu.projectid = pd.projectid
    )
    SELECT projectid, COUNT(*)
    FROM data
    WHERE bandwidth != 0
    GROUP BY projectid
    """
)
t1
#same as above but assumes first date after Jan 1 for project is start date, filters on project table only
#Joining project_usage table on itself twice
t1 = psql.sqldf( 
    """
    WITH data AS (  
       WITH project_dates AS
            (SELECT MIN(date) as startDate, projectid
            FROM project_usage
            GROUP BY projectid
            HAVING startDate > '2019-01-01') 
        SELECT pu.projectid, pd.startDate, 
            pu.date, pu2.date as yesterday, 
            pu.usage_bandwidth as bandwidth, 
            pu.usage_apiRequests as api, 
            pu.usage_users as users, pu2.usage_users as yesterday_users,
            pu.usage_documents as documents, pu2.usage_documents as yesterday_documents
        FROM project_usage pu
        INNER JOIN project_usage pu2 ON pu.date = date(pu2.date, '+1 day') AND pu.projectid = pu2.projectid
        JOIN project_dates pd ON pu.projectid = pd.projectid
    )
    SELECT *, 
        CASE api
             WHEN 0 THEN FALSE
        ELSE TRUE 
        END api_usage, 
        CASE bandwidth
            WHEN 0 THEN FALSE
        ELSE TRUE
        END bandwidth_usage, 
        CASE documents - yesterday_documents
            WHEN 0 THEN FALSE
        ELSE TRUE
        END document_change, 
        CASE users - yesterday_users
            WHEN 0 THEN FALSE
        ELSE TRUE
        END user_change
    FROM data
    WHERE (JULIANDAY(date) - JULIANDAY(startDate)) <= 7
    """
)

#identical query but changed time in final where clause
t2 = psql.sqldf( 
    """
    WITH data AS (  
       WITH project_dates AS
            (SELECT MIN(date) as startDate, projectid
            FROM project_usage
            GROUP BY projectid
            HAVING startDate > '2019-01-01') 
        SELECT pu.projectid, pd.startDate, 
            pu.date, pu2.date as yesterday, 
            pu.usage_bandwidth as bandwidth, 
            pu.usage_apiRequests as api, 
            pu.usage_users as users, pu2.usage_users as yesterday_users,
            pu.usage_documents as documents, pu2.usage_documents as yesterday_documents
        FROM project_usage pu
        INNER JOIN project_usage pu2 ON pu.date = date(pu2.date, '+1 day') AND pu.projectid = pu2.projectid
        JOIN project_dates pd ON pu.projectid = pd.projectid
    )
    SELECT *, 
        CASE api
             WHEN 0 THEN FALSE
        ELSE TRUE 
        END api_usage, 
        CASE bandwidth
            WHEN 0 THEN FALSE
        ELSE TRUE
        END bandwidth_usage, 
        CASE documents - yesterday_documents
            WHEN 0 THEN FALSE
        ELSE TRUE
        END document_change, 
        CASE users - yesterday_users
            WHEN 0 THEN FALSE
        ELSE TRUE
        END user_change
    FROM data
    WHERE (JULIANDAY(date) - JULIANDAY(startDate)) >= 60
    AND (JULIANDAY(date) - JULIANDAY(startDate)) <= 120
    """
)

#sum cases for t1
t3 = psql.sqldf( 
    """
    SELECT t1.projectid, SUM(t1.api_usage) as  api_usage_before_7, SUM(t1.bandwidth_usage) as bandwidth_usage_before_7, 
        SUM(t1.document_change) as document_change_before_7, SUM(t1.user_change) as user_change_before_7
    FROM t1
    GROUP BY projectid
    """
)

#same query as above for t2
t4 = psql.sqldf( 
    """
    SELECT projectid,
        SUM(t2.api_usage) as  api_usage_after_60, SUM(t2.bandwidth_usage) as bandwidth_usage_after_60, 
        SUM(t2.document_change) as document_change_after_60, SUM(t2.user_change) as user_change_after_60
    FROM t2
    GROUP BY t2.projectid
    """
)
#join queries together
t5 = psql.sqldf(
    """
    SELECT *
    FROM t3
    JOIN t4 ON t3.projectid = t4.projectid
    """
)
t5.corr()
#how many unique plans are there?
t1 = psql.sqldf(
    "SELECT COUNT(DISTINCT(plan_id)) as num_unique_plans FROM user_plans"
)
t1
#How many projects have more than one user?
t1 = psql.sqldf(
    "SELECT COUNT(*) FROM (SELECT projectid, COUNT(*) FROM user_plans GROUP BY projectid HAVING COUNT(*) > 1) as numProjects"
)
t1
#How many users have more than one project?
t1 = psql.sqldf(
    "SELECT COUNT(*) FROM (SELECT sanityUserId, COUNT(*) FROM user_plans GROUP BY sanityUserId HAVING COUNT(*) > 1) as numProjects"
)
t1
t1 = psql.sqldf(
    """
        SELECT MAX(date) as final_project_day, projectid 
        FROM project_usage 
        GROUP BY projectid
    """)
t1
#Understanding user retention
#Where is the knee? After how many months of usage does churn drop off?
t1 = psql.sqldf(
    """WITH final_project_date AS 
        (SELECT MAX(date) as final_project_day, projectid 
        FROM project_usage 
        GROUP BY projectid ) 
    SELECT *, MIN(startedAt)
    FROM user_plans 
    WHERE startedAt > '2019-01-01' AND startedAt < '2019-02-01'
    GROUP BY sanityUserId
    """
)
t1
#Returns number of days active for all users who have a project with projectid in both tables.
t1 = psql.sqldf(
    """WITH final_project_date AS 
        (SELECT MAX(date) as final_project_day, projectid 
        FROM project_usage 
        GROUP BY projectid ) 
    SELECT sanityUserId, MAX(final_project_day), MIN(startedAt), (JULIANDAY(MAX(final_project_day)) - JULIANDAY(SUBSTR(MIN(startedAt), 0, 11))) as days_active, user_plans.projectid
    FROM user_plans 
    JOIN final_project_date ON user_plans.projectid = final_project_date.projectid 
    GROUP BY user_plans.projectid, sanityUserId
    """
)
t1
#Returns table showing the number of projects that lasted x number of months
t1 = psql.sqldf(
    """WITH final_project_date AS 
        (SELECT COUNT(*) as num_days, projectid 
        FROM project_usage 
        GROUP BY projectid ) 
    SELECT sanityUserId, MAX(final_project_day), MIN(startedAt), (JULIANDAY(MAX(final_project_day)) - JULIANDAY(SUBSTR(MIN(startedAt), 0, 11))) as days_active, user_plans.projectid
    FROM user_plans 
    JOIN final_project_date ON user_plans.projectid = final_project_date.projectid 
    GROUP BY user_plans.projectid, sanityUserId
    """
)
t1
joined_table = psql.sqldf(
    "SELECT sanityUserId as userId, COUNT(*), MIN(date), MAX(date), SUM(usage_apiRequests), ROUND(AVG(usage_apiRequests),0) FROM project_usage JOIN user_plans ON project_usage.projectid=user_plans.projectid GROUP BY sanityUserId, user_plans.projectid LIMIT 10")
joined_table.head()

#Understanding user retention
#Where is the knee? After how many months of usage does churn drop off?
#basing on just user table
t1 = psql.sqldf(
    """
    SELECT AVG(days_active)
    FROM (SELECT sanityUserId, 
        (JULIANDAY(SUBSTR(MAX(endedAt), 0, 11)) - JULIANDAY(SUBSTR(MIN(startedAt), 0, 11))) as days_active
        FROM user_plans
        WHERE endedAt IS NOT NULL
        GROUP BY sanityUserId
        HAVING days_active > 1) as daily_users 
    """
)
t1
t1 = psql.sqldf(
    """
    SELECT sanityUserId, COUNT(*) as num_projects, (JULIANDAY(SUBSTR(MAX(endedAt), 0, 11)) - JULIANDAY(SUBSTR(MIN(startedAt), 0, 11))) as days_active
    FROM user_plans
    WHERE endedAt IS NOT NULL
    GROUP BY sanityUserId
    """
)
t1
#How does the number of projects a user has influence the number of days they have an active project?
#Results point to more projects = more active users. Potential correlation ≠ causation problem.
#It's also unclear if the same userid can work on multiple projects across organizations, which would make this mostly a moot point. 
t1 = psql.sqldf(
    """
    SELECT AVG(days_active), num_projects, COUNT(*) num_users
    FROM (SELECT sanityUserId, COUNT(*) as num_projects,
        (JULIANDAY(COALESCE((SUBSTR(MAX(endedAt), 0, 11)), '2020-05-10')) - JULIANDAY(SUBSTR(MIN(startedAt), 0, 11))) as days_active
        FROM user_plans
        WHERE sanityUserId != "service"
        AND endedAt IS NOT NULL
        GROUP BY sanityUserId) as q1
    GROUP BY num_projects
    """
)
t1
#How does the number of users on a project impact the average number of days that project remains active?
#More users on a project does increase how long that project remains active. Does not account for projects that 'roll over'
t1 = psql.sqldf(
    """
    SELECT AVG(days_active), num_users, COUNT(*) as num_projects
    FROM(   
        SELECT projectid, COUNT(*) as num_users,
        (JULIANDAY(COALESCE((SUBSTR(MAX(endedAt), 0, 11)), '2020-05-10')) - JULIANDAY(SUBSTR(MIN(startedAt), 0, 11))) as days_active
        FROM user_plans
        WHERE sanityUserId != "service"
        AND endedAt IS NOT NULL
        GROUP BY projectid) as q1
    GROUP BY num_users
    """
)
t1
