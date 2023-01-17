# To document how to use python to connect Power BI with AWS DynamoDB

# python -m pip install --upgrade pip

# !pip install boto3



# https://pypi.org/project/boto3/

# Boto3 is the Amazon Web Services (AWS) Software Development Kit (SDK) for Python, 

# which allows Python developers to write software that makes use of services like Amazon S3 

# and Amazon EC2. You can find the latest, most up to date, documentation at our doc site, 

# including a list of services that are supported.



print("""

# install library and set default region

import boto3

import pandas as pd



# set up default region

aws_region_name='xxxx'



# set up specical credentials

aws_access_key='xxxxx'

aws_secret_access_key='xxxxxx'

 

tableName='xxxxxxxxxxx'

 

dynamodb=boto3.resource('dynamodb',

                        aws_access_key_id=aws_access_key,

                        aws_secret_access_key=aws_secret_access_key,

                        region_name=aws_region_name

)

 

table=dynamodb.Table(tableName)



response=table.scan()

items=response['Items']

 

while 'LastEvaluatedKey' in response:

    response=table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])

    item.extend(response['Items'])

     

df.pd.DataFrame(items) """)
