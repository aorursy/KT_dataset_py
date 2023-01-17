!pip install --user google-api-core==1.11.0

!pip install --user google-cloud-bigquery==1.12.1
!pip show google-cloud-bigquery

import os

from google.auth import credentials

from google.auth.exceptions import RefreshError

from google.cloud import bigquery

from google.cloud.exceptions import Forbidden

from google.cloud.bigquery._http import Connection

from kaggle_secrets import UserSecretsClient





class _DataProxyConnection(Connection):

    """Custom Connection class used to proxy the BigQuery client to Kaggle's data proxy."""



    API_BASE_URL = os.getenv("KAGGLE_DATA_PROXY_URL")



    def __init__(self, client):

        super().__init__(client)

        print(dir(self))

        self._EXTRA_HEADERS["X-KAGGLE-PROXY-DATA"] = os.getenv("KAGGLE_DATA_PROXY_TOKEN")



    def api_request(self, *args, **kwargs):

        """Wrap Connection.api_request in order to handle errors gracefully.

        """

        try:

            print("foo!")

            print(args)

            print(kwargs)

            return super().api_request(*args, **kwargs)

            print(os.getenv("KAGGLE_DATA_PROXY_URL"))

            print(self.extra_headers)

            print("bar!")

        except Forbidden as e:

            msg = ("Permission denied using Kaggle's public BigQuery integration. "

                   "Did you mean to select a BigQuery account in the Kernels Settings sidebar?")

            print(msg)

            Log.info(msg)

            raise e





class PublicBigqueryClient(bigquery.client.Client):

    """A modified BigQuery client that routes requests using Kaggle's Data Proxy to provide free access to Public Datasets.

    Example usage:

    from kaggle import PublicBigqueryClient

    client = PublicBigqueryClient()

    """



    def __init__(self, *args, **kwargs):

        data_proxy_project = os.getenv("KAGGLE_DATA_PROXY_PROJECT")

        anon_credentials = credentials.AnonymousCredentials()

        anon_credentials.refresh = lambda *args: None

        super().__init__(

            project=data_proxy_project, credentials=anon_credentials, *args, **kwargs

        )

        # TODO: Remove this once https://github.com/googleapis/google-cloud-python/issues/7122 is implemented.

        self._connection = _DataProxyConnection(self)
# Create client object to access database

print(bigquery.__version__)

client = PublicBigqueryClient()



query = """

        SELECT taxi_id,

            trip_start_timestamp,

            trip_end_timestamp,

            trip_seconds,

            AVG(trip_seconds) 

                OVER (

                      PARTITION BY taxi_id

                      ORDER BY trip_start_timestamp

                     ) AS trip_seconds_avg

        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

        WHERE trip_start_timestamp BETWEEN '2017-05-01' AND '2017-05-02'

        """



result = client.query(query).result().to_dataframe()

result.head()