from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

gcs_service_account = user_secrets.get_secret("gcs_service_account")

print(gcs_service_account, file=open("/tmp/key.json", "w"))
!gcloud auth activate-service-account --key-file /tmp/key.json
!gsutil ls gs://foo/bar