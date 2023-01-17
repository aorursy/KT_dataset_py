# Obtendo as credenciais da API do Twitter

# https://paulovasconcellos.com.br/aprenda-a-fazer-um-analisador-de-sentimentos-do-twitter-em-python-3979454f2d0d
# biblioteca utilizada para baixar dados do twitter 

!pip install tweepy
import tweepy
consumer_key =  'SUA CONSUMER KEY'

consumer_secret = 'SUA CONSUMER SECRET'

access_token = 'SEU ACCESS TOKEN'

access_token_secret = 'SEU ACCESS TOKEN SECRET'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
class MyStreamListener(tweepy.StreamListener):

    def on_status(self, status):

        print(status.text)
myStreamListener = MyStreamListener()

myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)

#myStream.filter(track=['Trump'])
myStream.filter(track=['Trump'])