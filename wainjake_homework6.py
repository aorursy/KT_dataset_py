import turicreate
song_data = turicreate.SFrame('../input/basicml-lecture1/song_data.sframe')
song_data
west = song_data[ song_data['artist'] == 'Kanye West' ]
fighters = song_data[ song_data['artist'] == 'Foo Fighters' ]
swift = song_data[ song_data['artist'] == 'Taylor Swift' ]
gaga = song_data[ song_data['artist'] == 'Lady GaGa' ]
west_user = west['user_id'].unique()
fighters_user = fighters['user_id'].unique()
swift_user = swift['user_id'].unique()
gaga_user = gaga['user_id'].unique()
len(west_user)
len(fighters_user)
len(swift_user)
len(gaga_user)
grouped = song_data.groupby('artist', operations={'total_count': turicreate.aggregate.SUM('listen_count')})
grouped
grouped.sort('total_count').head()
grouped.sort('total_count').tail()
train_data,test_data = song_data.random_split(.8,seed=0)
personalized_model = turicreate.item_similarity_recommender.create(train_data,user_id = 'user_id',item_id = 'song')
subset_test_users = test_data['user_id'].unique()[0:10000]
rcd = personalized_model.recommend(subset_test_users,k=1)
rcd
rcd.groupby('song', operations={'recommend_count': turicreate.aggregate.COUNT()}).sort('recommend_count', False)
