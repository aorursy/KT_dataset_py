import turicreate as tc
image_data = tc.SFrame('../input/basicml-lecture1/Lecture_8/image_train_data')
test = tc.SFrame('../input/basicml-lecture1/Lecture_8/image_test_data')
image_data['label'].summary()
cat_train = image_data[ image_data['label'] == 'cat' ]
dog_train = image_data[ image_data['label'] == 'dog' ]
auto_train = image_data[ image_data['label'] == 'automobile' ]
bird_train = image_data[ image_data['label'] == 'bird' ]
cat_model = tc.nearest_neighbors.create(cat_train, features = ['deep_features'], label = 'id')
dog_model = tc.nearest_neighbors.create(dog_train, features = ['deep_features'], label = 'id')
auto_model = tc.nearest_neighbors.create(auto_train, features = ['deep_features'], label = 'id')
bird_model = tc.nearest_neighbors.create(bird_train, features = ['deep_features'], label = 'id')
cat_model.query(test[0:1])[0]
image_data[ image_data['id'] == 16289 ]['image'].explore()
dog_model.query(test[0:1])[0]
image_data[ image_data['id'] == 16976 ]['image'].explore()
cat_model.query(test[0:1])['distance'].mean()
dog_model.query(test[0:1])['distance'].mean()
dog_test = test[ test['label'] == 'dog' ]
distance = tc.SFrame({'dog-cat': cat_model.query(dog_test, k=1)['distance'], 'dog-dog': dog_model.query(dog_test, k=1)['distance'], 
                      'dog-auto': auto_model.query(dog_test, k=1)['distance'], 'dog-bird': bird_model.query(dog_test, k=1)['distance']})
distance['label'] = tc.SArray([min(distance[i], key=distance[i].get) for i in range(len(dog_test))])
distance[100:110]
len(distance[ distance['label'] == 'dog-cat' ])
len(distance[ distance['label'] == 'dog-dog' ])/len(dog_test)
