import turicreate as tc
people = tc.SFrame('../input/basicml-lecture1/people_wiki.sframe')
people
selected_one = people[people['name'] == 'Elton John']
selected_one
selected_one['word_count'] = tc.text_analytics.count_words(selected_one['text'])
selected_one
sorted(selected_one['word_count'][0].items(), key = lambda item:item[1], reverse=True)[0:5]
people['tfidf'] = tc.text_analytics.tf_idf(people['text'])
people.head()
selected_one = people[people['name'] == 'Elton John']
selected_one
sorted(selected_one['tfidf'][0].items(), key = lambda item:item[1], reverse=True)[0:5]
elton = people[people['name'] == 'Elton John']
beckham = people[people['name'] == 'Victoria Beckham']
mcCartney = people[people['name'] == 'Paul McCartney']
tc.distances.cosine(elton['tfidf'][0],beckham['tfidf'][0])
tc.distances.cosine(elton['tfidf'][0],mcCartney['tfidf'][0])
tfidf_model = tc.nearest_neighbors.create(people,features=['tfidf'],label='name', distance='cosine')
people['word_count'] = tc.text_analytics.count_words(people['text'])
word_count_model = tc.nearest_neighbors.create(people,features=['word_count'],label='name', distance='cosine')
elton = people[people['name'] == 'Elton John']
beckham = people[people['name'] == 'Victoria Beckham']
mcCartney = people[people['name'] == 'Paul McCartney']
tfidf_model.query(elton)
word_count_model.query(elton)
tfidf_model.query(beckham)
word_count_model.query(beckham)
tfidf_model.query(mcCartney)
word_count_model.query(mcCartney)