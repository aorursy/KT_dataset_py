import turicreate as tc
import turicreate.aggregate as agg
products = tc.SFrame('../input/basicml-lecture1/amazon_baby.sframe')
products.head()
products['word_count'] = tc.text_analytics.count_words(products['review'])
products.head()
selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']
awesome_count = lambda x: x['word_count']['awesome'] if 'awesome' in x['word_count'] else 0
products['awesome'] = products.apply(awesome_count)
products.sort('awesome', False)
great_count = lambda x: x['word_count']['great'] if 'great' in x['word_count'] else 0 
fantastic_count = lambda x: x['word_count']['fantastic'] if 'fantastic' in x['word_count'] else 0 
amazing_count = lambda x: x['word_count']['amazing'] if 'amazing' in x['word_count'] else 0 
love_count = lambda x: x['word_count']['love'] if 'love' in x['word_count'] else 0 
horrible_count = lambda x: x['word_count']['horrible'] if 'horrible' in x['word_count'] else 0 
bad_count = lambda x: x['word_count']['bad'] if 'bad' in x['word_count'] else 0 
terrible_count = lambda x: x['word_count']['terrible'] if 'terrible' in x['word_count'] else 0 
awful_count = lambda x: x['word_count']['awful'] if 'awful' in x['word_count'] else 0 
wow_count = lambda x: x['word_count']['wow'] if 'wow' in x['word_count'] else 0 
hate_count = lambda x: x['word_count']['hate'] if 'hate' in x['word_count'] else 0
products['great'] = products.apply(great_count)
products['fantastic'] = products.apply(fantastic_count)
products['amazing'] = products.apply(amazing_count)
products['love'] = products.apply(love_count)
products['horrible'] = products.apply(horrible_count)
products['bad'] = products.apply(bad_count)
products['terrible'] = products.apply(terrible_count)
products['awful'] = products.apply(awful_count)
products['wow'] = products.apply(wow_count)
products['hate'] = products.apply(hate_count)
products.head()
sum_list = [products['awesome'].sum(), products['great'].sum(), 
            products['fantastic'].sum(), products['amazing'].sum(), 
            products['love'].sum(), products['horrible'].sum(), 
            products['bad'].sum(), products['terrible'].sum(), 
            products['awful'].sum(), products['wow'].sum(), products['hate'].sum()]
sum_dict = {'word':selected_words, 'num':sum_list}
sum_frame = tc.SFrame(sum_dict)
sum_frame
sum_frame.sort('num').print_rows(11)
#ignore all 3* reviews
products = products[products['rating']!= 3]
#positive sentiment = 4-star or 5-star reviews
products['sentiment'] = products['rating'] >= 4
products['sentiment'].show()
train_data,test_data = products.random_split(.8,seed=0)
sentiment_model = tc.logistic_classifier.create(train_data,target='sentiment', features=['word_count'], validation_set=test_data)
selected_words_model = tc.logistic_classifier.create(train_data,target='sentiment', features=selected_words, validation_set=test_data)
selected_words_model.coefficients.sort('value').print_rows(12)
sentiment_model.evaluate(test_data)
selected_words_model.evaluate(test_data)
test_data.groupby('sentiment', operations={'sum': agg.COUNT()})
print("Accuracy: %.5f" % (27976/(27976+5328)))
diaper_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']
diaper_champ_reviews
diaper_champ_reviews['predicted_sentiment'] = sentiment_model.predict(diaper_champ_reviews, output_type = 'probability')
sorted_reviews = diaper_champ_reviews.sort('predicted_sentiment', False)
sorted_reviews
selected_word_result = sorted_reviews[0:1]
selected_word_result['predicted_sentiment'] = selected_words_model.predict(selected_word_result, output_type = 'probability')
selected_word_result