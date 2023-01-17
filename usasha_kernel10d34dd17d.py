import pickle
import Levenshtein

with open('../input/artist_model.pkl', 'rb') as f:
    model = pickle.load(f)
artist_name = 'лениград'

[name 
 for name, score in model.wv.most_similar(artist_name.lower(), topn=30) 
 if Levenshtein.ratio(artist_name, name) < 0.5][:10]