import pandas as pd



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.nlp.ex1 import *

print('Setup Complete')
# Load in the data from JSON file

data = pd.read_json('../input/nlp-course/restaurant.json')

data.head()
menu = ["Cheese Steak", "Cheesesteak", "Steak and Cheese", "Italian Combo", "Tiramisu", "Cannoli",

        "Chicken Salad", "Chicken Spinach Salad", "Meatball", "Pizza", "Pizzas", "Spaghetti",

        "Bruchetta", "Eggplant", "Italian Beef", "Purista", "Pasta", "Calzones",  "Calzone",

        "Italian Sausage", "Chicken Cutlet", "Chicken Parm", "Chicken Parmesan", "Gnocchi",

        "Chicken Pesto", "Turkey Sandwich", "Turkey Breast", "Ziti", "Portobello", "Reuben",

        "Mozzarella Caprese",  "Corned Beef", "Garlic Bread", "Pastrami", "Roast Beef",

        "Tuna Salad", "Lasagna", "Artichoke Salad", "Fettuccini Alfredo", "Chicken Parmigiana",

        "Grilled Veggie", "Grilled Veggies", "Grilled Vegetable", "Mac and Cheese", "Macaroni",  

         "Prosciutto", "Salami"]
# Check your answer (Run this code cell to receive credit!)

q_1.solution()
import spacy

from spacy.matcher import PhraseMatcher



index_of_review_to_test_on = 14

text_to_test_on = data.text.iloc[index_of_review_to_test_on]



# Load the SpaCy model

tokenizer = spacy.blank('en')



# Create the tokenized version of text_to_test_on

review_doc = ____



# Create the PhraseMatcher object. The tokenizer is the first argument. Use attr = 'LOWER' to make consistent capitalization

matcher = PhraseMatcher(tokenizer.vocab, attr='LOWER')



# Create a list of tokens for each item in the menu

menu_tokens_list = [____ for item in menu]



# Add the item patterns to the matcher. 

# Look at https://spacy.io/api/phrasematcher#add in the docs for help with this step

# Then uncomment the lines below 



# 

#matcher.add("MENU",            # Just a name for the set of rules we're matching to

#            None,              # Special actions to take on matched words

#            ____  

#           )



# Find matches in the review_doc

# matches = ____



q_2.check()
# Lines below will give you a hint or solution code

#q_2.hint()

#q_2.solution()
# for match in matches:

#    print(f"Token number {match[1]}: {review_doc[match[1]:match[2]]}")
from collections import defaultdict



# item_ratings is a dictionary of lists. If a key doesn't exist in item_ratings,

# the key is added with an empty list as the value.

item_ratings = defaultdict(list)



for idx, review in data.iterrows():

    doc = ____

    # Using the matcher from the previous exercise

    matches = ____

    

    # Create a set of the items found in the review text

    found_items = ____

    

    # Update item_ratings with rating for each item in found_items

    # Transform the item strings to lowercase to make it case insensitive

    ____



q_3.check()
# Lines below will give you a hint or solution code

#q_3.hint()

#q_3.solution()
# Calculate the mean ratings for each menu item as a dictionary

mean_ratings = ____



# Find the worst item, and write it as a string in worst_text. This can be multiple lines of code if you want.

worst_item = ____



q_4.check()
# Lines below will give you a hint or solution code

#q_4.hint()

#q_4.solution()
# After implementing the above cell, uncomment and run this to print 

# out the worst items. Otherwise you'll get an error.



# for item in worst_items:

#     print(f"{item:>25}{mean_ratings[item]:>10.3f}")
worst_item
counts = {item: len(ratings) for item, ratings in item_ratings.items()}



item_counts = sorted(counts, key=counts.get, reverse=True)

for item in item_counts:

    print(f"{item:>25}{counts[item]:>5}")
sorted_ratings = sorted(mean_ratings, key=mean_ratings.get)



print("Worst rated menu items:")

for item in sorted_ratings[:10]:

    print(f"{item:20} Ave rating: {mean_ratings[item]:.2f} \tcount: {counts[item]}")

    

print("\n\nBest rated menu items:")

for item in sorted_ratings[-10:]:

    print(f"{item:20} Ave rating: {mean_ratings[item]:.2f} \tcount: {counts[item]}")
# Check your answer (Run this code cell to receive credit!)

q_5.solution()