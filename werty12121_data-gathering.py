!pip install tweepy

!pip install bs4
from os import listdir,mkdir

from os.path import isfile, join

import re

import threading

from dataclasses import dataclass



import pandas as pd

import requests

import tweepy

from bs4 import BeautifulSoup as bs
consumer_key = 'sT3y6tzKtQirprWFkz9cMDiDi'

consumer_secret = 'mRnXj7cn4xH8N2UkcKZTOieMkngEOf7yGxB65GJ4CUnIaB7WuM'

access_token = '928670739368169472-NFmzdqYS5FRTQY0SfnzcxlYqbkiykwt'

access_token_secret = 'L7pY2kKpU4wMfV6lbdEGqgEhYuaCK9i1zMxdbTlzhQt6H'



auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_token, access_token_secret)



api = tweepy.API(auth)

def get_tweets(query: str, count: int = 10, save: bool = True, folder: str = '.'):

    try:

        data = [x.full_text.encode("utf-8") for x in tweepy.Cursor(api.search, q=query + " -filter:retweets", lang='en', include_entities=False, tweet_mode='extended', wait_on_rate_limit=True).items(count)]

        data = pd.DataFrame(data, columns=['text'])

        if save:

            data.to_csv(folder + '/raw_tweets ' + re.sub(r'[^a-zA-Z\s]', '', query) + '.csv')

        return data

    except tweepy.TweepError as e:

        print(e.reason)
@dataclass

class TargetWebsite:

    url: str

    how_deep: int = 3

    contains: str = None

    main_site_address: str = None

    excluded: list = None

    is_question_mark_allowed: bool = True

    split_by_dot: bool = False
class WebsiteCrawler:



    def __init__(self, target: TargetWebsite):

        self.url = target.url

        self.how_deep = target.how_deep

        self.contains = target.contains

        self.main_site_address = target.main_site_address

        self.excluded = target.excluded if target.excluded is not None else []

        self.is_question_mark_allowed = target.is_question_mark_allowed

        self.split_by_dot = target.split_by_dot

        self.gathered_data = pd.DataFrame(columns=['text'])

        self.already_crawled = []

        self.site_to_crawl = []

        self.site_to_crawl_deep = []



    @staticmethod

    def tag_visible(element):

        return element.parent.name not in ['style', 'script', 'head', 'title', 'meta', '[document]']



    @staticmethod

    def display_visible_html_using_re(text: str):

        return re.sub(r'(<.*?>)', "", text)



    @staticmethod

    def get_text_and_links_from_web(url: str):

        try:

            request = requests.get(url)

            soup = bs(request.text, 'html.parser')

            text = soup.findAll(text=True)

            links = soup.findAll('a', attrs={'href': re.compile(".*")})

            return text, links

        except requests.exceptions.RequestException as e:

#             print(e)

            return [], []



    def start_crawling(self, directory: str, min_text_length: int = 20, text_must_have: list = None):

        self.crawl_site(self.url, directory, self.how_deep, min_text_length, text_must_have)

        self.save_data_to_csv(directory)



    def save_data_to_csv(self, directory: str):

#         print("-DATA-SAVE-")

        self.gathered_data.drop_duplicates(subset='text', inplace=True)

        self.gathered_data.to_csv(directory, index=False)



    def clean_url(self, url: str, mother_url: str):

        if len(url) <= 0:

            return None

        mother_url = mother_url + '/' if len(mother_url.split('/')) < 4 else mother_url

        url = '/'.join(mother_url.split('/')[:-1]) + url[1:] if url[0] == '.' else url

        if url[0] == '/' or 'http' not in url:

            url = url if url[0] == '/' else '/' + url

            if self.main_site_address is None:

                url = '/'.join(mother_url.split('/')[:3]) + url

            else:

                url = '/'.join(self.main_site_address.split('/')[:3]) + url

        url = url.split('#')[0]

        url = url.split('?')[0] if not self.is_question_mark_allowed else url

        url = url.lstrip()

        url = url.rstrip()

        contain = (self.contains is not None and self.contains in url) or self.contains is None

        if url not in self.already_crawled and '.jpg' not in url and '.mp4' not in url and '.png' not in url and contain:

            return url if not any(ex in url for ex in self.excluded) else None



    def crawl_site(self, url: str, directory: str, how_deep: int = 5, min_text_length: int = 20, text_must_have: list = None):



        how_many_site_visited = 0

        self.site_to_crawl.append(url)

        self.site_to_crawl_deep.append(how_deep)

        self.already_crawled.append(url)



        while len(self.site_to_crawl) > 0:



            how_many_site_visited += 1

            if how_many_site_visited % 100 == 0:

                self.save_data_to_csv(directory)



            url_to_crawl = self.site_to_crawl.pop(0)

            how_deep_to_crawl = self.site_to_crawl_deep.pop(0)

#             print(url_to_crawl, how_deep_to_crawl)



            text_unfiltered, links = self.get_text_and_links_from_web(url_to_crawl)

            visible_texts = filter(self.tag_visible, text_unfiltered)

            for text in visible_texts:

                text = self.display_visible_html_using_re(text)

                text_must_have_condition = False

                if text_must_have is not None:

                    for t in text_must_have:

                        if t in text.lower():

                            text_must_have_condition = True

                if len(text) >= min_text_length and (text_must_have_condition or text_must_have is None):

                    if self.split_by_dot or len(text.split()) > 250:

                        text = pd.DataFrame(text.split('. '), columns=['text'])

                        self.gathered_data = self.gathered_data.append([text])

                    else:

                        self.gathered_data = self.gathered_data.append({'text': text}, ignore_index=True)

            for link in links:

                url_to_crawl_after = self.clean_url(link.get('href'), url_to_crawl)

                if url_to_crawl_after is not None and how_deep_to_crawl != 0:

                    self.site_to_crawl.append(url_to_crawl_after)

                    self.site_to_crawl_deep.append(how_deep_to_crawl - 1)

                    self.already_crawled.append(url_to_crawl_after)



        return how_many_site_visited
class DataGatherer:

    def __init__(self, classes: list, sites: list = None, twitter: list = None, text_must_have: list = None):

        self.classes = classes

        self.sites = sites

        self.twitter = twitter

        self.text_must_have = text_must_have



    def crawl_sites(self, sites: list, class_name: str):

        threads = []

        if sites is None:

            return

        for site in sites:

            thread = threading.Thread(target=lambda: WebsiteCrawler(site).start_crawling(class_name + '/' + re.sub(r'[^a-zA-Z\s]', '', site.url) + '.csv', text_must_have=self.text_must_have))

            thread.start()

            threads.append(thread)

        for thread in threads:

            thread.join()

        return True



    @staticmethod

    def get_tweets(tweeter_queries: list, class_name: str, twitter_max: int):

        if tweeter_queries is None:

            return

        for query in tweeter_queries:

            get_tweets(query, folder=class_name, count=twitter_max)



    def gather_data(self, twitter_max: int = 1000):

        for class_name, class_sites, class_twitter in zip(self.classes, self.sites, self.twitter):

            try:

                mkdir(class_name)

            except FileExistsError as e:

#                 print(e)

                pass

            print("Starting to gather data from web... " + class_name)

            self.crawl_sites(class_sites, class_name)

            print("Web data gathered. " + class_name)

            print("Started gathering data from twitter..." + class_name)

            self.get_tweets(class_twitter, class_name, twitter_max)

            print("Twitter data gathered. " + class_name)

class DataCleaner:

    def __init__(self, classes: list, exclusions: list = None, synonyms: list = None, inplaced_synonyms: str = None, text_must_haves: list = None, text_min_length: int = 20):

        self.classes = classes

        self.exclusions = exclusions

        self.synonyms = synonyms

        self.inplaced_synonyms = inplaced_synonyms

        self.text_must_haves = text_must_haves

        self.text_min_length = text_min_length



    @staticmethod

    def read_data_from_directory(directory: str):

        files = [f for f in listdir(directory) if isfile(join(directory, f))]

        data = pd.DataFrame(columns=['text'])

        for file in files:

            if '.csv' in file:

                data_in_file = pd.read_csv(directory + file)

                data = pd.concat([data, data_in_file], ignore_index=True)

            else:

                data_in_file = open(directory + file, "r", encoding="utf8")

                text = data_in_file.readlines()

                ds = pd.DataFrame(data=text, columns=['text'])

                data = data.append(ds, ignore_index=True)

                data_in_file.close()

        return data



    @staticmethod

    def clean_text(text: str):

        if type(text) != str:

            return " "

        text = re.sub(r'@\w+', '', text)  # mentions

        text = re.sub(r'http.?://[^\s]+[\s]?', '', text)  # urls

        text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # symbols and digits

        text = re.sub(r'\brt\b', '', text)  # retweets

        text = re.sub(r'\bgt\b', '', text)

        text = re.sub(r'\n', ' ', text)  # Enters

        text = re.sub(r'(\b[A-Za-z]\b)', '', text)  # one letter words

        text = re.sub(r'\s+', ' ', text)  # extra white spaces

        text = text.lstrip()

        text = text.rstrip()

        text = text.lower()

        return text



    @staticmethod

    def remove_excluded(text: str, excluded_words: list):

        for excluded_word in excluded_words:

            if excluded_word in text:

                return " "

        return text



    def remove_synonyms(self, text: str):

        for synonym in self.synonyms:

            text = re.sub(r'\b%s\b' % synonym, self.inplaced_synonyms, text)

        return text



    @staticmethod

    def check_if_text_contain_some_of_must_have_words(text: str, must_have: list):

        if must_have is None:

            return text

        for word in must_have:

            if word in text:

                return text

        return " "



    @staticmethod

    def remove_by_texts(text: str):

        if len(text.split()) < 1:

            return " "

        return text if not (text.split()[0] == 'by' and len(text.split()) < 10) else " "



    def check_text_length(self, text: str):

        return text if len(text) >= self.text_min_length else " "



    def clean(self, save: bool = True):

        for clas, class_excluded_words, text_must_have in zip(self.classes, self.exclusions, self.text_must_haves):

            data = self.read_data_from_directory(clas + '/')

            print(clas + '. Records before cleaning: ' + str(len(data)))

            data.text = data.text.apply(lambda elem: self.clean_text(elem))

            data.text = data.text.apply(lambda elem: self.remove_excluded(elem, class_excluded_words))

            data.text = data.text.apply(lambda elem: self.remove_synonyms(elem))

            data.text = data.text.apply(lambda elem: self.check_if_text_contain_some_of_must_have_words(elem, text_must_have))

            data.text = data.text.apply(lambda elem: self.remove_by_texts(elem))

            data.text = data.text.apply(lambda elem: self.check_text_length(elem))

            data.drop_duplicates(subset=['text'], keep='first', inplace=True)

            data = pd.DataFrame(data.text, columns=['text'])

            print(clas + '. Records after cleaning: ' + str(len(data)))

            if save:

                data.to_csv(clas + '.csv', index=False)   

            return

sites_to_crawl_animal = [

    TargetWebsite(

        url='https://en.wikipedia.org/wiki/Mouse',

        how_deep=0),

    TargetWebsite(

        url='https://en.wikipedia.org/wiki/Rat',

        how_deep=0),

    TargetWebsite(

        url='https://en.wikipedia.org/wiki/Chinchilla',

        how_deep=0),

    TargetWebsite(

        url='https://en.wikipedia.org/wiki/Murinae',

        how_deep=0),

    TargetWebsite(

        url='https://en.wikipedia.org/wiki/Rodent',

        how_deep=0),

    TargetWebsite(

        url='https://en.wikipedia.org/wiki/Jerboa',

        how_deep=0),

    TargetWebsite(

        url='http://www.petmousefanciers.com/',

        how_deep=4,

        contains='www.petmousefanciers.com',

        main_site_address='http://www.petmousefanciers.com',

        excluded=['http://www.petmousefanciers.com/abuse'],

        is_question_mark_allowed=False),

    TargetWebsite(

        url='http://www.allaboutmice.co.uk',

        how_deep=-1,

        contains='http://www.allaboutmice.co.uk',

        main_site_address='http://www.allaboutmice.co.uk'),

    TargetWebsite(

        url='http://themouseconnection.forumotion.com/',

        how_deep=-1,

        contains='themouseconnection.forumotion.com',

        main_site_address='http://themouseconnection.forumotion.com')

]

sites_to_crawl_computer = [

    TargetWebsite(

        url='https://en.wikipedia.org/wiki/Computer_mouse',

        how_deep=0),

    TargetWebsite(

        url='https://www.overclock.net/forum/375-mice',

        contains='/forum/375-mice',

        how_deep=4,

        excluded=['?styleid'],

        main_site_address='https://www.overclock.net'),

    TargetWebsite(

        url='https://forums.tomsguide.com/tags/mice/',

        how_deep=2,

        contains='forums.tomsguide.com',

        main_site_address='https://forums.tomsguide.com'),

    TargetWebsite(

        url='https://hardforum.com/forums/mice-and-keyboards.124/',

        how_deep=2,

        contains='hardforum.com',

        main_site_address='https://hardforum.com/'),

    TargetWebsite(

        url='https://www.walmart.com/browse/computer-keyboards-mice/computer-mouse-mouse-pads/3944_3951_132959_1008621_4842284',

        how_deep=2,

        contains='www.walmart.com',

        main_site_address='https://www.walmart.com'),

    TargetWebsite(

        url='https://www.noelleeming.co.nz/shop/computers-tablets/accessories/computer-mice',

        how_deep=-1,

        contains='/shop/computers-tablets/accessories/computer-mice',

        main_site_address='https://www.noelleeming.co.nz'),

    TargetWebsite(

        url='https://www.currys.co.uk/gbuk/computing-accessories/computer-accessories/mice-and-keyboards/mice/318_3058_30060_xx_xx/xx-criteria.html',

        how_deep=-1,

        contains='/gbuk/computing-accessories/computer-accessories/mice-and-keyboards/mice/',

        main_site_address='https://www.currys.co.uk')

]

twitter_search_queries_animal = [

    'animal mouse -computer -computers -pc -pcs -keyboard',

    'animal mice -computer -computers -pc -pcs -keyboard',

    'animal chinchilla -computer -computers -pc -pcs -keyboard',

    'animal rat -computer -computers -pc -pcs -keyboard',

    'animal rodent -computer -computers -pc -pcs -keyboard',

    'dead rats -computer -computers -pc -pcs -keyboard',

    'fieldmouse -computer -computers -pc -pcs -keyboard',

    'jerboa -computer -computers -pc -pcs -keyboard',

    'little mouse -computer -computers -pc -pcs -keyboard',

    'mouse pet -computer -computers -pc -pcs -keyboard',

    'murine animal -computer -computers -pc -pcs -keyboard',

    'mus musculus',

    'rat -computer -computers -pc -pcs -keyboard'

]

twitter_search_queries_computer = [

    'cable mouse -animal',

    'computer mice -animal',

    'computer mouse -animal',

    'device mouse -animal',

    'gaming mouse -animal',

    'gaming mice -animal',

    'hardware mouse -animal',

    'mouse controller -animal',

    'peripheral mouse -animal',

    '"plug and play" mouse -animal',

    'ps2 mouse -animal',

    'usb mouse -animal',

    'wireless mouse -animal'

]

sites_to_crawl = [sites_to_crawl_animal, sites_to_crawl_computer]

twitter_search_queries = [twitter_search_queries_animal, twitter_search_queries_computer]

classes = ['animal', 'computer']

text_must_have = ['mice', 'mouse', ' rat ', ' rats ', 'rodent']



data_gatherer = DataGatherer(classes, sites_to_crawl, twitter_search_queries, text_must_have)

data_gatherer.gather_data()
exclusions_animal = ['computer', ' pc ', 'keyboard']

exclusions_computer = ['animal']

exclusions = [exclusions_animal, exclusions_computer]



synonyms = ['mice', 'mices', 'rat', 'rats', 'mouserat', 'mouserats', 'mousing', 'chinchilla', 'chinchillas', 'mousie', 'mousies', 'mus']

inplaced_synonyms = 'mouse'

text_must_haves = [['mouse'], ['mouse']]



data_cleaner = DataCleaner(classes, exclusions, synonyms, inplaced_synonyms, text_must_haves)

data_cleaner.clean()