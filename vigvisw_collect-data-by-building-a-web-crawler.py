# import the require libraries 

from bs4 import BeautifulSoup

from urllib import request

import re

import numpy as np

import time

import os



# uncomment if you are using this on Colab

# from google.colab import files





# the domain name of the website we are crawling is a global variable used by the crawler

domain_name = 'https://www.gsmarena.com/'
# if using Google Colab, I use a list to collect the log of any issues

# you give give a shot at refactoring this function to use the logging module

debug_collector = []



def collect_debug(error):

  '''A function for logging any unexpected behaviour'''

  global debug_collector

#   print(error)

  debug_collector.append(error)
# STEP 1



# get meta data and links to all the makers in GSMArena  

def get_maker_links(url):

  '''A function for getting links to all makers from the GSMArena makers list.

 

     Takes in the url of the list of makers and returns a list of lists of the form

     [[maker_name_1, maker_name_1, num_devices_1, maker_link_1],.....]

  '''

  # get the maker page and read it

  page = request.urlopen(url)

  html = page.read()

  

   # create the BeautifulSoup(bs) object 

  bs = BeautifulSoup(html, 'html.parser')



  # find the div-tag which contains the table

  table = bs.findChild('div', class_='st-text').table



  # if there is no table in the seed page, log it 

  if not table:

    error = 'Maker Page Error: has no brands table| function_name: {}| url: {}'.format(function_name, url)

    collect_debug(error)

  # if the table is present then get the the maker information from it

  else:

    # inside each table, the data in is maker is stored under td-tag which we collect using a list

    rows = table.findChildren('td')

    rows_collector = []

    # takes in data as [index, maker_name, link, #phones]

    for maker_id, row in enumerate(rows):

    # get the maker name and maker link. if there is no a-tag, collect a log

      row_a_tag = row.a

      if not row_a_tag:

        error = 'Maker Page Error: no row_a_tag| function_name: {}| url: {}| row_num: {}'.format(function_name, url, n)

        collect_debug(error)

      else:

        maker_link = domain_name + row_a_tag['href']

        # use the stripped_strings generator to get a tuple of the maker name and num of devices 

        # if you are wondering what is going in the line above, please check out comprehensions in Python

        maker_name, num_devices = (item for item in row_a_tag.stripped_strings)



        # extract the numerical portion of num_devices and convert it into an integer

        num_devices = re.findall(re.compile('\d+'), num_devices)[0]

        num_devices = int(num_devices)

        # append all of the maker data to the rows_collector and return it

        rows_collector.append([maker_id, maker_name, num_devices, maker_link])

  return rows_collector
seed_path = 'makers.php3'

seed_url = domain_name + seed_path 



# test out the function that we just created

maker_list = get_maker_links(seed_url)

print(maker_list)
# define a function for giving us the nav_page_num 1 of a maker given their name

def get_makers_link(maker_name):

  '''A function for getting the link to a maker given the maker's name.

  

     Takes in a maker's name, say 'Samsung' returns 'https://www.gsmarena.com/samsung-phones-9.php'

     Maker name is case insenstive.

  '''

  # go through the maker_list and find list_item[1], i.e maker_name

  global maker_list

  if not maker_list:

      error = 'No maker_list!'

      collect_debug(error)

  else:

    for list_item in maker_list:

      if maker_name.lower() == list_item[1].lower():

        print('maker_link called for {} \n{}'.format(maker_name, list_item[-1]))

        

  

# test it out

maker_name = 'Samsung'

get_makers_link(maker_name)



maker_name = 'samSung'

get_makers_link(maker_name)
# since we call a webpage and get the bs object of the page a lot, we can define a function to make it easier  

def get_bs(url, parser='html.parser'):

  '''A function for returning the BeautifulSoup object of a webpage given its url

  

     Uses 'html.parser' by defualt and can be modified using the optional argument 'parser'

  '''

  # return the bs onject for a given webpage

  page = request.urlopen(url)

  html = page.read()

  bs = BeautifulSoup(html, parser)

  return bs
# since Samsung is the maker we want from maker_ist, we write a function to get the largest maker

def get_largest_maker():

  '''A function that returns the maker data corresponding to largest maker from the maker_list'''

  global maker_list

  # compare and set the num_devices under each maker against this variable if num_devices > is_largest

  is_largest = 0

  maker_id = None

  # iterate through all the maker's in maker_list and return the largest maker

  for maker in maker_list:

    num_devices = maker[2]

    if num_devices > is_largest:

      is_largest = num_devices

      maker_id = maker[0]

  return maker_list[maker_id]



# test it out

maker = get_largest_maker()

print(maker)
# STEP 2



# iterate through each maker in the maker list and apply this function over the maker to get the nav_page_links

def get_nav_page_links(maker):

  '''A function for getting all the nav pages under a given maker

  

    This function takes in a maker_list item of the form [maker_id, maker_name, num_devices, maker_link]

    Returns a dict of the form {nav_page_num:nav_page_link} for the maker

  '''

  # unpack the items in the list

  maker_id, maker_name, num_devices, maker_link = maker

  # a dictionary that will be used to collect all the nav_pages for a given maker

  maker_nav_pages = {}

  # first add the landing page as nav_page_num = 1

  maker_nav_pages[1] = maker_link

  # get the maker's page

  bs = get_bs(maker_link)



  # find the div-tag containing the nav_pages

  nav_pages = bs.findChild('div', class_='nav-pages')

  # if the maker has no nav_pages, which is possible, collect it for logging

  if not nav_pages:

    error = '{} does not have nav_pages| maker_link: {}'.format(maker_name, maker_link)

    collect_debug(error)

  # otherwise we can get a list of all the nav_pages 

  else:

    # insde this div tag, the pages we want are inside a-tags

    nav_pages = nav_pages.findChildren('a', recursive=False)

    for nav_page_num, nav_page in enumerate(nav_pages):

        # nav_page_num needs to be offset by 2 before using as a key to add the nav page link

        maker_nav_pages[nav_page_num + 2] = domain_name + nav_page['href']

  return maker_nav_pages



# test it out

nav_page_links = get_nav_page_links(maker)  

print(nav_page_links)  
# STEP 3



# get the information about the devices present in a nav_page by iterating through the all the devices on that page

# all the devices by a given maker are collected as elements in dictionary of the form

# {Samsung:[device_1_data, device_2_data,.......], Acer:[device_1_data, device_2_data,....],....}



# we also need to define a couple of global variables, which are results from the earlier functions

devices_collector = {}

maker_name = maker[1]

maker_link = maker[-1]



# for nav_page in nav_pages, we will iterate through this function

nav_page_links = get_nav_page_links(maker) 



# collect all devices by the makers in the devices_collector dict using maker_name's as the key

devices_collector[maker_name] = [] 

def get_device_links(nav_page_link, devices_collector, maker_name):

  '''A function to get the device links and device info for all devices in a nav page

     This function will be called for every device in GSMArena when used with the crawler

  '''

  # unpack the 

#   global devices_collector, maker_name

  # get the nav_page

  bs = get_bs(nav_page_link)



  # get the list items under the div-tag with the class name 'makers'

  devices = bs.findChild('div', class_='makers').ul

  devices = devices.findChildren('li', recrusive=False)



  # iterate through each device and collect the device_name, device_info, device_img_link, device_link

  page_device_collector = []

  for device_num, device in enumerate(devices):

    device_name = device.get_text()

    # we cannot collect the link for a device if it does not have an a-tag

    if not device.a:

        error = "{} does not have a link : nav_page: {}| maker_name {}: ".format(device_name, nav_page_link , maker_name)

        collect_debug.append(error)

    else:

      device_link = domain_name + device.a['href']

      # img_link, and title are stored in the img tag

      img_tag = device.a.findChild('img')

      if not img_tag:

        error = "{} does not have a img_tag| nav_page: {}| maker_name: {}".format(device_name, nav_page_link , maker_name)

        collect_debug.append(error)

      else:

        device_img_link = img_tag['src']

        device_info = img_tag['title']

    page_device_collector.append([device_name, device_info, device_img_link, device_link]) 

  # concat the device info from this nav page onto what is already present on the list

  devices_collector[maker_name] += page_device_collector



# test it out 

for nav_page_num, nav_page_link in nav_page_links.items():

  get_device_links(nav_page_link, devices_collector, maker_name)

  

print(devices_collector.keys())

print(devices_collector['Samsung'])

print(devices_collector['Samsung'].__len__())
# get the data for Samsung Galaxy S10 (devices[3])so that we can build a sample crawler for a device 

devices = devices_collector['Samsung']

device = devices[3]



# to get the device information, the functions take in each of these device info as attributes

device_link = device[-1]

device_name = device[0]



# a collector dict which consolidates all features, including the banner for a device

specs_collector = {}
# get the spec_sheet for a device from the device_link

def get_device_specs(bs, specs_collector, device_name, device_link):

  '''A function for findinf the specs tabele of a device given a bs object of the device webpage'''

  # the specs are stored inside individual tables, so find them all

  specs_tables = bs.findChildren('table')

  if not specs_tables:

    error = '{} has no specs_tables| device_link: {}'.format(device_name, device_link)

    collect_debug.append(error)

  # if the phone does have a spec-list

  else:

    # get the spec category like 'Network', 'Launch', 'Memory', 'Battery', ...

    for table in specs_tables:

      # find all the rows in the table 

      table_rows = table.findChildren('tr', recursive=False)

      # each table will only hacve one child th-tag, i.e a header

      # this header of the table is the name of the spec

      table_header = table.findChild('th').get_text(strip=True)



      # for row in tables: if the class = 'ttl' or 'nfo', it is a column in the table

      # 'ttl' tags correspond to a potential feature that we could extract such as Dimension, Weight, Date Announced, etc..

      # 'nfo' coresponds to a actual data point corresponding to the 'ttl' feature

      ttl_collector = {}

      for row_num, row in enumerate(table_rows):

        ttl_tag = row.findChild('td', class_='ttl')

        nfo_tag = row.findChild('td', class_='nfo')



        # if neither the ttl_tag or nfo tag are present, we want to log it

        if (not ttl_tag) or (not nfo_tag):

          error = '{} has ttl-tag OR nfo-tag| device_link:{}'.format(device_name, device_link)

          collect_debug(error)

          # we also want to set the text to NaN if a column is empty so that it can later be processed easily 

          ttl_tag_text = np.NaN

          nfo_tag_text = np.NaN

        # if either the ttl_tag or nfo tag are present, we want to collect them and log any missing values

        else:

          if not ttl_tag:

            error = '{} has no ttl-tag| device_link:{}'.format(device_name, device_link)

            collect_debug(error)

          else:

            ttl_tag_text = ttl_tag.get_text(strip=True)

            if ttl_tag_text == '\xa0' or ttl_tag_text == '':

              ttl_tag_text = np.NaN



          if not nfo_tag:

            error = 'No nfo-tag: {}: {}: {}'.format(n, link, row)

            collect_debug(error_mess)

          else:

            nfo_tag_text = nfo_tag.get_text(strip=True)

            if nfo_tag_text == '\xa0' or nfo_tag_text == '':

              nfo_tag_text = np.NaN

        # add the values of the ttl-tag and nfo-tag as key value pairs

        ttl_collector.setdefault(ttl_tag_text, nfo_tag_text)

      # add the table header and the collected attribute value pairs to the specs_collector

      specs_collector.setdefault(table_header, ttl_collector)

      

      

# test it out

bs = get_bs(device_link)

get_device_specs(bs, specs_collector, device_name, device_link)

for key, value in specs_collector.items():

  print('{} : {}'.format(key, value))

# ge the device banner data and add it to the specs_collector with the key 'Banner'

def get_device_banner(bs, specs_collector, device_name, device_link):

  '''A function to scrape data from the banner of a a device'''

  # get the unordered list with the class name 'specs-spotlight-features'

  banner = bs.findChild('ul', class_='specs-spotlight-features')

  # if a banner is not present, collect the information for dbugging

  if not banner:

    error = '{} has no banner| device_link:{}'.format(device_name, device_link)

    collect_debug(error)

  # else get all the list items and find the data stored in the banner

  else:

    banner_items = banner.findChildren('li')

    banner_specs_collector = {}

    for list_item in banner_items:

      # find all the items in the list falling into the data-spec category, such as battery-hl, screen-hl, etc...

      banner_specs = list_item.findChildren(['span', 'div'], {'data-spec':re.compile('.*')})

      # for each spec in the banner iterate through the key value pairs and add it to banner_spec_collector

      for banner_spec in banner_specs:

        banner_spec_name = banner_spec['data-spec']

        if banner_spec_name:

          # setting strip = True removes any white space space characters

          banner_spec_value = banner_spec.get_text(strip=True)

          if banner_spec_value:

            banner_specs_collector[banner_spec_name] = banner_spec_value



      # we now need to find the device popularity and hits from the webpage

      if 'help-popularity' in list_item['class']:

        # get information about the device's popularity and collect debug if it does not have the attribute

        device_popularity = list_item.findChild('strong')

        if not device_popularity:

          error = '{} has no device_popularity| device_link:{}'.format(device_name, device_link)

          collect_debug(error)

        else:

          device_popularity = device_popularity.get_text()

          # do no capture the Unicode white space character '\xa0'

          if device_popularity == '\xa0' or device_popularity == '' :

            device_popularity = None        

        # collect information about the device's popularity and collect debug if it does not have the attribute          

        device_hits = list_item.findChild('span')

        if not device_hits:

          error = '{} has no device_hits| device_link:{}'.format(device_name, device_link)

          collect_debug(error)

        else:

          device_hits = device_hits.get_text()

          if device_hits == '\xa0'or device_hits == '':

            device_hits = None



        # add device_popularity and divice_hits to the banner_specs_collector if they are present

        if device_popularity:

          banner_specs_collector['device_popularity'] = device_popularity

        if device_hits:

          banner_specs_collector['device_hits'] = device_hits

    specs_collector['Banner'] = banner_specs_collector



# test it out

get_device_banner(bs, specs_collector, device_name, device_link)

for key, value in specs_collector.items():

  print('{} : {}'.format(key, value))

  

# we can see that the Banner has now been successfully added to the specs_collector
# the last thing we want to grab from the device page is the 'Total user opinions' at the bottom of the page

def get_device_opinions(bs,specs_collector, device_name, device_link):

  '''A function to get the 'Total user opionions' for a device form the devie pages bs object'''

  opinions = bs.findChild('div', id='opinions-total')

  if not opinions:

    error = '{} has no Total user opinions| device_link:{}'.format(devie_name, device_link)

    collect_debug(error)

  else:

    num_opinions = opinions.b.get_text(strip=True)

    specs_collector.setdefault('Opinions', num_opinions)

  



# test it out

get_device_opinions(bs, specs_collector, device_name, device_link)

get_device_banner(bs, specs_collector, device_name, device_link)

for key, value in specs_collector.items():

  print('{} : {}'.format(key, value))
# put everything we have made so far for collecting the specs of a device into a single function

def get_device_data(device_link):

  '''A function to get the banner data and spec-sheet from a device on GSMArena

  

     Takes in a device's url and returns a dict with all the specs

  '''

  specs_collector = {}

  # get the devie bs object

  bs = get_bs(device_link)

  device_name = bs.findChild('h1', class_='specs-phone-name-title').get_text()

  # get te device_specs

  specs = get_device_specs(bs,specs_collector, device_name, device_link)

  # get the banner using the get_device_banner method, defined below

  banner = get_device_banner(bs, specs_collector, device_name, device_link)

  # get the user opinions for the device

  opinions = get_device_opinions(bs,specs_collector, device_name, device_link)

  # get the banner spec_sheet using the get_device_specs method, defined below 

  return specs_collector



    

# test it out

device_link = 'https://www.gsmarena.com/samsung_galaxy_s10-9536.php'

get_device_data(device_link)
# helper function that allows returns the maker_id of of maker

def get_maker_id(name_of_maker, maker_list):

  '''A function for returning the maker_id of a maker given the maker_name.

     

     This function is case insensitive.

  '''

  for maker in maker_list:

    name = maker[1]

    maker_id = maker[0]

    if name_of_maker.lower() == name.lower():

      return maker_id

  # if a name is not found in the maker list, we want to throw an exception and collect it for log

  raise NameError('GSMArena has no maker \'{}\''.format(maker_name))

  

  

def switch(maker_id, name_of_maker, maker_list):

  '''A function for returning a bool which tells the crawler which maker(s) to scrape for data.'''

  # if no name_of_maker is given, return true in all cases

  if name_of_maker is None:

    return True

  # else get maker_id for the given maker name and return True only when current maker_id == given maker_id

  else:

    given_maker_id = get_maker_id(name_of_maker, maker_list)

    if given_maker_id == maker_id:

      return True

    else:

      return False
# assemble the functions we built earlier in the right format in order to get the functionality we want



def GSMCrawler(seed_url, name_of_maker=None):

  '''A crawler to return device data from GSMArena

      

     Takes in the seed_url 'https://www.gsmarena.com/makers.php3'.

     If name_of_maker is specified, device info for will be collected only for that maker.

  '''

  # we want to measure how long the crawling took to excecute

  start_time = time.time()

# STEP 1: get the links to all the makers in GSMArena

  print('Starting GSMArena Crawler...\n')

  maker_list = get_maker_links(seed_url)

#   maker_list = get_maker_links(seed_url)

  print('Successfully retrived maker_list!\n')

  

  # tell us if we the crawl is being done for a single maker or all makers

  if name_of_maker is None:

    print('Crawling for devices by ALL makers...\n')

  else:

    print('Crawling for devices by {}...\n'.format(name_of_maker))

  

# STEP 2: iterate trough each maker and get the device links and device info from all the nav pages

  devices_collector = {}

  for maker_id, maker in enumerate(maker_list):

    if switch(maker_id, name_of_maker, maker_list):

      maker_link = maker[-1]

      maker_name = maker[1]



# STEP 3: the first thing we want to do on the makers page is to get a list of all nav links

      nav_pages_links = get_nav_page_links(maker)

      # for each nav page in a maker's nav_pages, get the device info for all devices by that maker

      print('Getting nav_page_links for {}...\n'.format(maker_name))

      devices_collector[maker_name] = []

      for nav_page_num, nav_page_link in nav_pages_links.items():

        get_device_links(nav_page_link, devices_collector, maker_name)

      print('Successfully collected all device info for {}!\n'.format(maker_name))

      

  # notify us of how many devices were collected in total 

  total_num_devices = 0

  for maker, devices_info in devices_collector.items():

    total_num_devices += devices_info.__len__()

  print('Successfully collected info for all devices! {} devices were collected\n'.format(total_num_devices))



# STEP 4: go through each each device_link in the devices_collector and pass it onto get_device_data

  print('Collecting spec sheets for all devices. This could take a while. Sit back and relax...\n')

# WARNING: This loop will scrape the spec sheet of every device in GSM Arena.

 # it is good practice to put this under a try block; in case some thing we want to collect some debug info

  try:

    for maker, devices_info in devices_collector.items():

      print('Getting spec sheets for {} devices by {}\...n'.format(devices_info.__len__(), maker))

      for device_num, device in enumerate(devices_info):

        device_link = device[-1]

        # get the device data using the get_device_data function we defined earlier

        device_specs =  get_device_data(device_link)

        device.append(device_specs)

# WARNING END

      print('Successfully scraped info for all devices by {}\n!'.format(maker))

  

  except Exception as e :

      error = 'Device crawl exception: {}| device_name: {}| device_links:{}\n'.format(e, device_link)

      collect_debug(error)

  # if nothing went wrong, let us know that all has gone well

  else:

      end_time = time.time()

      print('GSMCrawler has completed excecuting! All credits for this data goes to the GSMArena team\n')

      print('Time time required to excecute for {}: {} seconds'.format(end_time - start_time))

      print('Time time per : {} seconds'.format(end_time - start_time))

      print('='*50)

  finally:

    # finally return the data_collector

    return devices_collector
# try out our newly built crawler



seed_path = 'makers.php3'

seed_url = domain_name + seed_path 



# due to Colab's limitations, I will run the crawler only for Samsung

# you can find the data the full set of devices on my GitHub page under the name devices_data.json

devices_collector = GSMCrawler(seed_url,'Samsung')



# uncomment the code below to run the crawler for the full site

# devices_collector = GSMCrawler(seed_url)



print(devices_collector.keys())

print(devices_collector['Samsung'])
# we want to convert the data that we just collected into a JSON oject to interact with later 

def make_devices_json(devices_collector, save_json=False):

  '''A function for coverting devices_collector text into a JSON obj and optionally saving the file

     If save_json is True, a file called devices_data.txt will be made in your current working directory

  '''

  json_dict = {}

  for maker, devices_info in devices_collector.items():

    maker_dict = {}

    for device_id, device in enumerate(devices_info):

      device_dict = {}

      device_name, device_info, device_img_link, device_link, device_specs = device



      # start adding data as key value pairs into the device_dict

      device_dict['device_name'] = device_name

      device_dict['device_info'] = device_info

      device_dict['device_img_link'] = device_img_link

      device_dict['device_link'] = device_link

      device_dict['device_specs'] = device_specs





      # use the device_id as key to to set the device

      maker_dict[device_id] = device_dict

    # set the maker id to the json_dict with the maker name as key

    json_dict[maker] = maker_dict

    

  # if save json is true, then save the devices collected by the crawler in the working directory as a json file

  if save_json:

    cwd = os.getcwd()

    save_file_name = cwd + '/devices_data.txt'

    with open(save_file_name, 'w', encoding='utf-8') as file:

      json.dump(json_dict, file, ensure_ascii=False)

    # notify us where the file was saved

    print('Successfully saved device data as a JSON file at {}'.format(save_file_name))

    

  return json.dumps(json_dict, ensure_ascii=False)

  

# test it out

devices_json = make_devices_json(devices_collector, save_json=True)

devices_json
# verify that the newly created json file is present in you local directory

!ls
# if you are using Colab and want to download the file we just created

files.download('devices_data.txt')
def read_devices_json(file_path):

  '''A function for reading in a JSON obj of the devices data i.e devices.txt

     

     Takes in the string file_path

  '''

  with open(file_path, 'r', encoding='utf-8') as file:

    return json.load(file)



# test it out

cwd = os.getcwd()

file_path = cwd + '/devices_data.txt'

json_dict = read_devices_json(file_path)



# double check that we have a dictionary 

json_dict.keys()