"""

    Python Example of API Call to the Chilicon Power Cloud

"""

import json

import requests



# Credentials for authentication - TODO: Change for each user

USERNAME = 'info@chiliconpower.com'  # put correct usename here

PASSWORD = 'guest'  # put correct password here



# URL paths

URL_root = 'http://cloud.chiliconpower.com'  # cloud main URL

login_URL = URL_root + '/login'

INSTALLATION_HASH = '02b6612795e062f02db5d49b01d1ea6966d3c4b597d8e58802f9b84c2b5d3ace'  # can be found in the URL when

# accessing that site in a browser

INSTALLATION_URL = URL_root + '/installation/' + INSTALLATION_HASH

fetchOwnerUpdate_URL = URL_root + '/ajax/fetchOwnerUpdate?today=%s'



print("\n\nThis is an example to pull some data from the cloud using the Chilicon Power API\n"

      "The user authenticates on the public account info@chiliconpower.com\n"

      "with the password 'guest'\n"

      "It retrieves the data from a call to fetchOwnerUpdate for installation Hr Health Dept Solar\n")



session = requests.session()  # create a session object to persist the login session across all our requests



req_headers = {

    'Content-Type': 'application/x-www-form-urlencoded'

}



form_data = {

    'username': USERNAME,

    'password': PASSWORD,

    'Login': 'Login'

}



# Authenticate

print("\n____________LOGIN_POST_______")

r = session.post(login_URL, data=form_data, headers=req_headers, allow_redirects=False)

print("Request Headers: %s" % r.headers)

print("Request Status Code: %s" % r.status_code)

print("Request Text: %s" % r.text)



# Read data

url = fetchOwnerUpdate_URL % '2020-01-01'

r2_headers = {

    'Host': 'cloud.chiliconpower.com',

    'Referer': INSTALLATION_URL,

    'today': '2020-01-01'

}

r2 = session.get(url, headers=r2_headers)

print("\n____________DATA_____________")

print(r2.headers)

print(r2.status_code)

print(r2.text)



print("\n____________PARSED RESULTS___")

[today, lifetimeEnergy, currentProduction] = json.loads(r2.text)

print('Today =', today)

print("Lifetime Energy = ", lifetimeEnergy)

print("Current Production = ", currentProduction)