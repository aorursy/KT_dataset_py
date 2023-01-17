import urllib

import urllib.request

from bs4 import BeautifulSoup

from collections import defaultdict

import os



def soup(url):

    thepage = urllib.request.urlopen(url)

    soupdata = BeautifulSoup(thepage, "html.parser")

    return soupdata





# chhattisgarh

edatas = ""

edata1 = ""

states = defaultdict(dict)

constituenciesRange = []



soup4 = "https://results.eci.gov.in/pc/en/constituencywise/ConstituencywiseU011.htm?ac=1" #2014

soup5 = soup(soup4)



for records2 in soup5.findAll("select", {"id": "ddlState"}):

    for option in records2.findAll('option'):

        if(option['value'] != "Select State" and option.text != "Select State"):

            #states.append(option['value'])

            states[str(option['value'])] = option.text

constituenciesName = {"U01": {"1": "Andaman & Nicobar Islands"}, "S01": {"1": "Adilabad", "24": "Amalapuram", "22": "Anakapalli", "36": "Anantapur", "18": "Aruku", "32": "Bapatla", "14": "Bhongir", "10": "CHELVELLA", "42": "Chittoor", "27": "Eluru", "30": "Guntur", "37": "Hindupur", "9": "Hyderabad", "38": "Kadapa", "23": "Kakinada", "3": "Karimnagar", "17": "Khammam", "35": "Kurnool", "28": "Machilipatnam", "16": "Mahabubabad", "11": "Mahbubnagar", "7": "Malkajgiri", "6": "Medak", "12": "Nagarkurnool", "13": "Nalgonda", "34": "Nandyal", "31": "Narasaraopet", "26": "Narsapuram", "39": "Nellore", "4": "Nizamabad", "33": "Ongole", "2": "Peddapalle", "25": "Rajahmundry", "41": "Rajampet", "8": "Secundrabad", "19": "Srikakulam", "40": "Tirupati", "29": "Vijayawada", "21": "Visakhapatnam", "20": "Vizianagaram", "15": "Warangal", "5": "Zahirabad"}, "S02": {"2": "ARUNACHAL EAST", "1": "ARUNACHAL WEST"}, "S03": {"3": "Autonomous District", "6": "Barpeta", "4": "Dhubri", "13": "Dibrugarh", "7": "Gauhati", "12": "Jorhat", "11": "Kaliabor", "1": "Karimganj", "5": "Kokrajhar", "14": "Lakhimpur", "8": "Mangaldoi", "10": "Nowgong", "2": "Silchar", "9": "Tezpur"}, "S04": {"9": "Araria", "32": "Arrah", "37": "Aurangabad", "27": "Banka", "24": "Begusarai", "26": "Bhagalpur", "33": "Buxar", "14": "Darbhanga", "38": "Gaya (SC)", "17": "Gopalganj (SC)", "21": "Hajipur (SC)", "36": "Jahanabad", "40": "Jamui (SC)", "7": "Jhanjharpur", "35": "Karakat", "11": "Katihar", "25": "Khagaria", "10": "Kishanganj", "13": "Madhepura", "6": "Madhubani", "19": "Maharajganj", "28": "Munger", "15": "Muzaffarpur", "29": "Nalanda", "39": "Nawada", "2": "Paschim Champaran", "31": "Pataliputra", "30": "Patna Sahib", "12": "Purnia", "3": "Purvi Champaran", "23": "Samastipur (SC)", "20": "Saran", "34": "Sasaram (SC)", "4": "Sheohar", "5": "Sitamarhi", "18": "Siwan", "8": "Supaul", "22": "Ujiarpur", "16": "Vaishali", "1": "Valmiki Nagar"}, "U02": {"1": "CHANDIGARH"}, "S26": {"10": "BASTAR", "5": "BILASPUR", "7": "DURG", "3": "JANJGIR-CHAMPA", "11": "KANKER", "4": "KORBA", "9": "MAHASAMUND", "2": "RAIGARH", "8": "RAIPUR", "6": "RAJNANDGAON", "1": "SARGUJA"}, "U03": {"1": "Dadar & Nagar Haveli"}, "U04": {"1": "Daman & diu"}, "S05": {"1": "North Goa", "2": "South Goa"}, "S06": {"7": "Ahmedabad East", "8": "Ahmedabad West", "14": "Amreli", "16": "Anand", "2": "Banaskantha", "23": "Bardoli", "22": "Bharuch", "15": "Bhavnagar", "21": "Chhota Udaipur", "19": "Dahod", "6": "Gandhinagar", "12": "Jamnagar", "13": "Junagadh", "1": "Kachchh", "17": "Kheda", "4": "Mahesana", "25": "Navsari", "18": "Panchmahal", "3": "Patan", "11": "Porbandar", "10": "Rajkot", "5": "Sabarkantha", "24": "Surat", "9": "Surendranagar", "20": "Vadodara", "26": "Valsad"}, "S07": {"1": "Ambala", "8": "Bhiwani-Mahendragarh", "10": "Faridabad", "9": "Gurgaon", "4": "Hisar", "5": "Karnal", "2": "Kurukshetra", "7": "Rohtak", "3": "Sirsa", "6": "Sonipat"}, "S08": {"3": "Hamirpur", "1": "Kangra", "2": "Mandi", "4": "Shimla"}, "S09": {"3": "Anantnag", "1": "Baramulla", "6": "Jammu", "4": "Ladakh", "2": "Srinagar", "5": "Udhampur"}, "S27": {"4": "Chatra", "7": "Dhanbad", "2": "Dumka", "6": "Giridih", "3": "Godda", "14": "Hazaribagh", "9": "Jamshedpur", "11": "Khunti", "5": "Kodarma", "12": "Lohardaga", "13": "Palamau", "1": "Rajmahal", "8": "Ranchi", "10": "Singhbhum"}, "S10": {"3": "Bagalkot", "25": "Bangalore central", "24": "Bangalore North", "23": "Bangalore Rural", "26": "Bangalore South", "2": "Belgaum", "9": "Bellary", "7": "Bidar", "4": "Bijapur", "22": "Chamarajanagar", "27": "Chikkballapur", "1": "Chikkodi", "18": "Chitradurga", "17": "Dakshina Kannada", "13": "Davanagere", "11": "Dharwad", "5": "Gulbarga", "16": "Hassan", "10": "Haveri", "28": "Kolar", "8": "Koppal", "20": "Mandya", "21": "Mysore", "6": "Raichur", "14": "Shimoga", "19": "Tumkur", "15": "Udupi Chikmagalur", "12": "Uttara Kannada"}, "S11": {"15": "Alappuzha", "9": "Alathur", "19": "Attingal", "11": "Chalakudy", "12": "Ernakulam", "13": "Idukki", "2": "Kannur", "1": "Kasaragod", "18": "Kollam", "14": "Kottayam", "5": "Kozhikode", "6": "Malappuram", "16": "Mavelikkara", "8": "Palakkad", "17": "Pathanamthitta", "7": "Ponnani", "20": "Thiruvananthapuram", "10": "Thrissur", "3": "Vadakara", "4": "Wayanad"}, "U06": {"1": "Lakshadweep"}, "S12": {"15": "BALAGHAT", "29": "BETUL", "2": "BHIND", "19": "BHOPAL", "16": "CHHINDWARA", "7": "DAMOH", "21": "DEWAS", "25": "DHAR", "4": "GUNA", "3": "GWALIOR", "17": "HOSHANGABAD", "26": "INDORE", "13": "JABALPUR", "8": "KHAJURAHO", "28": "KHANDWA", "27": "KHARGONE", "14": "MANDLA", "23": "MANDSOUR", "1": "MORENA", "20": "RAJGARH", "24": "RATLAM", "10": "REWA", "5": "SAGAR", "9": "SATNA", "12": "SHAHDOL", "11": "SIDHI", "6": "TIKAMGARH", "22": "UJJAIN", "18": "VIDISHA"}, "S13": {"37": "Ahmadnagar", "6": "Akola", "7": "Amravati", "19": "Aurangabad", "35": "Baramati", "39": "Beed", "11": "Bhandara - gondiya", "23": "Bhiwandi", "5": "Buldhana", "13": "Chandrapur", "2": "Dhule", "20": "Dindori", "12": "Gadchiroli-Chimur", "48": "Hatkanangle", "15": "Hingoli", "3": "Jalgaon", "18": "Jalna", "24": "Kalyan", "47": "Kolhapur", "41": "Latur", "43": "Madha", "33": "Maval", "31": "Mumbai   South", "26": "Mumbai North", "29": "Mumbai North central", "28": "Mumbai North East", "27": "Mumbai North West", "30": "Mumbai South central", "10": "Nagpur", "16": "Nanded", "1": "Nandurbar", "21": "Nashik", "40": "Osmanabad", "22": "Palghar", "17": "Parbhani", "34": "Pune", "32": "Raigad", "9": "Ramtek", "46": "Ratnagiri - sindhudurg", "4": "Raver", "44": "Sangli", "45": "Satara", "38": "Shirdi", "36": "Shirur", "42": "Solapur", "25": "Thane", "8": "Wardha", "14": "Yavatmal-Washim"}, "S14": {"1": "Inner manipur", "2": "Outer manipur"}, "S15": {"1": "Shillong", "2": "Tura"}, "S16": {"1": "MIZORAM"}, "S17": {"1": "Nagaland"}, "U05": {"1": "CHANDNI CHOWK", "3": "EAST DELHI", "4": "NEW DELHI", "2": "NORTH EAST DELHI", "5": "NORTH WEST DELHI", "7": "SOUTH DELHI", "6": "WEST DELHI"}, "S18": {"19": "Aska", "6": "Balasore", "1": "Bargarh", "20": "Berhampur", "7": "Bhadrak", "18": "Bhubaneswar", "10": "Bolangir", "14": "Cuttack", "9": "Dhenkanal", "16": "Jagatsinghpur", "8": "Jajpur", "11": "Kalahandi", "13": "Kandhamal", "15": "Kendrapara", "4": "Keonjhar", "21": "Koraput", "5": "Mayurbhanj", "12": "Nabarangpur", "17": "Puri", "3": "Sambalpur", "2": "Sundargarh"}, "U07": {"1": "Puducherry"}, "S19": {"2": "Amritsar", "6": "Anandpur Sahib", "11": "Bathinda", "9": "Faridkot", "8": "Fatehgarh Sahib", "10": "Firozpur", "1": "Gurdaspur", "5": "Hoshiarpur", "4": "Jalandhar", "3": "Khadoor Sahib", "7": "Ludhiana", "13": "Patiala", "12": "Sangrur"}, "S20": {"13": "Ajmer", "8": "Alwar", "20": "Banswara", "17": "Barmer", "9": "BHARATPUR", "23": "Bhilwara", "2": "Bikaner", "21": "Chittorgarh", "3": "Churu", "11": "Dausa", "1": "Ganganagar", "7": "Jaipur", "6": "Jaipur Rural", "18": "Jalore", "25": "JHALAWAR-BARAN", "4": "Jhunjhunu", "16": "Jodhpur", "10": "KARAULI-DHOLPUR", "24": "Kota", "14": "Nagaur", "15": "Pali", "22": "Rajsamand", "5": "Sikar", "12": "TONK-SAWAI MADHOPUR", "19": "Udaipur"}, "S21": {"1": "Sikkim"}, "S22": {"7": "Arakkonam", "12": "Arani", "4": "Chennai central", "2": "Chennai North", "3": "Chennai South", "27": "Chidambaram", "20": "Coimbatore", "26": "Cuddalore", "10": "Dharmapuri", "22": "Dindigul", "17": "Erode", "14": "Kallakurichi", "6": "Kancheepuram", "39": "Kanniyakumari", "23": "Karur", "9": "Krishnagiri", "32": "Madurai", "28": "Mayiladuthurai", "29": "Nagapattinam", "16": "Namakkal", "19": "Nilgiris", "25": "Perambalur", "21": "Pollachi", "35": "Ramanathapuram", "15": "Salem", "31": "Sivaganga", "5": "Sriperumbudur", "37": "Tenkasi", "30": "Thanjavur", "33": "Theni", "1": "Thiruvallur", "36": "Thoothukkudi", "24": "Tiruchirappalli", "38": "Tirunelveli", "18": "Tiruppur", "11": "Tiruvannamalai", "8": "Vellore", "13": "Viluppuram", "34": "Virudhunagar"}, "S23": {"2": "Tripura East", "1": "Tripura West"}, "S24": {"18": "Agra", "44": "Akbarpur", "15": "Aligarh", "52": "Allahabad", "55": "Ambedkar Nagar", "37": "Amethi", "9": "Amroha", "24": "Aonla", "69": "Azamgarh", "23": "Badaun", "11": "Baghpat", "56": "Bahraich", "72": "Ballia", "48": "Banda", "67": "Bansgaon", "53": "Barabanki", "25": "Bareilly", "61": "Basti", "78": "Bhadohi", "4": "Bijnor", "14": "Bulandshahr", "76": "Chandauli", "66": "Deoria", "29": "Dhaurahra", "60": "Domariyaganj", "22": "Etah", "41": "Etawah", "54": "Faizabad", "40": "Farrukhabad", "49": "Fatehpur", "19": "Fatehpur Sikri", "20": "Firozabad", "13": "Gautam Buddha Nagar", "12": "Ghaziabad", "75": "Ghazipur", "70": "Ghosi", "59": "Gonda", "64": "Gorakhpur", "47": "Hamirpur", "31": "Hardoi", "16": "Hathras", "45": "Jalaun", "73": "Jaunpur", "46": "Jhansi", "2": "Kairana", "57": "Kaiserganj", "42": "Kannauj", "43": "Kanpur", "50": "Kaushambi", "28": "Kheri", "65": "Kushi Nagar", "68": "Lalganj", "35": "Lucknow", "74": "Machhlishahr", "63": "Maharajganj", "21": "Mainpuri", "17": "Mathura", "10": "Meerut", "79": "Mirzapur", "32": "Misrikh", "34": "Mohanlalganj", "6": "Moradabad", "3": "Muzaffarnagar", "5": "Nagina", "51": "Phulpur", "26": "Pilibhit", "39": "Pratapgarh", "36": "Rae Bareli", "7": "Rampur", "80": "Robertsganj", "1": "Saharanpur", "71": "Salempur", "8": "Sambhal", "62": "Sant Kabir Nagar", "27": "Shahjahanpur", "58": "Shrawasti", "30": "Sitapur", "38": "Sultanpur", "33": "Unnao", "77": "Varanasi"}, "S28": {"3": "Almora", "2": "Garhwal", "5": "Hardwar", "4": "Nainital-udhamsingh Nagar", "1": "Tehri Garhwal"}, "S25": {"2": "Alipurduars", "29": "Arambagh", "40": "Asansol", "10": "Baharampur", "6": "Balurghat", "14": "Bangaon", "36": "Bankura", "17": "Barasat", "38": "Bardhaman Purba", "15": "Barrackpore", "18": "Basirhat", "42": "Birbhum", "37": "Bishnupur", "41": "Bolpur", "39": "Burdwan - durgapur", "1": "Cooch behar", "4": "Darjeeling", "21": "Diamond harbour", "16": "Dum dum", "32": "Ghatal", "28": "Hooghly", "25": "Howrah", "22": "Jadavpur", "3": "Jalpaiguri", "9": "Jangipur", "33": "Jhargram", "19": "Joynagar", "31": "Kanthi", "23": "Kolkata Dakshin", "24": "Kolkata Uttar", "12": "Krishnanagar", "8": "Maldaha Dakshin", "7": "Maldaha Uttar", "20": "Mathurapur", "34": "Medinipur", "11": "Murshidabad", "35": "Purulia", "5": "Raiganj", "13": "Ranaghat", "27": "Srerampur", "30": "Tamluk", "26": "Uluberia"}}

constituenciesRange = {'U01':2, 'S01':26, 'S02':3, "S03":15, 'S04':41, 'U02':2, 'S26':12, 'U03':2, 'U04':2, 'S05':3, 'S06':27, 'S07':11, 'S08':5, 'S09':7, 'S27':15, 'S10':29, 'S11':21, 'U06':2, 'S12':30, 'S13':49, 'S14':3, 'S15':3, 'S16':2, 'S17':2, 'U05':8, 'S18':22, 'U07':2, 'S19':14, 'S20':26, 'S21':2, 'S22':40, 'S23':3, 'S24':81, 'S28':6, 'S25':43}



for code in constituenciesRange:

    rangeLimit = constituenciesRange[code]

    for const in range(1, rangeLimit):

        #soup3 = "http://eciresults.nic.in/Constituencywise" + code + str(const) + ".htm?ac=" + str(const)#2019

        soup3 =  "https://results.eci.gov.in/pc/en/constituencywise/Constituencywise"+code+str(const)+".htm?ac="+str(const)

        #print(soup3)

        soup2 = soup(soup3)

        for records2 in soup2.findAll("div", {"id": "div1"}):

            for records in records2.findAll("tr", {"style": "font-size:12px;"}):

                edata = ""

                for data1 in records.findAll('td'):

                    data = data1.text.replace(',', '')

                    edata = edata + "," + data

                edatas = edatas + "\n" + edata[1:] + "," + code+"," + str(states[code])+","+ str(const)+","+constituenciesName[code][str(const)]

                print(edatas)



print(edatas)



header = "snNo,candidateName,party,evmVotes,postalVotes,votes,%ofVotes,stateCode,stateName,pcno,pcName"

file = open(os.path.expanduser("Candi2019.csv"), "wb")#2018

file.write(bytes(header, encoding="ascii", errors="ignore"))

file.write(bytes(edata1, encoding="ascii", errors="ignore"))

file.write(bytes(edatas, encoding="ascii", errors="ignore"))


