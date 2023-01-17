!pwd
!ls -l
!git clone https://github.com/COVIEWED/coviewed_web_scraping
!pip install -r coviewed_web_scraping/requirements.txt
!git clone https://github.com/COVIEWED/coviewed_data_collection
!pip install -r coviewed_data_collection/requirements.txt
!cd coviewed_data_collection/ && python3 src/get_reddit_submission_links.py -a=2020-03-28T23:59:00 -b=2020-03-29T00:00:00 -s=Coronavirus --verbose
!ls coviewed_data_collection/data/*.tsv
!cd coviewed_data_collection/ && python3 src/get_reddit_submission_texts.py -s=Coronavirus --verbose
!ls coviewed_data_collection/data/*.txt
!cp coviewed_data_collection/data/news_urls.txt coviewed_web_scraping/data/
!ls coviewed_web_scraping/data/*.txt
!wc -l coviewed_web_scraping/data/news_urls.txt
!cd coviewed_web_scraping/ && sh run.sh
!ls coviewed_web_scraping/data | grep .txt | wc -l