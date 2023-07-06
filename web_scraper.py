# import package

import argparse
import logging
import os
from pathlib import Path
import pandas as pd
from time import localtime, strftime
import yaml
from util_scraper.scraper_helper import scraper
import requests
import ssl 

LOGGER = logging.getLogger(__name__)

if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--county', required=True, default='Taipei', help='Which county you want to crawl from mobile01.com?')
    parser.add_argument('-r', '--run', required=True, choices=['scrape_end2end', 'scrape_topic_summary', 'scrape_topic_by_urls', 'scrape_topic'], help='Which function you want to run. Please see ReadMe for more details.')
    parser.add_argument('-ts', '--topic_summary', required=False, default='None', help='The topic_summary.pkl that you want to keep scraping.')
    parser.add_argument('-fp', '--final_page', required=False, default='0', help='The final page you want to keep scraping topic_summary with.')
    parser.add_argument('-tc', '--topic_content', required=False, default='None', help='The topic_content.pkl that you want to keep scraping.')
    parser.add_argument('-sp', '--start_page', required=False, default='1', help='The start page you want to keep scrapping topic_content with.')
    args = parser.parse_args()
    
    with open('util_scraper/config.yaml', 'r') as stream:
        config = yaml.load(stream, Loader = yaml.FullLoader)
        id_dict = config['county_ids']
        county_name = args.county
        
        if county_name not in id_dict.keys():
            print('Invalid county name. Please check the config to see the correct name of a county.')
        else:
            county_id = int(id_dict[county_name])
            final_page = int(args.final_page)
            file_name_topic_summary = args.topic_summary
            start_page = int(args.start_page)
            file_name_topic_content = args.topic_content
            
            if args.run == 'scrape_end2end':
                print(f'********Start to end2end scrape {county_name}********')
                s = scraper(county_id = county_id)
                s.scrape_end2end()
                print(f'start to end2end scrape {county_name}')
                print('********Finished********')
                
            elif args.run == 'scrape_topic_summary':
                print(f'********Start to scrape topic summaries in {county_name}********')
                s = scraper(county_id = county_id)
                s.scrape_topic_summary(final_page=final_page, file_name=file_name_topic_summary)
                print('********Finished********')
                
            elif args.run == 'scrape_topic_by_urls':
                print(f'********Start to scrape topic contents from the urls in the file:{file_name_topic_summary}********')
                s = scraper(county_id = county_id)
                s.scrape_topic_by_urls(file_name_topic_summary=file_name_topic_summary, file_name_topic_content=file_name_topic_content)
                print('********Finished********')
                
            elif args.run == 'scrape_topic':
                url = input('Please enter the url of the topic you want to scrape:')
                print(f'********Start to scrape the topic contents********')
                s = scraper(county_id = county_id)
                s.scrape_topic(url=url, file_name_topic_content=file_name_topic_content, start_page=start_page)
                print('********Finished********')
                    
                
