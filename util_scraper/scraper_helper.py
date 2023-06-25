##### mobile01 crawler
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
import time
import numpy as np
import os
import pandas as pd
import re
import random
from time import localtime, strftime




class scraper:

    def __init__(self, county_id, st, et, page=0, final_page=0):
        self.county_id = county_id
        self.county_url = f'https://www.mobile01.com/topiclist.php?f={county_id}'
        self.st = st
        self.et = et
        self.save_path = f'scraped_result/{self.county_id}/'
        self.save_dt = strftime("%Y%m%d", localtime())
        file_name = self.save_path + f'topic_summary_{self.save_dt}.pkl'
        self.topic_dict = self.check_parsed_files(file_name)
        
    def check_parsed_files(self, file_name):
        if not os.path.exists(file_name):
            topic_dict = {'url':[], 'title':[], 'first_post_time':[], 'last_post_time':[]}
        else :
            topic_df = pd.read_pickle(file_name)
            topic_dict = topic_df.to_dict('list')
        return topic_dict
                
    def scrape_topic_summary(self, final_page=0):
        
        driver = self.create_driver()
        # 
        if final_page == 0:
            url = self.county_url
            
            self.try_get_url(driver, url)

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            print('Start to sleep')
            time.sleep(5)
            print('End of sleeping')
        
            # Check final page
            final_page = self.get_final_page(soup)
            print('End of parsing final_page')
            
        for p in range(1, final_page+1)[::-1]:
            print(f'Go to page {p}')
            
            # start time
            start = time.time()
            
            page_url = self.county_url + f'&p={p}'
            self.try_get_url(driver, page_url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            self.topic_dict = self.get_topic_urls(soup)
            # end time
            end = time.time()
            sec = end - start
            print(f"Time spend: {sec:.2f} seconds")
            
            # save scraped result
            
            self.save_dt = strftime("%Y%m%d", localtime())
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)
            file_name = self.save_path + f'topic_summary_{self.save_dt}.pkl'
            pd.DataFrame(self.topic_dict).to_pickle(file_name)
            
            self.page = p
            print('Start to sleep')
            time.sleep(1+random.random()*10)
            print('End of sleeping')
        driver.close()
        
    def scrape_topic_url(self, topic_df):
        topic_urls = topic_df['url'].tolist()
        for ind, url in enumerate(topic_urls):
            print('parsing: ', topic_df.iloc[ind]['title'])
            content_dict = self.scrape_topic(url = url, cid = self.county_id)
            
            self.save_dt = strftime("%Y%m%d", localtime())
            filename = self.save_path + f'topic_{self.save_dt}.pkl'
            content_df = pd.DataFrame(content_dict)
            content_df['cid'] = self.county_id
            content_df.to_pickle(filename)
    
    def scrape_topic(self, url, cid, start_page=1, content_dict = {}):
        driver = self.create_driver()
        final_page = 1
        if len(content_dict) == 0:
            content_dict = {'title':[], 'content':[], 'author':[], 'post_time':[], 'floor':[], 'post_time':[], 'cid':[]}
        
        while start_page <= final_page:
            print(f'start to parse page {start_page}')
            topic_url = f'https://www.mobile01.com/{url}&p={start_page}'

            # start time
            start = time.time()
            
            self.try_get_url(driver, topic_url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            if final_page == 1:
                final_page = self.get_final_page(soup)

            temp_dict = self.scrape_topic_content(soup)
            print(temp_dict)
            for k, v in temp_dict.items():
                content_dict[k] += v
            
            # end time
            end = time.time()
            sec = end - start
            print(f"Time spend: {sec:.2f} seconds")
            
            self.save_dt = strftime("%Y%m%d", localtime())
            filename = self.save_path + f'topic_{self.save_dt}.pkl'
            content_df = pd.DataFrame(content_dict)
            print(content_df.shape)
            content_df.to_pickle(filename)
            start_page +=1
            print('Start to sleep')
            time.sleep(3+random.random()*10)
            print('End of sleeping')
        driver.close()
        
        return content_dict
    
    def scrape_topic_content(self, soup):
        content_dict = {'title':[], 'content':[], 'author':[], 'post_time':[], 'floor':[], 'cid':[]}
        title = soup.find('h1', {'class':'t2'}).text.strip()
        contents = soup.find_all("div", {"class": "l-articlePage"})
        
        for c in contents:    
            author = c.find('div', {'class':'c-authorInfo__id'})
            if not author:
                continue
            content_dict['author'].append(author.text.strip())
            navigation = c.find(lambda tag: tag.name == 'div' and 
                                   tag.get('class') == ['l-navigation__item'])
            content_dict['post_time'].append(navigation.find_all('span', {'class':'o-fNotes o-fSubMini'})[0].text.strip())
            floor = navigation.find_all('span', {'class':'o-fNotes o-fSubMini'})[1].text.replace('#', '').strip()
            content_dict['floor'].append(floor)
            all_text = c.find('article').get_text(' ')
            quote_list = c.find('article').find_all('blockquote')
            for quote in quote_list:
                quote_text = quote.get_text(' ')
                all_text = all_text.replace(quote_text, '')
            content_dict['content'].append(all_text.strip())
            content_dict['cid'].append(self.county_id)
            content_dict['title'].append(title)
            
            appear_area = c.find('div', {'class':'l-appearArea'})
            if appear_area:
                replies = appear_area.find_all('div', {'class':'l-leaveMsg__msg'})
                for ind, r in enumerate(replies):
                    content_dict['author'].append(r.find('a', {'class':'c-link c-link--gn u-username'}).text.strip())
                    content_dict['post_time'].append(r.find('span', {'class':'o-fSubMini o-fNotes'}).text.strip())
                    content_dict['floor'].append(floor+'-'+str(ind+1))
                    content_dict['content'].append(r.find('div', {'class':'msgContent c-summary__desc'}).get_text(' ').strip())
                    content_dict['cid'].append(self.county_id)
                    content_dict['title'].append(title)
        return content_dict
    
    def create_driver(self):
        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
        prefs = {"profile.managed_default_content_settings.images": 2}
        
        opt = uc.ChromeOptions() 
        opt.add_experimental_option("prefs", prefs)
        opt.add_argument('--disable-gpu')
        opt.add_argument('--user-agent=%s' % user_agent)
        opt.add_argument("--disable-notifications")
        opt.add_argument("--disable-extensions")
        opt.add_argument("--disable-blink-features=AutomationControlled")
        opt.add_argument("--headless")
        driver = uc.Chrome(use_subprocess=True, options=opt) 
        
        return driver

    def try_get_url(self, driver, url):
        try: 
            driver.get(url)
        except:
            print('Fail to get url, will try to get again in 300s.')
            time.sleep(300)
            driver.get(url)

    def get_topic_urls(self, soup):
        topic_dict = self.topic_dict
        table = soup.find_all("div", {"class": "l-listTable__tbody"})[0]
        temp_topic = table.find_all('a', {'class': 'c-link u-ellipsis'})
        temp_time = table.find_all('div', {'class': 'o-fNotes'})
        print(temp_time)
        for ind in range(len(temp_topic)):
            topic_dict['url'].append(temp_topic[ind].get('href'))
            topic_dict['title'].append(temp_topic[ind].text)
            topic_dict['first_post_time'].append(temp_time[ind*2].text)
            topic_dict['last_post_time'].append(temp_time[ind*2+1].text)
        return topic_dict
        

    def get_final_page(self, soup):
        page_list = soup.find_all("a", {"class": "c-pagination"})
        
        if len(page_list) == 0:
            final_page = '1'
        else :
            final_page = page_list[-1].text
        return int(final_page)


