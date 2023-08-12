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
    '''
        Scrapper can be used to scrape threads related to the house market on the mobile01 website.   
    '''
    
    def __init__(self, county_id:str) -> None:
        '''
            Specify the country_id of the country you want to parse. 
        '''
        
        self.county_id = county_id
        self.county_url = f'https://www.mobile01.com/topiclist.php?f={county_id}'
        self.save_path = f'result/scraped_result/{self.county_id}/'
        self.save_dt = strftime("%Y%m%d", localtime())
        self.__topic_summary_dict = {'url' : [], 'title' : [], 'first_post_time' : [], 'last_post_time' : [], 'cid' : []}
        self.__topic_content_dict = {'title' : [], 'content' : [], 'author' : [], 'post_time' : [], 'floor':[], 'cid' : []}
    
    def scrape_end2end(self) -> None:
        '''
            Scrape threads of the county on the website end to end.
        '''
        # scrape topic summary and update self.topic_summary_dict
        self.scrape_topic_summary()
        topic_summary_df = self.topic_summary_df
        
        # remove duplications
        topic_summary_df = topic_summary_df.sort_values(by = 'last_post_time', ascending = False)
        topic_summary_df = topic_summary_df.reset_index(drop = True)
        topic_summary_df = topic_summary_df[ ~ topic_summary_df[['topic', 'url']].duplicated()]
        self.__topic_summary_dict = topic_summary_df.to_dict('list')
        
        # scrape topic content and update self.topic_content_dict
        self.scrape_topic_by_urls(topic_summary_df = topic_summary_df)
    
    def scrape_topic_by_urls(self, file_name_topic_summary:str = 'None', file_name_topic_content:str = 'None') -> None:
        '''
            Scrape topics by their urls. 
        '''
        
        self.__topic_content_dict = self.check_parsed_files(file_name = file_name_topic_content, file_type='topic_content')
        self.__topic_summary_dict = self.check_parsed_files(file_name = file_name_topic_summary, file_type='topic_summary')

        topic_summary_df = self.topic_summary_df
        url_list = topic_summary_df['url'].tolist()
        
        # Scrape topic content by its url.
        for ind, url in enumerate(url_list):
            
            print('parsing: ', topic_summary_df.iloc[ind]['title'])
            self.scrape_topic(url = url)
            
    
    def scrape_topic(self, url: str, file_name_topic_content:str = 'None', start_page:int = 1) -> None:
        '''
            Scrape discussion contents from a topic page by page and update self.__topic_content_dict.
        '''
        
        driver = self.create_driver()
        final_page = start_page
        self.__topic_content_dict = self.check_parsed_files(file_name = file_name_topic_content, file_type = 'topic_content')
        
        # Parse topic content from its first page to its final page.
        while start_page <= final_page:
            topic_url = f'https://www.mobile01.com/{url}&p={start_page}'

            # start time
            start = time.time()
            
            self.try_get_url(driver, topic_url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # If final page is not checked, then parse the final page of the threads of topic on the website.
            if final_page == start_page:
                final_page = self.get_final_page(soup)
            
            # Parse topic content and update topic_content_dict
            print(f'start to parse page {start_page}/{final_page}')
            temp_dict = self.scrape_topic_content(soup)
            for k, v in temp_dict.items():
                self.__topic_content_dict[k] += v
            
            # end time
            end = time.time()
            sec = end - start
            print(f"Time spend: {sec:.2f} seconds")
            
            # save scraped result
            self.save_dt = strftime("%Y%m%d", localtime())
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)
            save_file_name = self.save_path + f'topic_content_{self.save_dt}.pkl'
            content_df = pd.DataFrame(self.__topic_content_dict)
            content_df.to_pickle(save_file_name)
            
            start_page += 1
            
            print('Start to sleep')
            self.sleep(5, random_time = True)
            print('End of sleeping')
            
        driver.close()
        
    
    def scrape_topic_content(self, soup) -> dict:
        '''
            Parse contents in a page and return topic_content_dict.
        '''
        
        topic_content_dict = {'title' : [], 'content' : [], 'author' : [], 'post_time' : [], 'floor' : [], 'cid' : []}
        title = soup.find('h1', {'class' : 't2'}).text.strip()
        contents = soup.find_all("div", {"class" : "l-articlePage"})
        
        # parse content by floors
        for c in contents:
            # find author
            author = c.find('div', {'class' : 'c-authorInfo__id'})
            if not author:
                continue
            topic_content_dict['author'].append(author.text.strip())
            
            # find post_time and floor
            navigation = c.find(lambda tag : tag.name == 'div' and 
                                   tag.get('class') == ['l-navigation__item'])
            topic_content_dict['post_time'].append(navigation.find_all('span', {'class' : 'o-fNotes o-fSubMini'})[0].text.strip())
            floor = navigation.find_all('span', {'class' : 'o-fNotes o-fSubMini'})[1].text.replace('#', '').strip()
            topic_content_dict['floor'].append(floor)
            
            # find content 
            all_text = c.find('article').get_text(' ')
            # remove quotation
            quote_list = c.find('article').find_all('blockquote')
            for quote in quote_list:
                # replace <\br> to ' '
                quote_text = quote.get_text(' ')
                all_text = all_text.replace(quote_text, '')
            topic_content_dict['content'].append(all_text.strip())
            topic_content_dict['cid'].append(self.county_id)
            topic_content_dict['title'].append(title)
            
            # check whether there are replies in a floor
            appear_area = c.find('div', {'class' : 'l-appearArea'})
            if appear_area:
                replies = appear_area.find_all('div', {'class' : 'l-leaveMsg__msg'})
                for ind, r in enumerate(replies):
                    topic_content_dict['author'].append(r.find('a', {'class' : 'c-link c-link--gn u-username'}).text.strip())
                    topic_content_dict['post_time'].append(r.find('span', {'class' : 'o-fSubMini o-fNotes'}).text.strip())
                    topic_content_dict['floor'].append(floor + '-' + str(ind + 1))
                    topic_content_dict['content'].append(r.find('div', {'class' : 'msgContent c-summary__desc'}).get_text(' ').strip())
                    topic_content_dict['cid'].append(self.county_id)
                    topic_content_dict['title'].append(title)
        return topic_content_dict

    def scrape_topic_summary(self, final_page:int = 0, file_name:str = 'None') -> None:
        '''
            Scrape topic summaries and update self.__topic_summary_dict.
        '''
        
        driver = self.create_driver()
        self.__topic_summary_dict = self.check_parsed_files(file_name = file_name, file_type = 'topic_summary')

        # If final page is not specified, then parse the final page of the threads of county on the website.
        if final_page == 0:
            
            url = self.county_url
            self.try_get_url(driver, url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            print('Start to sleep')
            self.sleep(5, random_time = True)
            print('End of sleeping')
        
            # parse final page
            final_page = self.get_final_page(soup)
            print('End of parsing final_page')
            
        # Scrape the topic summary from final page to the first page
        for p in range(1, final_page + 1)[:: -1]:
            print(f'Go to page {p}')
            
            # start time
            start = time.time()
            
            page_url = self.county_url + f'&p={p}'
            self.try_get_url(driver, page_url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            self.__topic_summary_dict = self.get_topic_summary(soup)
            
            # end time
            end = time.time()
            sec = end - start
            print(f"Time spend: {sec:.2f} seconds")
            
            # save scraped result
            self.save_dt = strftime("%Y%m%d", localtime())
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)
            file_name = self.save_path + f'topic_summary_{self.save_dt}.pkl'
            pd.DataFrame(self.__topic_summary_dict).to_pickle(file_name)
            
            print('Start to sleep')
            self.sleep(5, random_time=True)
            print('End of sleeping')
            
        driver.close()
    
    def get_topic_summary(self, soup) -> dict:
        '''
            Get topic summary in a page and update self.topic_summary_dict.
        '''
        topic_summary_dict = self.__topic_summary_dict
        table = soup.find_all("div", {"class" : "l-listTable__tbody"})[0]
        temp_topic = table.find_all('a', {'class' : 'c-link u-ellipsis'})
        temp_time = table.find_all('div', {'class' : 'o-fNotes'})
        for ind in range(len(temp_topic)):
            topic_summary_dict['url'].append(temp_topic[ind].get('href'))
            topic_summary_dict['title'].append(temp_topic[ind].text)
            topic_summary_dict['first_post_time'].append(temp_time[ind * 2].text)
            topic_summary_dict['last_post_time'].append(temp_time[ind * 2 + 1].text)
            topic_summary_dict['cid'].append(self.county_id)
        return topic_summary_dict
                    
    def get_final_page(self, soup) -> int:
        '''
            Find the final page.
        '''
        page_list = soup.find_all("a", {"class" : "c-pagination"})
        
        if len(page_list) == 0:
            final_page = '1'
        else :
            final_page = page_list[-1].text
            
        return int(final_page)
    
    def check_parsed_files(self, file_name : str, file_type : str) -> dict:
        '''
            Check whether to use self.__topic_summary_dict/__self.topic_content_dict or to import them from `file_name`.
        '''
        file_path = self.save_path + file_name
        if file_type == 'topic_summary':
            if file_name == 'None':
                result_dict = self.__topic_summary_dict
            elif not os.path.exists(file_path):
                raise ValueError(f"No such file:{file_path}")
            else :
                result_df = pd.read_pickle(file_path)
                if str(result_df['cid'].iloc[0]) != str(self.county_id):
                    raise ValueError(f"Please check the county_id. Initialized cid is not equal to the cid in file:{file_name}")
                result_dict = result_df.to_dict('list')
                
        elif file_type == 'topic_content':
            if file_name == 'None':
                result_dict = self.__topic_content_dict
            elif not os.path.exists(file_path):
                raise ValueError(f"No such file:{file_path}")
            else :
                result_df = pd.read_pickle(file_path)
                if str(result_df['cid'].iloc[0]) != str(self.county_id):
                    raise ValueError(f"Please check the county_id. Initialized cid is not equal to the cid in file:{file_name}")
                result_dict = result_df.to_dict('list')
        else:
            raise ValueError("file_type can only be either 'topic_summary' or 'topic_content'.")
        
        return result_dict
    
    def create_driver(self):
        '''
            Create a driver with option arguments.
        '''
        
        opt = uc.ChromeOptions() 
        opt.add_argument('--disable-gpu')
        opt.add_argument("--disable-notifications")
        opt.add_argument("--disable-extensions")
        opt.add_argument("--disable-blink-features=AutomationControlled")
        opt.add_argument("--headless")
        driver = uc.Chrome(use_subprocess = True, options = opt) 
        
        return driver

    def try_get_url(self, driver, url:str) -> None:
        '''
            set the url to the driver. Will try again after 300s if it fails. 
        '''
        try: 
            driver.get(url)
        except:
            print('Fail to get url, will try to get again in 300s.')
            self.sleep(300, random_time = False)
            driver.get(url)

    def sleep(self, t, random_time = True) -> None:
        '''
            Sleep for {t} + random seconds to avoid getting banned.
        '''
        if random:
            r = random.random()
            if r <= 0.5:
                time.sleep(t+random.random() * 10)
            elif r <= 0.9:
                time.sleep(t+random.random() * 20)
            else:
                time.sleep(t+random.random() * 50)
        else:
            time.sleep(t)
            
    @property
    def topic_summary_df(self):
        return pd.DataFrame(self.__topic_summary_dict)
    
    @property
    def topic_content_df(self):
        return pd.DataFrame(self.__topic_content_dict)
    
    @property
    def topic_summary_dict(self):
        return self.__topic_summary_dict
    
    @property
    def topic_content_dict(self):
        return self.__topic_content_dict
    