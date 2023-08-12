import jieba
import numpy as np
import pandas as pd
import re
import yaml



class TextCleaner:
    '''
        Clean text for future analysis.
    '''
    def __init__(self, config_loc:str = 'util_text_cleaner/config.yaml') -> None:
        
        # load config
        self.config_loc = config_loc
        with open(config_loc, 'r') as stream:
            config = yaml.load(stream, Loader = yaml.FullLoader)
        self.replace_dict = config['replace_dict']
        
    def replace_text(self, col:pd.Series) -> pd.Series:
        '''
            replace words in replace_dict. 
        '''
        new_col = col.str.lower()
        new_col = col.replace(self.replace_dict, regex = True)
        return new_col

    def clean_text(self, col:pd.Series) -> pd.Series:
        '''
            clean text.
        '''
        
        # replace words
        new_col = self.replace_text(col)
        
        # delete names
        new_col = new_col.replace('[A-Za-z]+大', ' name ', regex = True)
        new_col = new_col.replace('[A-Za-z]+君', ' name ', regex = True)
        new_col = new_col.replace('name', '')
        
        # delete symbols
        new_col = new_col.replace('[^\w!?！？]', ' ',  regex = True)
        new_col = new_col.replace('\s+', ' ',  regex = True)
        
        # strip words
        new_col = new_col.str.strip()
        return new_col

    def to_corpus(self, col:pd.Series) -> pd.Series:
        '''
            From cut to corpos
        '''
        corpus_col = ' '.join(col)
        return corpus_col
    
    def lcut_to_corpus(self, col:pd.Series)->pd.Series:
        '''
            From text to corpus
        '''
        
        # load dictionary
        jieba.load_userdict('util_text_cleaner/add_words.txt')
        jieba.set_dictionary('util_text_cleaner/dict.txt')
        
        # cut words
        cut_col = col.apply(lambda x : jieba.lcut(x))
        
        # generate corpos
        corpus_col = list(map(lambda x: ' '.join(x), cut_col))
        return cut_col, corpus_col
   