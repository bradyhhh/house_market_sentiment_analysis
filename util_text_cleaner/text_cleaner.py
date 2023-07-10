import pandas as pd
import numpy as np
import re
from jieba import lcut
import yaml

class text_cleaner:
    
    def __init__(self, config_loc:str='util_text_cleaner/config.yaml')->None:
        self.config_loc = config_loc
        with open(config_loc, 'r') as stream:
            config = yaml.load(stream, Loader = yaml.FullLoader)
        self.replace_dict = config['replace_dict']
        
    def replace_text(self, col:pd.Series)->pd.Series:
        new_col = col.str.lower()
        new_col = col.replace(self.replace_dict, regex=True)
        return new_col

    def clean_text(self, col:pd.Series)->pd.Series:
        
        new_col = self.replace_text(col)
        new_col = new_col.replace('[A-Za-z]+大', ' name ', regex=True)
        new_col = new_col.replace('[A-Za-z]+君', ' name ', regex=True)
        new_col = new_col.replace('name', '')
        new_col = new_col.replace('[^\w!?！？]', ' ',  regex=True)
        new_col = new_col.replace('\s+', ' ',  regex=True)
        return new_col

    def to_corpus(self, col:pd.Series)->pd.Series:
        
        corpus_col = ' '.join(col)
        return corpus_col
    
    def lcut_to_corpus(self, col:pd.Series)->pd.Series:
        
        new_col = col.apply(lambda x : jieba.lcut(x))
        corpus_col = ' '.join(new_col)
        return corpus_col
    
