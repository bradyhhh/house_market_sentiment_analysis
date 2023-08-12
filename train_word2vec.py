import pandas as pd
import numpy as np
import re
import os
import logging

import jieba
from gensim.models import Word2Vec

from util_text_cleaner.text_cleaner import TextCleaner

LOGGER = logging.getLogger(__name__)

if __name__ == '__main__':

    # load data for training
    print("Loading data.")
    df_1 = pd.read_pickle('data/all_county_data.pkl')
    df_2 = pd.read_pickle('data/news.pkl')
    df_3 = pd.read_pickle('data/no_county_data.pkl')

    # concat data
    data = pd.DataFrame()
    data['content'] = pd.concat([df_1['content'], df_2['content'], df_3['content']])
    # drop non string contents
    data = data[data['content'].apply(lambda x: isinstance(x, str))]
    data = data.reset_index(drop=True)
    print('Finished.')
    print(f'Data shapes: {data.shape[0]}')

    # load self-defined dictionaries
    print("Loading self-defined dictionaries.")
    jieba.set_dictionary('util_text_cleaner/dict.txt')
    jieba.load_userdict('util_text_cleaner/add_words.txt')
    print('Finished.')

    # clean text
    print('Cleaning contents.')
    tc = TextCleaner()
    data['content_cleaned'] = tc.clean_text(data['content'])
    print('Finished.')
    
    # implement Jieba word segementation 
    print('Start to word segementation.')
    data['cut'] = data['content_cleaned'].apply(lambda x: jieba.lcut(x))
    print('Finished.')

    # train word2vec model
    print('Start to train word2vec model.')
    model300 = Word2Vec(data['cut'], min_count = 5, vector_size = 300)
    
    # save model
    model300.save("result/word2vec_result/word2vec.model300")
    print('Model saved.')