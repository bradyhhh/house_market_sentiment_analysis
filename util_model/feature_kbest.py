import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



class FeatureKBest:
    '''
        Generate K best word features
    '''
    def __init__(self, k:int) -> None:
        '''
            Define K. 
        '''
        self.__k = k
        self.__k_best_words = []
        pass
    
    def gen_text2n_gram_dict(self, text:str, n:int) -> dict:
        '''
            Generate `n`_gram dictionary from a text
        '''
        dic = {}
        text_length = len(text)
        last_index = text_length - (n - 1)
        
        # for loop text index 
        for i in range(last_index):
            
            # cut text into n_gram
            gram = text[i : i + n]
            
            # count the time of occurance of a n_gram word
            if gram in dic.keys():
                dic[text[i : i + n]] += 1
            else :    
                dic[text[i : i + n]] = 1
        return dic
    
    def gen_gram_cut(self, col:pd.Series, max_n:int = 5, min_n:int = 1) -> pd.Series:
        '''
            Generate n_gram cut from min_n to max_n.
        '''
        length = len(col)
        gram_list = [[] for x in range(length)]
        
        # for loop from min_n to max_n, and append the gram_list
        for n in range(min_n, max_n + 1):
            
            # Generate dictionaries which contain the `n`_gram words and their counts in texts
            n_gram_dict = list(map(lambda x: self.gen_text2n_gram_dict(x, n), col))
            
            # Generate lists which contain the `n`_gram word in the n_gram_cut
            n_gram_list = list(map(lambda x: list(x.keys()), n_gram_dict))
            
            # append n_gram_list to the gram_list
            gram_list = [l1_row + l2_row for l1_row, l2_row in zip(gram_list, n_gram_list)]
        return gram_list
    
    def gen_kbest_words(self, x_train_cut:pd.Series, y_train:pd.Series) -> list:
        '''
            Genreate the K best words in the training dataset.
        '''
        
        # get the corpus and count the occurance of words
        x_train_corpus = list(map(lambda x: ' '.join(x), x_train_cut))
        vectorizer = CountVectorizer()
        x_features = vectorizer.fit_transform(x_train_corpus)
        
        # chi square feature selections for the features
        ch2 = SelectKBest(chi2, k = self.__k)
        x_features_select = ch2.fit_transform(x_features, y_train)
        
        # get the words
        k_best_index = ch2.get_support()
        words = np.array(vectorizer.get_feature_names())
        self.__k_best_words = list(words[k_best_index])
        return self.__k_best_words

    def gen_word_feature(self, word:str, cut_list:list) -> int:
        '''
            Generate dummay value to check if the word in the cut list
        '''
        if word in cut_list:
            dummy = 1
        else:
            dummy = 0
        return dummy

    def gen_kbest_features(self, cut:pd.Series) -> np.array:
        '''
            Generate Kbest features  
        '''
        
        # Create the set of cuts of texts
        cut = list(map(lambda x: set(x), cut))
        
        # For word in kbest words, genereate its dummy variable
        array_list = []
        for word in self.__k_best_words:
            word_array = np.array(list(map(lambda x:self.gen_word_feature(word, x), cut)))
            array_list.append(word_array)
        
        # change into np.array and return
        word_arrays = np.column_stack(array_list)
        return word_arrays

    @property
    def k(self):
        return self.__k

    @property
    def k_best_words(self):
        return self.__k_best_words
    