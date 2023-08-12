from gensim.models import Word2Vec
import jieba
import numpy as np
import pandas as pd



class FeatureWord2Vec:
    '''
        Generate word2vec features
    '''
    def __init__(self) -> None:
        
        # Load pretrained word2vec model
        self.__model = Word2Vec.load("result/word2vec_result/word2vec.model300")
        self.__Word2Vec_size = 300
    
    def cut2wordembed_mean(self, cut:list, vector_size:int, index2word_set:dict) -> np.array:
        '''
            Generate word2vector feature from a cut
        '''
        # Create pre-initialize (for speed) feature vector
        feature_vec = np.empty((0, vector_size), dtype = "float32")  
        
        # Get pre-trained word vector
        for word in cut:
            if word in index2word_set: 
                vector = self.__model.wv[word]
                feature_vec = np.append(feature_vec, [vector], axis = 0)
                
        # Average the word vectors of text(from nX300 to 1X300)
        feature_vec = np.mean(feature_vec, axis = 0)
        return feature_vec  
    
    def gen_cut2wordembed_mean(self, col:pd.Series) -> np.array:
        '''
            Generate word2vect features from cuts
        '''
        # Get words known to the pre-trained word2vec model
        index2word_set = set(self.__model.wv.index_to_key)  
        
        # Generate word2vect features
        result_array = list(map(lambda x : self.cut2wordembed_mean(x, vector_size = self.__Word2Vec_size, index2word_set=index2word_set), col))
        return result_array
    
    def gen_corpus_from_cut(self, col:pd.Series) -> pd.Series:
        '''
            Generate cut from corpus
        '''
        return list(map(lambda x: ' '.join(x), col))
    
    @property
    def word2vec_size(self) -> int:
        return self.__Word2Vec_size