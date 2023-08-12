from collections import defaultdict
import jieba
import numpy as np
import pandas as pd
import pickle
from util_text_cleaner.text_cleaner import TextCleaner
from util_model.model_helper import GenInput
from util_model.feature_kbest import FeatureKBest

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from util_model.model_helper import OptimizedXGB 
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from xgboost import XGBClassifier
import logging


LOGGER = logging.getLogger(__name__)

if __name__ == '__main__':
    
    # Load data
    data = pd.read_pickle('data/data_labeled.pkl')
    data['label'] = data['label_a'] - data['label_d']
    data['label'] = data['label'].map({-1:0, 0:1, 1:2})
    
    # Clean text and generate cut and corpus
    tc = TextCleaner()
    data['content_cleaned'] = tc.clean_text(data['content'])
    data['cut'], data['corpus'] = tc.lcut_to_corpus(data['content_cleaned'])
    
    # Generate 1 - 5 gram cut
    # Since I don't tune the `n` gram in this demo, generate `n_gram_cut` here \\
    # rather than in the training process will save a lot of training time. 
    f_kb = FeatureKBest(k = 3000)
    data['n_gram_cut'] = f_kb.gen_gram_cut(data['content_cleaned'], max_n = 5)
    
    # Set some parameters
    random_state = 100
    scoring = 'f1_weighted' # I use f1_weighted as the metric because we are facing a unblanaced multiclass problem
    skf5 = StratifiedKFold(5, shuffle = True, random_state = random_state) # Predefined  5 fold for hyperparameter tuning.
    model_result_dict = defaultdict(dict)
    
    # Define training and test datasets
    train = data.sample(frac = 0.8, random_state = random_state)
    test = data[~data.index.isin(train.index)]
    print(f'Size of train: {train.shape[0]}')
    print(f'Size of test: {test.shape[0]}')
    
    y_train = train['label']
    y_test = test['label']
    
    result = defaultdict(dict)
    xgb_model = OptimizedXGB(random_state=random_state)
    xgb_model.fit(train, y_train, cv=skf5)
    
    y_pred = xgb_model.predict(test)
    f1 = f1_score(y_test, y_pred, average = 'weighted')
    cm = confusion_matrix(y_test, y_pred)

    print(f'Best parameters : {xgb_model.best_params}')
    print(f'f1 : {f1}')
    
    filename = 'result/model_result/xgb_model3.pkl'
    pickle.dump(xgb_model, open(filename, 'wb'))
    print('Model_saved!')
    
