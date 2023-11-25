from collections import defaultdict
import pandas as pd
import numpy as np
from util_model.feature_word2vec import FeatureWord2Vec
from util_model.feature_kbest import FeatureKBest
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.early_stop import no_progress_loss
from sklearn.model_selection import train_test_split
import sklearn.model_selection
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, log_loss



class GenInput:
    '''
        Generate input for model training.
    '''
    def __init__(self, k:int, standardized:bool, fillna:bool = False) -> None:
        '''
            k_best : int, how many words are use to train models under chi square feature selection for the n_grams.
            standardized : boolean, whether to standardize the word2vector features.
            fillna : boolean, whether to fill NaN with 0 for the word2vector features.
        '''
        self.__k = k
        self.__standardized = standardized
        self.__kbest_cols = None
        self.__w2v_cols = None
        self.__fillna = fillna
        
    def fit(self, x_train:pd.DataFrame, y_train:pd.Series) -> None:
        '''
            Fit x_train and y_train.
        '''
        x_train_copy = x_train.copy()
        
        # fit k_best
        self.__f_kb = FeatureKBest(self.__k)
        self.__kbest_cols = self.__f_kb.gen_kbest_words(x_train_copy['n_gram_cut'], y_train)
        
        # fit w2v
        self.__f_w2v = FeatureWord2Vec()
        vector_size = self.__f_w2v.word2vec_size
        self.__w2v_cols = [f'v_{i}' for i in range(vector_size)]
        
        # if standardized=True, then fit the scaler
        if self.__standardized:
            self.__scaler = StandardScaler()
            x_train_copy[self.__w2v_cols] = self.__f_w2v.gen_cut2wordembed_mean(x_train_copy['cut'])
            self.__scaler.fit(x_train_copy[self.__w2v_cols])
        
        print('feature fitted.')

    def transform(self, x_train:pd.DataFrame) -> pd.DataFrame:
        '''
            Generate Input features. There will be K_best + 300(word2vec) features for model training.
        '''
        
        # Check if fitted
        if ((not hasattr(self, '_GenInput__kbest_cols')) | (not hasattr(self, '_GenInput__w2v_cols'))):
            raise NotFittedError('Call `fit` before `transform`.')
            
        else:
            
            x_train_copy = x_train.copy()
            
            # Generate Word2Vector features
            print('Generate Word2Vector features.')
            x_train_copy[self.__w2v_cols] = self.__f_w2v.gen_cut2wordembed_mean(x_train_copy['cut'])
            if self.__standardized:
                x_train_copy[self.__w2v_cols] = self.__scaler.transform(x_train_copy[self.__w2v_cols])
            if self.__fillna:
                x_train_copy[self.__w2v_cols] = x_train_copy[self.__w2v_cols].fillna(0)
            print('Finished.')
            
             # Generate Kbest features
            print(f'Generate {self.__k} KBest features.')
            x_train_copy[self.__kbest_cols] = self.__f_kb.gen_kbest_features(x_train_copy['n_gram_cut'])
            print('Finished.')
            
            input_cols = self.__kbest_cols + self.__w2v_cols
            x_input_train = x_train_copy[input_cols]
            
            return x_input_train

    def fit_transform(self, x_train:pd.DataFrame, y_train:pd.DataFrame) -> pd.DataFrame:
        '''
            Fit first and transform.
        '''
        self.fit(x_train, y_train)
        return self.transform(x_train)

    @property
    def kbest_cols(self):
        return self.__kbest_cols

    @property
    def k(self):
        return self.__k

    @property
    def standardized(self):
        return self.__standardized

    @property
    def fillna(self):
        return self.__fillna

    @property
    def StandardScaler(self):
        return self.__scaler
    
class OptimizedXGB(BaseEstimator, ClassifierMixin):
    '''
        XGB with optimized parameters.
    '''
    def __init__(self, random_state, custom_params_space=None) -> None:

        self.__random_state = random_state
        self.__cv_results = defaultdict(dict)
        self.__best_params = defaultdict(list)

        if custom_params_space:
            self.__custom_params_space = custom_params_space
        else :
            self.__custom_params_space = {
                'learning_rate': hp.uniform('learning_rate', 0.001, 0.05),
                'max_depth': hp.quniform('max_depth', 3, 10, 1),
                'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1),
                'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
                'gamma': hp.quniform('gamma', 0.8, 1.2, 0.05),
                'colsample_bytree': hp.quniform('colsample_bytree', 0.3, 0.7, 0.05),
                'colsample_bylevel': hp.quniform('colsample_bylevel', 0.3, 0.7, 0.05),
                'colsample_bynode': hp.quniform('colsample_bynode', 0.3, 0.7, 0.05),
                'reg_lambda': hp.loguniform('reg_lambda', np.log(0.0001), np.log(100)),
                'reg_alpha': hp.loguniform('reg_alpha', np.log(0.0001), np.log(100)),
                'random_state' : self.__random_state
            }

    def fit(self, train:np.array, y_train:np.array, cv) -> None:
        '''
            Train a XGB model with cv.
        '''
        
        _space = self.__custom_params_space
        
        if isinstance(cv, float) : 
            if ((cv <= 0) | (cv >=1)):
                raise ValueError('percentage cv should (0, 1).')
                
            # Split train into train/validation
            X_train, X_val, Y_train, Y_val = train_test_split(train, y_train, test_size = cv, stratify = y_train, random_state = self.__random_state)
            gen_input_train = GenInput(k = 3000, standardized = True, fillna = False)
            X_input_train = gen_input_train.fit_transform(X_train, Y_train)  
            X_input_val = gen_input_train.transform(X_val)
            
            # train model based on params_space
            opt, trial = self.optimize_params(X_train = X_input_train, y_train = Y_train, X_val = X_input_val, y_val = Y_val, params_space = _space)
            
            opt['n_estimators'] = trial.best_trial['result']['n_estimators']
            f1 = 1 - trial.best_trial['result']['loss']
            
            # save result
            index = 0
            print(f'cv {index} training result:')
            print(opt)
            print(f'f1_score: {f1}')
            self.__cv_results[index]['params'] = opt
            self.__cv_results[index]['f1'] = f1

        elif isinstance(cv, sklearn.model_selection._split.StratifiedKFold):
            
            # for loop cv and optimized based on train and val datasets 
            for index, value in enumerate(cv.split(train, y_train)):

                print(f'Training cv {index}')
                train_index = value[0]
                test_index = value[1]
                X_train, X_val = train.iloc[train_index], train.iloc[test_index]
                Y_train, Y_val = y_train.iloc[train_index], y_train.iloc[test_index]
                
                # generate X features
                gen_input_train = GenInput(k = 3000, standardized = True, fillna = False)
                X_input_train = gen_input_train.fit_transform(X_train, Y_train)
                X_input_val = gen_input_train.transform(X_val)

               # train model based on params_space                
                opt, trial = self.optimize_params(X_train = X_input_train, y_train = Y_train, X_val = X_input_val, y_val = Y_val, params_space = _space)

                opt['n_estimators'] = trial.best_trial['result']['n_estimators']
                f1 = 1 - trial.best_trial['result']['loss']

                # save result
                print(f'cv {index} training result:')
                print(opt)
                print(f'f1_score: {f1}')
                self.__cv_results[index]['params'] = opt
                self.__cv_results[index]['f1'] = f1

        # if cv is not float or StratifiedKFold, raise TypeError
        else:
            raise TypeError('cv should be either a percentage or a StratifiedKFold.')
        print(self.__cv_results)

        # Calculate mean of the best hyperparameters    
        for ind, value in self.__cv_results.items(): 
            for p, v in value['params'].items():
                self.__best_params[p].append(v)

        for k, v in self.__best_params.items():
            self.__best_params[k] = np.mean(v)

        self.__best_params['n_estimators'] = int(self.__best_params['n_estimators'])
        self.__best_params['max_depth'] = int(self.__best_params['max_depth'])
        self.__best_params['random_state'] = self.__random_state
        print(self.__best_params)
        


        # Instantiate `xgboost.XGBClassifier` with optimized parameters
        best = XGBClassifier(objective = 'multi:softprob',
                                        tree_method = 'hist',
                                        **self.__best_params)

        self.gen_input = GenInput(k = 3000, standardized = True, fillna = False)
        x_input_train = self.gen_input.fit_transform(train, y_train)
        best.fit(x_input_train, y_train)
            
        self.__best_estimator_ = best

    def predict(self, X:np.array) -> np.array:
        '''
            Predict labels with trained XGB model.
        '''
        if not hasattr(self, '_OptimizedXGB__best_estimator_'):
            raise NotFittedError('Call `fit` before `predict`.')
        else:
            x_input_train = self.gen_input.transform(X)
            return self.__best_estimator_.predict(x_input_train)

    def predict_proba(self, X:np.array) -> np.array:
        '''
            Predict labels probaiblities with trained XGB model.
        '''
        if not hasattr(self, '_OptimizedXGB__best_estimator_'):
            raise NotFittedError('Call `fit` before `predict_proba`.')
        else:
            x_input_train = self.gen_input.transform(X)
            return self.__best_estimator_.predict_proba(x_input_train)
    
    def optimize_params(self, X_train, y_train, X_val, y_val, params_space):
        '''
            Hypterparameter tuning with hyperopt 
        '''

        # Estimate XGB params
        def objective(_params):
            _clf = XGBClassifier(n_estimators = 15000,
                                 max_depth = int(_params['max_depth']),
                                 learning_rate = _params['learning_rate'],
                                 min_child_weight = _params['min_child_weight'],
                                 subsample = _params['subsample'],
                                 colsample_bytree = _params['colsample_bytree'],
                                 colsample_bylevel = _params['colsample_bylevel'],
                                 colsample_bynode = _params['colsample_bynode'],
                                 gamma = _params['gamma'],
                                 reg_alpha = _params['reg_alpha'],
                                 reg_lambda = _params['reg_lambda'],
                                 objective = 'multi:softprob',
                                 tree_method = 'hist',
                                 early_stopping_rounds = 30,
                                 eval_metric = 'mlogloss',
                                 random_state = self.__random_state
                                )
            _clf.fit(X_train, y_train, eval_set = [(X_val, y_val)])
            y_pred = _clf.predict(X_val)
            f1 = f1_score(y_val, y_pred, average = 'weighted')
            best_iteration = _clf.best_iteration
            return {'loss' : 1. - f1, 'status' : STATUS_OK, 'n_estimators' : best_iteration}

        trials = Trials()
        return fmin(fn = objective,
                space = params_space,
                algo = tpe.suggest,
                max_evals = 100,
                trials = trials,
                early_stop_fn = no_progress_loss(30),
                rstate = np.random.default_rng(self.__random_state),
                verbose = 1), trials
    
    @property
    def cv_results(self):
        return self.__cv_results

    @property
    def best_params(self):
        return self.__best_params

    @property
    def training_params_space(self):
        return self.__custom_params_space

    @property
    def random_state(self):
        return self.__random_state

    @property
    def best_estimator_(self):
        return self.__best_estimator_

class SelfDefinedMetric:
    
    def __init__(self):
        pass
    
    def f1_scorer(self, y_true, y_pred):
        y_pred = np.argmax(y_pred, axis = 1)
        err = 1 - f1_score(y_true, y_pred, average = 'weighted')
        return err
    