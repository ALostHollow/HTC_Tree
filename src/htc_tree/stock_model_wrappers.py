import pandas as pd
import numpy as np
import time

# Sklearn Imports:
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# XGBoost Imports:
from xgboost import XGBClassifier

class SKLearn_Model_Wrapper():
    """
    This Simple Model supports a set of SKLearn models.

    This class should be replaced with classes supporting other models with these same key methods:
    1. Train
      - data: (pd.DataFrame) the data
      - 
    2. score:
      - 
      - 
    3. Predict:
      - 
      - 
    """

    # Class Methods Overloads:
    def __init__(self, estimator, tfidf:TfidfVectorizer, paramater_grid:dict, test_size:float=0.2, verbose=False) -> None:
        self.verbose = verbose # if True show process info
        self.estimator = estimator
        self.tfidf = tfidf # the vectorizer used in both Train and Predict
        self.param_grid = paramater_grid
        self.test_size = test_size # The test size used in Trian

        self.ready = False
        self.best_params = None
        self.test_acc = 0 # used in comparison so should be set on __init__

    def __repr__(self) -> str:
        return str(self.estimator).replace('()', '')
    
    # Class Methods:
    def Train(self, features: list | np.ndarray | pd.DataFrame, targets: list | np.ndarray | pd.DataFrame, **kwargs) -> None:
        '''
        Runs the training operations for this model.

        args:
            features:   (list | ndarray | DataFrame) The arraylike to use as input in training.
            targets:    (list | ndarray | DataFrame) The arraylike to use as targets for training.

        kwargs:
            use_best: (bool) False
            search_method: (str) 'Random'. could be 'Random' or 'Grid'
            n_iter: (int) 20
            cv_folds: (int) 2
            n_jobs: (int) -2

        Returns:
            None
        '''
        
        # Parsing keyword arguments:
        parse = lambda x, arg, defalt : x[arg] if arg in x else defalt # default value is 20
        #   use_best:
        use_best = 'use_best' in kwargs and kwargs['use_best']
        #   search_method:
        if 'search_method' in kwargs: search_method = kwargs['search_method']
        else: search_method = 'Random' # default value is 'Random'
        #   n_iter:
        n_iter = parse(kwargs, 'n_iter', 20)
        #   cv_folds:
        cv_folds = parse(kwargs, 'cv_folds', 2)
        #   n_jobs:
        n_jobs = parse(kwargs, 'n_jobs', -2)

        if self.verbose: 
            start_time = time.time()

        # Generating parameter grid to use for search:
        if use_best and self.best_params != None:
            parameters = {}
            for param in self.best_params.keys():
                parameters[param] = [self.best_params[param]]
            search_method = "Grid"
        else:
            parameters = self.param_grid

        # Calculating Size of Hyperparameter space
        hyperparam_space_size = 1
        for param in parameters.keys():
            hyperparam_space_size *= len(parameters[param])

        # Create Classifier w/ Gridsearch: 
        # (It will default to this is HP space is lower than the iteration limit)
        if search_method == "Grid" or hyperparam_space_size <= n_iter:
            if self.verbose: print(f"Grid search: {hyperparam_space_size}")
            self.classifier = GridSearchCV(
                estimator=self.estimator,
                param_grid=parameters,
                verbose=(1 if self.verbose else 0),
                cv=cv_folds,
                n_jobs = n_jobs
            )
        
        # Create Classifier w/ RandomizedSearchCV:
        elif search_method == "Random":
            if self.verbose: print(f"Random search: {hyperparam_space_size}")
            self.classifier = RandomizedSearchCV(
                estimator=self.estimator,
                param_distributions=parameters,
                verbose=(1 if self.verbose else 0),
                cv=cv_folds,
                n_iter=n_iter,
                n_jobs = n_jobs
            )

        else:
            raise ValueError(f"Train expected 'Random' or 'Grid' for search_method, recieved {search_method}")

        # Generate Test/Train Split:
        train_features, test_features, train_targets, test_targets = train_test_split(
            features, targets,
            test_size=self.test_size,
            stratify=targets
        )

        # Fit the classifier to training data:
        search = self.classifier.fit(self.tfidf.fit_transform(train_features), train_targets)
        
        # Assess Accuracy of classifier:
        test_predictions = self.classifier.predict(self.tfidf.transform(test_features))
        self.test_acc = accuracy_score(test_targets, test_predictions)
        self.test_f1 = f1_score(test_targets, test_predictions, average="micro")

        train_predictions = self.classifier.predict(self.tfidf.transform(train_features))
        self.train_acc = accuracy_score(train_targets, train_predictions)
        self.train_f1 = f1_score(train_targets, train_predictions, average="micro")

        self.ready = True
        self.best_params = self.classifier.best_params_

        # Report Test/Train scores:
        if self.verbose:
            training_time_str = str(time.time() - start_time)
            training_time_str = training_time_str[:training_time_str.find('.')+2]
            print(f"Completed Training in {training_time_str}s"),
            print("Results:")
            print(f"\tTest Accuracy: {self.test_acc}\n\tTest F1: {self.test_f1}")
            print(f"\tTrain Accuracy: {self.train_acc}\n\tTrain F1: {self.train_f1}")
            print(f"Best Parameters: {search.best_params_}")
            # print(f"Length of Trian Targets: {train_targets.shape}")

    def Predict(self, input:pd.Series|pd.DataFrame, column:str=None):
        # column: the column to predict on (input column title)
        
        # Validate input Types:
        if type(input) == pd.DataFrame and column == None or type(column) != str:
            raise AttributeError(f"'Predictions' expects a '{type(str())}' as 'column' when 'input' is {type(pd.DataFrame())}")
        
        # Make Predictions:
        if type(input) == pd.DataFrame:
            predictions = self.classifier.predict(self.tfidf.transform(input[column]))
        else:
            predictions = self.classifier.predict(self.tfidf.transform(input))

        return predictions
    
    def score(self) -> float:
        '''
        This method is to be used in both graphing and for the weight in voting.
        This returns a float between 0 and 1.
        '''
        return self.test_acc
    

class XGBoost_Model_Wrapper():
    def __init__(self, tfidf:TfidfVectorizer, paramater_grid:dict, test_size:float=0.2, objective:str='binary:logistic', nthread=4) -> None:
        self.estimator = XGBClassifier(
            objective=objective,
            nthread=nthread,
        )
        self.tfidf = tfidf
        self.param_grid = paramater_grid
        self.test_size=test_size

        self.label_encoder = LabelEncoder()
        self.ready = False
        self.best_params = None
        self.test_acc = 0 # used in comparison so should be set on __init__
    

    def __repr__(self) -> str:
        return "XBGoost Classifier"


    def Train(self, features: list | np.ndarray | pd.DataFrame, targets: list | np.ndarray | pd.DataFrame, **kwargs) -> None:
        '''
        Runs the training operations for this model.

        args:
            features:   (list | ndarray | DataFrame) The arraylike to use as input in training.
            targets:    (list | ndarray | DataFrame) The arraylike to use as targets for training.

        kwargs:
            use_best: (bool) False
            search_method: (str) 'Random'. could be 'Random' or 'Grid'
            n_iter: (int) 20
            cv_folds: (int) 2
            n_jobs: (int) -2

        Returns:
            None
        '''
                
        # Parsing keyword arguments:
        parse = lambda x, arg, defalt : x[arg] if arg in x else defalt # default value is 20
        #   use_best:
        use_best = 'use_best' in kwargs and kwargs['use_best']
        #   search_method:
        if 'search_method' in kwargs: search_method = kwargs['search_method']
        else: search_method = 'Random' # default value is 'Random'
        #   n_iter:
        n_iter = parse(kwargs, 'n_iter', 20)
        #   cv_folds:
        cv_folds = parse(kwargs, 'cv_folds', 2)
        #   n_jobs:
        n_jobs = parse(kwargs, 'n_jobs', -2)
        
        # Generating parameter grid to use for search:
        if use_best and self.best_params != None:
            parameters = {}
            for param in self.best_params.keys():
                parameters[param] = [self.best_params[param]]
            search_method = "Grid"
        else:
            parameters = self.param_grid
        
        # Calculating Size of Hyperparameter space
        hyperparam_space_size = 1
        for param in parameters.keys():
            hyperparam_space_size *= len(parameters[param])

        # Create Classifier w/ Gridsearch: 
        # (It will default to this is HP space is lower than the iteration limit)
        if search_method == "Grid" or hyperparam_space_size <= n_iter:
            # print(f"Grid Search of {hyperparam_space_size} hyperspace", end=' ',flush=True)
            self.classifier = GridSearchCV(
                estimator=self.estimator,
                param_grid=parameters,
                n_jobs = n_jobs,
                cv = cv_folds,
                verbose=False
            )

        # Create Classifier w/ RandomizedSearchCV:
        elif search_method == "Random":
            # print(f"Random Search of {hyperparam_space_size} hyperspace", end=' ',flush=True)
            self.classifier = RandomizedSearchCV(
                estimator=self.estimator,
                param_distributions=parameters,
                cv=cv_folds,
                n_jobs = n_jobs,
                n_iter=n_iter,
            )

        else:
            raise ValueError(f"Train expected 'Random' or 'Grid' for search_method, recieved {search_method}")

        # Generate Test/Train Split:
        train_features, test_features, train_targets, test_targets = train_test_split(
            features, targets,
            test_size=self.test_size,
            stratify=targets
        )

        # Generate Label Encoding for y:
        train_targets_encoded = self.label_encoder.fit_transform(train_targets)

        # Fit the classifier to training data:
        search = self.classifier.fit(self.tfidf.fit_transform(train_features), train_targets_encoded)
       
        # Assess Accuracy of classifier:
        test_predictions = self.classifier.predict(self.tfidf.transform(test_features))
        self.test_acc = accuracy_score(test_targets, self.label_encoder.inverse_transform(test_predictions))

        train_predictions = self.classifier.predict(self.tfidf.transform(train_features))
        self.train_acc = accuracy_score(train_targets, self.label_encoder.inverse_transform(train_predictions))

        self.ready = True
        self.best_params = self.classifier.best_params_


    def Predict(self, input:pd.Series|pd.DataFrame, column:str=None):
        # column: the column to predict on (input column title)
        
        # Validate input Types:
        if type(input) == pd.DataFrame and column == None or type(column) != str:
            raise AttributeError(f"'Predictions' expects a '{type(str())}' as 'column' when 'input' is {type(pd.DataFrame())}")
        
        # Make Predictions:
        if type(input) == pd.DataFrame:
            predictions = self.classifier.predict(self.tfidf.transform(input[column]))
        else:
            predictions = self.classifier.predict(self.tfidf.transform(input))

        return self.label_encoder.inverse_transform(predictions)


    def score(self) -> float:
        '''
        This method is to be used in both graphing and for the weight in voting.
        This returns a float between 0 and 1.
        '''
        return self.test_acc