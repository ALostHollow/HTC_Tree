import pandas as pd
import numpy as np
import time

# Sklearn Imports:
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

class SimpleModel():
    """
    This Simple Model supports a set of SKLearn models.

    This class should be replaced with classes supporting other models with these same key methods:
    1. Train
      - data: (pd.DataFrame) the data
      - 
    2. Compare:
      - 
      - 
    3. Predict:
      - 
      - 
    """

    # Class Methods Overloads:
    def __init__(self, estimator, tfidf:TfidfVectorizer, paramater_grid:dict, cv_folds:int, test_size:float=0.2, search:str="Random", verbose=False) -> None:
        self.verbose = verbose # if True show process info
        self.estimator = estimator
        self.tfidf = tfidf # the vectorizer used in both Train and Predict
        self.param_grid = paramater_grid
        self.cv_folds = cv_folds
        self.test_size = test_size # The test size used in Trian
        self.search_method = search
        self.ready = False
        self.best_params = None
        self.test_acc = 0 # used in comparison so should be set on __init__
    
    # Class Methods:
    def Train(self, features: list | np.ndarray | pd.DataFrame, targets: list | np.ndarray | pd.DataFrame, **kwargs)->None:#, use_best:bool=False, n_iter:int=20)->None:
        # features: the column of predictions is based on (input column title)
        # targets: the column of classes to predict (classes column title)

        # Runs the training operations that must be done per simple model
        
        # Parse kwargs:
        use_best = 'use_best' in kwargs and kwargs['use_best']
        result = lambda x : x['n_iter'] if 'n_iter' in x else 20 # default value is 20
        n_iter = result(kwargs)

        if self.verbose: 
            start_time = time.time()
            print()

        # Generating parameter grid to use for search:
        if use_best and self.best_params != None:
            parameters = {}
            for param in self.best_params.keys():
                parameters[param] = [self.best_params[param]]
            search_method = "Grid"

        else:
            parameters = self.param_grid
            search_method = "Random"

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
                cv=self.cv_folds,
            )
        
        # Create Classifier w/ RandomizedSearchCV:
        elif search_method == "Random":
            if self.verbose: print(f"Random search: {hyperparam_space_size}")
            self.classifier = RandomizedSearchCV(
                estimator=self.estimator,
                param_distributions=parameters,
                verbose=(1 if self.verbose else 0),
                cv=self.cv_folds,
                n_iter=n_iter,
            )

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
    
    def score(self)->float:
        '''
        This method is to be used in both graphing and weight in voting.
        This returns a float between 0 and 1.
        '''
        return self.test_acc