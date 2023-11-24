

# Imports:
import pickle
import pandas as pd
import numpy as np
import time
import uuid
import os
import warnings
import json
import copy
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Sklearn Imports:
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score


# Handles Serialization and Deserialization:
class Serializer:
    def to_pickle(self, file_path:str, overwite:bool=True, verbose:bool=False) -> None:
        """
        Saves object as a pickle file at given location. 

        Args:
            self:       (Any) Object to serialize.
            file_path:  (str) File path to save pickle file to.
            overwite:    (bool) If 'True', overwrite any file present at 
                        <file_path>. If 'False', it will raise FileExistsError.
            verbose:    (bool) If 'True' print verbose output.
            **kwargs:   (any) Key word arguments to pass to pickle.dump()

        Returns:
            None
        """
        # Verbose Variables:
        start_time = time.time()

        # Exceptions:
        if os.path.exists(file_path) and not overwite:
            raise FileExistsError(f"Serializer Error: '{file_path}' already exists. Use 'overwrite' to replace this file.")
        
        # Verbose Messages:
        if os.path.exists(file_path) and verbose: 
            print(f"A file exists at '{file_path}'. Overwritting...", flush=True)
        
        # Serializing Object:
        with open(file_path, 'wb') as file:
            if verbose: print(f"Saving {self} at '{file_path}' as Pickle... ", end='', flush=True)
            pickle.dump(self, file)
            if verbose: print(f"Done. Operation took {time.time()-start_time}s")

        return
    
class Deserializer:
    @staticmethod
    def from_pickle(file_path:str, verbose:bool=False, **kwargs):
        """
        Loads object from <file_path>.

        Args:
            file_path:  (str) File path to load pickle file from.
            verbose:    (bool) If 'True' print verbose output.
            **kwargs:   (any) Key word arguments to pass to pickle.dump()

        Returns:
            Obejct at <file_path>.
        """

        # Exceptions:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' does not exist")
        
        # Verbose Messages:
        if verbose: 
                print(f"Loading object at '{file_path}'... ",end='', flush=True)
                start_time = time.time()

        # Opening File:
        with open(file_path, 'rb') as file:
            object =  pickle.load(file, **kwargs)
            if verbose: print(f"Done. Operations took {time.time()-start_time}s",flush=True)
            return object

    
# The lowest level object
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

    
# An Ensemble of the lower models
class Ensemble(Serializer, Deserializer):
    # Class Method Overloads:
    def __init__(self, models:list[SimpleModel|type(any)]) -> None:
        # Validating 'models' parameter types:
        for model in models:
            self._validate_model(model)
        
        # Add Instance Attributes:
        self.models = models
        self.ready = False  # indicates if models can make predictions
        self.train_flag = True # indicates if models should to be trained
        self.created = time.time() # the time of instance instantiation

    # Class Methods:
    def Train(self, data:pd.DataFrame, train_features:str, train_targets:str, verbose:bool=False, **kwargs)->None:
        start_time = time.time()
        self.train_features = train_features # the column to base predictions on (can be overwritten in Predict)
        self.train_targets = train_targets # the column of classes to predict in training (can be overwritten in Predict)

        # Cleanining Training Data:
        clean_data = self._clean_Data(data=data)
        if verbose: print(f"\nTraining Weak Learners for {train_targets}. {int(clean_data.shape[0]/data.shape[0]*100)}% of data is usable.", flush=True)
        
        # Getting max length of self.model[].estimator() as strings:
        if verbose: max_length = max([len(str(model.estimator)) for model in self.models])

        # Training Each Weak Learner:
        for i in range(0, len(self.models)):
            training_start_time = time.time()
            if verbose: print(f"Training Learner {i + 1} of {len(self.models)}: {self.models[i].estimator}...", end=' ', flush=True) #/r - start of line
            
            # Copy so that we can compare if previously trained:
            temp_model = copy.deepcopy(self.models[i])
            
            # Train the temp model:
            temp_model.Train(clean_data[self.train_features].astype(str), clean_data[self.train_targets].astype(str), **kwargs)

            # Get amount of spaces required to line up text:
            if verbose: spaces=(max_length - len(str(self.models[i].estimator))) * ' '
            
            # Compare newly trained model:
            if temp_model.score() > self.models[i].score():
                if verbose: print(f"{spaces}Done ({time.time() - training_start_time}s)", flush=True)
                self.models[i] = temp_model
            else:
                if verbose: print(f"{spaces}New model was not better ({time.time() - training_start_time}s)", flush=True)

        # Mark self as Ready for predictions and as not needing training:
        self.ready = True
        self.train_flag = False

        # Build Report String:
        if verbose:
            self.training_time = time.time() - start_time
            cursor_up_chars = "\033[A" * (len(self.models) + 1) # Move cursor up to where the print above the loop was:
            result_str1 = 'All Weak Learners Trained for'
            result_str2 = f"{int(clean_data.shape[0]/data.shape[0]*100)}% of data was used. "
            result_str3 = f"Operations took {self.training_time}s"

            # Report Result:
            print(cursor_up_chars + result_str1 + ' ' + train_targets + '. ' + result_str2 + result_str3,
                end="\n" * (len(self.models) + 1), # Move cursor down to below loop prints
                flush=True)

    def Predict(self, input:pd.DataFrame, prediction_title:str=None, input_column:str=None, **kwargs)->pd.DataFrame:
        
        # Parse kwargs:
        result = lambda x : x['weighted_voting'] if 'weighted_voting' in x else True # default value is True
        weighted_voting = result(kwargs)
        verbose = 'verbose' in kwargs and kwargs['verbose']
        
        # column: the column to based predictions on
        if not self.ready:
            raise Exception("Ensemble is not ready for predictions.")
        
        if not input_column: input_column = self.train_features
        if not prediction_title: prediction_title = self.train_targets
        if verbose: print(f"Ensemble predicting for {prediction_title}")

        predictions_df = pd.DataFrame(index=input.index)

        # Run predictions for each model in ensemble:
        for model in self.models:
            if verbose: print(f"model {self.models.index(model)} of {len(self.models)} predicting", end='\r', flush=True)
            predictions_df[str(self.models.index(model))] = model.Predict(input, input_column)

        # print(predictions_df.head())
        if verbose: print(f"All Learners made predictions", end='\n\n', flush=True)

        # Tally model votes:
        predictions_df.reindex(input.index)
        output = input.copy(deep=True)
        if weighted_voting:
            output[[prediction_title, prediction_title + " confidence"]] = predictions_df.apply(self._weighted_vote, axis=1, prediction_title=prediction_title)
        else:
            output[prediction_title] = predictions_df.mode(axis=1)[0]
            output[prediction_title + " confidence"] = predictions_df.apply(self._simple_vote, axis=1)

        return output
  
    def _clean_Data(self, data:pd.DataFrame)->pd.DataFrame:
        # Removing NaN items:
        clean_data = data[data[self.train_targets].notna()]
        # Removing non-duplicate data: (at least 2 items must be present in each class)
        clean_data = clean_data[clean_data.duplicated(subset=[self.train_targets], keep=False)]

        # Returnining Cleaned Data:
        return clean_data  
    
    def _weighted_vote(self, row, prediction_title:str)->pd.Series:
        '''
        Returns a pd.Series containing the precition and a 'confidence' score (0-1).

        The confidence score is the highest sum of unique prediction weights over total predictions.
        Effectively counting disagreeing weights as 0.0 then averaging all the weights.
        '''
        # Get sum weights for each unique prediction:
        item_counts = {}
        i = 0
        for model in self.models:           
            if row[i] not in item_counts.keys(): item_counts[row[i]] = model.score()
            else: item_counts[row[i]] += model.score()
            i += 1
            
        # Find the highest sum of weights:
        highest = ['', 0]
        for item in item_counts.keys():
            if item_counts[item] > highest[1]: highest = [item, item_counts[item]]

        # return highest count / number of total predictions:
        return pd.Series([highest[0], highest[1]/len(row)], index=[prediction_title, prediction_title + ' confidence'])
    
    # Class Static Methods:
    @staticmethod
    def _validate_model(model)->None:
        try:
            callable(getattr(model, "Train"))
        except:
            raise AttributeError(f"{type(model)} has no method 'Train'\n  Use a wrapper for your models that includes a 'Train' method")
        
        try:
            callable(getattr(model, "Predict"))
        except:
            raise AttributeError(f"{type(model)} has no method 'Predict'\n  Use a wrapper for your models that includes a 'Predict' method")
    
    @staticmethod
    def _simple_vote(row)->float:
        # count all predictions of each class predicted:
        item_counts = {}
        for item in row:
            if item not in item_counts.keys(): item_counts[item] = 1
            else: item_counts[item] += 1

        # find the highest count:
        highest = 0
        for item in item_counts.keys():
            if item_counts[item] > highest: highest = item_counts[item]

        # return highest count / number of total predictions:
        return highest/len(row)
    
    
        

# A node in the hierarchy:
class ClassificationNode(Serializer, Deserializer):
    # Class Method Overloads:
    def __init__(self, prediction_title:str, ensemble:Ensemble=None, ensemble_path:str=None, input_column:str='client description', node_id:str=None, branches:dict=None, save_ensembles:bool=True, serializer:callable=None):
        # Exceptions:
        if ensemble == None and ensemble_path == None:
            raise ValueError(f"ClassificationNode expects either an <ensemble> or and <ensemble_path>. Neither supplied.")
        
        # If ensemble not given, load ensemble from ensemble_path: TODO make callable...
        if ensemble == None:
            ensemble = Ensemble.from_pickle(ensemble_path)

        # If Id not given, create a random UUID:
        if node_id != None: self.node_id = node_id
        else: self.node_id = str(uuid.uuid4())

        # Object Attributes:
        self.prediction_title = prediction_title
        self.input_column = input_column
        self.ensemble=ensemble
        self.branches=branches
        self.ensemble_path = f"Ensembles/{prediction_title} ensemble-{self.node_id}.pkl"

        # Save ensemble at new location: TODO make callable...
        if ensemble_path != self.ensemble_path and save_ensembles:
            if serializer == None:
                self.ensemble.to_pickle(self.ensemble_path)
            else:
                serializer(self, self.ensemble_path)

    def __str__(self) -> str:
        part1 = f"Predicts '{self.prediction_title}'"
        part2 = 'has no branches' if self.branches == {} else f"has {len(self.branches.keys())} branches"

        part3 = 'Ensemble ready' if self.ensemble.ready == True else 'Ensemble not ready'
        part4 = f'contains {len(self.ensemble.models)} weak learners'

        return f"<ClassificationNode Object: {part1}, {part2}. {part3}, {part4}>"
    
    
    # Class Methods:
    def Train(self, data: pd.DataFrame, force_retrain:bool=False, save_on_train:bool=True, use_best:bool=False, verbose:bool=False, serializer:callable=None, **kwargs)->None:
        self._set_train_flags(force_true=force_retrain, save_on_train=save_on_train, verbose=verbose, serializer=serializer)
        
        self._recursive_train(
            data=data, 
            save_on_train=save_on_train,
            use_best=use_best,
            verbose=verbose,
            serializer=serializer,
            **kwargs
        )

    def _recursive_train(self, data: pd.DataFrame, save_on_train:bool=False, use_best:bool=False, verbose:bool=False, serializer:callable=None, **kwargs)->None:
        # Training node ensemble:
        if self.ensemble.train_flag:
            self.ensemble.Train(data, self.input_column, self.prediction_title, verbose=verbose, **kwargs)

            # Save Ensemble:
            if save_on_train:
                if serializer == None:
                    self.ensemble.to_pickle(file_path=self.ensemble_path, overwite=True, verbose=verbose)
                else:
                    serializer(self.ensemble, self.ensemble_path, **kwargs)

        elif verbose:
            print(f"'{self.ensemble.train_targets}' ensemble already trained.")

        # Check if this is a leaf node:
        if self.branches not in [None, {}]:
            # For each key classification:
            for branch in self.branches:
                # Split data for each branch:
                if branch != '*':
                    sub_data = data[data[self.prediction_title].astype(str) == branch]
                else:
                    sub_data = data

                # Check if sub_node_data is empty:
                if sub_data.shape[0] == 0:
                    warnings.warn(
                        f'''no data to use for '{self.prediction_title}' at '{branch}'. {sub_data.shape[0]}/{data.shape[0]} was usable.
                        Stopping tree iteration down this branch. Other nodes will be unaffected'''
                    )
                    continue

                # Run training for each branch:
                for node in self.branches[branch]:
                    node._recursive_train(
                        data=sub_data,
                        save_on_train=save_on_train,
                        use_best=use_best,
                        verbose=verbose,
                        serializer=serializer,
                        **kwargs
                    )
        

    def Predict(self, input:pd.DataFrame, shallow=False, **kwargs)->pd.DataFrame:
        # Node ensemble making predictions:
        node_predictions = self.ensemble.Predict(input, **kwargs)
        
        # Check if this is a leaf node:
        if self.branches not in [None, {}] and shallow == False:
            # For each key classification:
            for branch in self.branches.keys():
                # Split data for each branch:
                if branch != '*':
                    sub_node_data = node_predictions[node_predictions[self.prediction_title].astype(str) == branch]
                else:
                    sub_node_data = node_predictions
                
                # Check if sub_node_data is empty:
                if sub_node_data.shape[0] == 0:
                    print(f"no data to use for '{self.prediction_title}' at '{branch}'. {sub_node_data.shape[0]}/{node_predictions.shape[0]}")
                    continue

                # Run predictions for each branch: (New Columns)
                for node in self.branches[branch]:
                    # Recusrsive Call:
                    sub_node_predictions = node.Predict(input=sub_node_data, **kwargs)
                    # print(sub_node_predictions.head())
                    if branch != '*':
                        column_indexer = sub_node_predictions.columns
                        row_indexer = node_predictions[self.prediction_title] == branch
                        node_predictions.loc[row_indexer, column_indexer] = sub_node_predictions[column_indexer]
                        node_predictions.loc[row_indexer, column_indexer] = sub_node_predictions[column_indexer]
                    else:
                        node_predictions = sub_node_predictions                    

        return node_predictions


    def gernate_design(self)->dict:
        design = {
            "root": self._recursive_generate_design()
        }
        return design

    def _recursive_generate_design(self)->dict:
        map = {}

        for branch in self.branches.keys():
            map[branch] = []
            for node in self.branches[branch]:
                map[branch].append(node._recursive_generate_design())

        return {
            "ensemble path": self.ensemble_path,
            "input column": self.input_column,
            "prediction title": self.prediction_title,
            "node id": self.node_id,
            "branches": map
            }
    

    def update_from_json(self, file_path:str, verbose:bool=False)->None:
        # Exceptions:
        if not os.path.exists(file_path):
            raise FileExistsError(f"File '{file_path}' does not exist")
        
        # Verbose Messages:
        if verbose: 
                print(f"Building New Parts of Structure from '{file_path}'...", flush=True)
                start_time = time.time() 

        
        temp_node = ClassificationNode.build_from_json(file_path)
        
        self._recursize_update(
            temp_node=temp_node,
        )

        # Verbose Messages:
        if verbose: 
                print(f"Done. Operations took {time.time()-start_time}s", flush=True)

    def _recursize_update(self, temp_node:'ClassificationNode')->None:
        for branch in list(self.branches.keys()):
            if branch not in temp_node.branches.keys():
                print(f"Branch '{branch}' no longer present, removing")
                self.branches.pop(branch)

        for branch in temp_node.branches.keys():
            print(f"\nBranch: '{branch}'", end=' ')

            if branch not in self.branches.keys():
                print('new branch')
                self.branches[branch] = temp_node.branches[branch]

            else:
                print('not new. Checking sub nodes:')
                current_node_titles = {}
                for node in self.branches[branch]:
                    current_node_titles[node.prediction_title] = self.branches[branch].index(node)

                temp_node_titles = {}
                for node in temp_node.branches[branch]:
                    temp_node_titles[node.prediction_title] = temp_node.branches[branch].index(node)

                for title in temp_node_titles.keys():
                    if title not in current_node_titles.keys():
                        print(f"new Sub node: {title}")
                        self.branches[branch].append(temp_node.branches[branch][temp_node_titles[title]])
                    else:
                        # Recursive Call:
                        print(f"Not new Sub node: {title}")
                        self.branches[branch][current_node_titles[title]]._recursize_update(
                            temp_node.branches[branch][temp_node_titles[title]]
                        )
    

    def _collect_data(self, tree_path:str=None)->list:

        node_data = {}
        for i in range(0, len(self.ensemble.models)):
            node_data[str(self.ensemble.models[i].estimator).replace('()', '')] = self.ensemble.models[i].score()

        if tree_path == None:
            node_title = self.prediction_title
            
        else:
            node_title = tree_path + '\n' + self.prediction_title
        
        data_list = [[node_title, node_data]]
        
        for branch in self.branches.keys():
            for node in self.branches[branch]:
                for item in node._collect_data(tree_path=node_title+"='"+branch+"'"):
                    data_list.append(item)
            

        return data_list
    

    def graph_weak_learner_performance(self, file_path:str='Weak Learner Plot.png'):
        # Variables:
        colors = ['indianred', 'gold', 'teal', 'royalblue','salmon', 'darkorange', 'seagreen', 'darkorchid']
        
        # Get node data:
        data = self._collect_data()

        # Calculate number of rows and columns in graph space:
        sqrt = int(math.sqrt(len(data))) + 1
        if ((sqrt*sqrt) - sqrt) >= len(data): n_rows = sqrt - 1
        else: n_rows = sqrt
        n_rows = max(n_rows, 2) # minimum size of gride is 2x2

        # Create Graph
        fig, axes = plt.subplots(n_rows, sqrt, sharey='row')
        fig.suptitle('Weak Learner Test Accuracy Scores by Classification Node')
        fig.supylabel('Test Accuracy Score (0.0 - 1.0)')
        
        # Add each Bar plot to graph space:
        for i in range(0,len(data)):
            # Get values needed for plotting from data:
            x_labels=list(data[i][1].keys())
            values = list(data[i][1].values())
            node_title = data[i][0]
            
            # Get row and column in graph space for this plot:
            row = int(i/sqrt)
            col = int(i%sqrt)

            # Plot Bar:
            bars = axes[row, col].bar(
                x_labels,
                values,
                color=colors
            )
            
            # Set title of Bar Graph:
            axes[row, col].set_title(
                node_title + ' Weak Learners',
                fontsize=5
            )
            
            # Style Axes:
            axes[row, col].get_xaxis().set_ticks([])
            axes[row, col].get_yaxis().set_ticks([])
            pos = axes[row, col].get_position()
            axes[row, col].set_position([pos.x0, pos.y0, pos.width - 0.02, pos.height])

            # Annote Bars:
            for bar in bars:
                height = bar.get_height()
                axes[row, col].annotate(
                    '{:.3f}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, -5),
                    fontsize=4,
                    textcoords="offset points",
                    ha='center', 
                    va='bottom',
                )
        
        # Hide remaining subplots in grid:
        for i in range(len(data), n_rows*sqrt):
            axes.flat[i].set_visible(False)

        # Create List of elements for the legend:
        legend_lines = [Line2D([0], [0], color=colors[x_labels.index(i)%len(colors)], lw=3, label=i) for i in x_labels]

        # Create legend:
        fig.legend(
            handles= legend_lines,
            loc='upper left', 
            fancybox=False, 
            shadow=False, 
            framealpha=0.6,
            fontsize="5"
        )

        # Style and Size Figure:
        fig.tight_layout()
        fig.set_size_inches(10, 5)

        # Save Figure:
        fig.savefig(file_path, dpi=300)


    def _set_train_flags(self, force_true:bool=False, verbose:bool=False, save_on_train:bool=True, serializer:callable=None)->None:
        '''
        Recursively sets all the ensemble's <train_flag> attribute.
        See <_set_train_flags> method for explination on how/why/when this is done.
        TODO attribute and return descriptions
        '''
        # This may be hard to read so I'll try to explain:
        # Training or not training an ensebmle is decided based on the ensemble's attribute <train_flag>.
        # If True, the ensemble will be trained.

        # The method <_recursive_set_train_flags> recursively iterates over the tree and will set these flags to True if:
        #   - The <force_true> parameter is set
        #   - No ensembles in the tree have the attribute <traing_flag> set to True
        # Otherwise it doesn't change any of these attributes
        # These two cases are handle by the first call to the <_recursive_set_train_flags> method, and the second call, respectively.

        # This allows:
        # 1. A call that will always train every ensemble: if <force_true> is True
        # 2. Every call will train all ensembles that have not yet been trained: due to ensemble's <ready> attribute being False on __init__ and only being set after being trained.
        # 3. Every call will train all ensembles that were left untrained after a previous (1. call): due to the program being stopped before all models were trained.

        if False not in self._recursive_set_train_flags(force_true=force_true, verbose=False, save_on_train=save_on_train, serializer=serializer) and not force_true:
            self._recursive_set_train_flags(force_true=True, verbose=verbose, save_on_train=save_on_train, serializer=serializer)

    def _recursive_set_train_flags(self, force_true:bool=False, verbose:bool=False, save_on_train:bool=True, serializer:callable=None, **kwargs)->set:
        if force_true: 
            self.ensemble.train_flag = True

        # Save Ensemble:
        if save_on_train:
            if serializer == None:
                self.ensemble.to_pickle(file_path=self.ensemble_path, overwite=True, verbose=verbose, **kwargs)
            else:
                serializer(self.ensemble, self.ensemble_path, **kwargs)

        flag_list = [self.ensemble.train_flag]
        for branch in self.branches.keys():
            for node in self.branches[branch]:
                for item in node._recursive_set_train_flags(force_true=force_true, verbose=verbose, save_on_train=save_on_train, serializer=serializer, **kwargs):
                    flag_list.append(item)

        return set(flag_list)


    # Static Class Methods:
    @staticmethod
    def build_from_json(file_path:str, verbose:bool=False, save_ensembles:bool=False, serializer:callable=None)->'ClassificationNode':
        start_time = time.time() 
        
        # Exceptions:
        if not os.path.exists(file_path):
            raise FileExistsError(f"File '{file_path}' does not exist")
        
        # Opening JSON File:
        with open(file_path) as file:
            test_json = json.load(file)

        if "verbose" in test_json.keys():
            verbose = test_json["verbose"]
        
        if "save_ensembles" in test_json.keys():
            save_ensembles = test_json["save_ensembles"]
        
        # Verbose Messages:
        if verbose: 
                print(f"Building Tree Structure from '{file_path}'...",end='', flush=True)
                       

        # Building Tree Structure:
        root_node = ClassificationNode._recursive_build(
            structure=test_json["root"],
            save_ensembles=save_ensembles,
            serializer=serializer
        )

        # Verbose Messages:
        if verbose: 
                print(f"Done. Operations took {time.time()-start_time}s", flush=True)
        
        # Returning Root Node of Tree Structure:
        return root_node
    
    @staticmethod
    def _recursive_build(structure:dict, save_ensembles:bool=True, serializer:callable=None)->'ClassificationNode':
        # Building The Sub Nodes in Each Branch:
        branches = {}
        for branch in structure["branches"].keys():
            branches[branch] = []

            # Build Each Sub Node in a Branch:
            for sub_node_structure in structure["branches"][branch]:
                # Recursive Call to Build a Node:
                sub_node = ClassificationNode._recursive_build(
                    structure=sub_node_structure,
                    save_ensembles=save_ensembles,
                    serializer=serializer
                )
                
                # Adding Built Node to Branches:
                branches[branch].append(sub_node)
        
        if 'node id' not in structure.keys(): id = None
        else: id = structure['node id']

        # Building This Node:
        node = ClassificationNode(
            ensemble_path=structure['ensemble path'],
            prediction_title=structure['prediction title'],
            input_column=structure['input column'],
            node_id=id,
            branches=branches,
            save_ensembles=save_ensembles,
            serializer=serializer,
        )

        # Returning This Node:
        return node


