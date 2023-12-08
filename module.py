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


# An Ensemble of the lower models
class Ensemble(Serializer, Deserializer):
    # Class Method Overloads:
    def __init__(self, models:list[type(any)]) -> None:
        # Validating 'models' parameter types:
        for model in models:
            self._validate_model(model)
        
        # Add Instance Attributes:
        self.models = models
        self.ready = False  # indicates if models can make predictions
        self.train_flag = True # indicates if models should to be trained
        self.created = time.time() # the time of instance instantiation
        self.single_class = None

    # Class Methods:
    def Train(self, data:pd.DataFrame, train_features:str, train_targets:str, verbose:bool=False, **kwargs) -> None:
        start_time = time.time()
        self.train_features = train_features # the column to base predictions on (can be overwritten in Predict)
        self.train_targets = train_targets # the column of classes to predict in training (can be overwritten in Predict)

        # Cleanining Training Data:
        clean_data = self._clean_Data(data=data.copy(deep=True))

        if len(clean_data[self.train_targets].unique()) == 1:
            # warnings.warn(f"'{train_targets}' given training data with a single class.\nAll predictions will be this single class: '{clean_data[self.train_targets].unique()[0]}'.")
            print(f"Warning: '{train_targets}' given training data with a single valid class. All predictions will be '{clean_data[self.train_targets].unique()[0]}'.")
            self.single_class = clean_data[self.train_targets].unique()[0]
            self.ready = True
            self.train_flag = False
            return


        if verbose: print(f"\nTraining Weak Learners for {train_targets}. {int(clean_data.shape[0]/data.shape[0]*100)}% of data is usable.", flush=True)

        # Getting max length of self.model[].estimator() as strings:
        if verbose: max_length = max([len(str(model.__repr__())) for model in self.models])

        # Training Each Weak Learner:
        for i in range(0, len(self.models)):
            training_start_time = time.time()
            if verbose: print(f"Training Learner {i + 1} of {len(self.models)}: {self.models[i].__repr__()}...", end=' ', flush=True) #/r - start of line
            
            # Copy so that we can compare if previously trained:
            temp_model = copy.deepcopy(self.models[i])
            
            # Train the temp model:
            temp_model.Train(clean_data[self.train_features].astype(str), clean_data[self.train_targets].astype(str), **kwargs)

            # Get amount of spaces required to line up text:
            if verbose: spaces=(max_length - len(str(self.models[i].__repr__()))) * ' '
            
            # Compare newly trained model:
            if temp_model.score() > self.models[i].score():
                if verbose: print(f"{spaces}Done ({time.time() - training_start_time}s)", flush=True)
                self.models[i] = temp_model
            else:
                if verbose: print(f"{spaces}New model was not better ({time.time() - training_start_time}s)", flush=True)

        # Mark self as Ready for predictions and as not needing training:
        self.ready = True
        self.train_flag = False
        self.single_class = None

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

    def Predict(self, input:pd.DataFrame, prediction_title:str=None, input_column:str=None, **kwargs) -> pd.DataFrame:
        if not self.ready:
            raise Exception("Ensemble is not ready for predictions.")
    
        # Parse kwargs:
        result = lambda x : x['weighted_voting'] if 'weighted_voting' in x else True # default value is True
        weighted_voting = result(kwargs)
        verbose = 'verbose' in kwargs and kwargs['verbose']
        
        if not input_column: input_column = self.train_features
        if not prediction_title: prediction_title = self.train_targets
        if verbose: print(f"Ensemble predicting for {prediction_title}")

        if self.single_class != None:
            output = input.copy(deep=True)
            output[prediction_title] = self.single_class
            output[prediction_title + " confidence"] = 1
            return output

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
  
    def _clean_Data(self, data:pd.DataFrame) -> pd.DataFrame:
        # Removing NaN items:
        # clean_data = data[data[self.train_targets].notna()]
        data[self.train_targets] = data[self.train_targets].fillna('NaN')

        # Removing non-duplicate data: (at least 2 items must be present in each class)
        data = data[data.duplicated(subset=[self.train_targets], keep=False)]

        # Returnining Cleaned Data:
        return data  
    
    def _weighted_vote(self, row, prediction_title:str) -> pd.Series:
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
    def _validate_model(model) -> None:
        try:
            callable(getattr(model, "Train"))
        except:
            raise AttributeError(f"{type(model)} has no method 'Train'\n  Use a wrapper for your models that includes a 'Train' method")
        
        try:
            callable(getattr(model, "Predict"))
        except:
            raise AttributeError(f"{type(model)} has no method 'Predict'\n  Use a wrapper for your models that includes a 'Predict' method")
    
    @staticmethod
    def _simple_vote(row) -> float:
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
    def Train(self, data: pd.DataFrame, force_retrain:bool=False, save_on_train:bool=True, verbose:bool=False, serializer:callable=None, **kwargs) -> None:
        self._set_train_flags(force_true=force_retrain, save_on_train=save_on_train, verbose=verbose, serializer=serializer)
        
        self._recursive_train(
            data=data, 
            save_on_train=save_on_train,
            verbose=verbose,
            serializer=serializer,
            **kwargs
        )

    def _recursive_train(self, data: pd.DataFrame, save_on_train:bool=False, verbose:bool=False, serializer:callable=None, **kwargs) -> None:
        # Training node ensemble:
        if self.ensemble.train_flag or self.ensemble.single_class != None:
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
                        verbose=verbose,
                        serializer=serializer,
                        **kwargs
                    )
        

    def Predict(self, input:pd.DataFrame, shallow=False, **kwargs) -> pd.DataFrame:
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
                    # print(f"no data to use for '{self.prediction_title}' at '{branch}'. {sub_node_data.shape[0]}/{node_predictions.shape[0]}")
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


    def gernate_design(self, **kwargs) -> dict:
        design = {
            "root": self._recursive_generate_design()
        }
        for arg in kwargs:
            design[arg] = kwargs[arg]

        return design

    def _recursive_generate_design(self) -> dict:
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


    def graph_weak_learner_performance(self):
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
        
        # Get a list of all weak learner titles:
        all_weak_learner_titles = []
        for i in range(0,len(data)):
            for learner in data[i][1].keys():
                all_weak_learner_titles.append(learner)
        all_weak_learner_titles = list(set(all_weak_learner_titles))
        
        # Add each Bar plot to graph space:
        for i in range(0,len(data)):
            node_title = data[i][0] 

            x_labels = [] # The order of weak learners will differ, so indexing them by a
            values = [] # single list of all weak learners is needed for the legend to be accurate
            for label in all_weak_learner_titles:
                if label in data[i][1].keys():
                    x_labels.append(label)
                    values.append(data[i][1][label])               
            
            # If this node doesn't have all weak learners: 
            # remove that label's colors from color list to plot
            colors_to_use = copy.deepcopy(colors)
            if len(x_labels) != len(all_weak_learner_titles):
                # This creates a list of labels not present in x_labels
                labels_to_remove = [x for x in all_weak_learner_titles if x not in x_labels]
                # This gets the indexes of these labels in the list of all weak learners
                index_to_remove = [all_weak_learner_titles.index(x) for x in labels_to_remove]
                # This removes those indexes from the list of colors to use in plotting bars
                for index in sorted(index_to_remove, reverse=True):
                    colors_to_use.pop(index)
                        

            # Get row and column in graph space for this plot:
            row = int(i/sqrt)
            col = int(i%sqrt)

            # Plot Bar:
            bars = axes[row, col].bar(
                x_labels,
                values,
                color=colors_to_use
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
            if 'Fixed Class' not in x_labels:
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
        legend_lines = [Line2D([0], [0], color=colors[all_weak_learner_titles.index(i)%len(colors)], lw=3, label=i) for i in all_weak_learner_titles]

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

        # Returning figure:
        return fig

    def _collect_data(self, tree_path:str=None) -> list:

        node_data = {}
        if self.ensemble.single_class:
            node_data['Fixed Class'] = 1.0
        else:
            for i in range(0, len(self.ensemble.models)):
                node_data[self.ensemble.models[i].__repr__()] = self.ensemble.models[i].score()

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
    

    def _set_train_flags(self, force_true:bool=False, verbose:bool=False, save_on_train:bool=True, serializer:callable=None) -> None:
        '''
        Recursively sets all the Ensemble()'s <train_flag> attribute.
        Training or not training an Esemble() is decided based on the ensemble's attribute <train_flag>.
        If True, the Ensemble() will be trained.

        This method could be hard to interpret, so I'll provide a few different ways of explaining.

        Truth Table: 
        F = <force_true>, E1 = an Ensemble(), E2 = another Ensemble()
         F   E1 E2  
        |F| |F| |F|  -> Set all <train_flag> to T
        |F| |F| |T|  -> Don't touch <train_flag>
        |F| |T| |F|  -> Don't touch <train_flag>
        |F| |T| |T|  -> Don't touch <train_flag>
        |T| |F| |F|  -> Set all <train_flag> to T
        |T| |F| |T|  -> Set all <train_flag> to T
        |T| |T| |F|  -> Set all <train_flag> to T
        |T| |T| |T|  -> Set all <train_flag> to T

        This table represents a tree with only two Ensembles(), but conditions would hold 
        regardless of adding more as checking these conditions is done by checking for the
        presence of a True value in the list of flag values.
        See 'conditions' below for summary explanation.
        
        This results in these conditions:
            If all Ensemble()s in the tree have <train_flag> False: Set all <train_flag>s to True
            If any Ensemble()s in the tree are set to True: Dont change any <train_flag>s
            If <force_true> is True: Set all <train_flag>s to True (This overrides the above two conditions).

        Effectively: 
            if True not in set(<list of trained flags>) or <force_true>:
                Set all <train_flag>s to True 

        The <force_true> set of conditions are sorted out by passing this argument to _recursive_set_train_flags.
        '''
        if True not in self._recursive_set_train_flags(force_true=force_true, verbose=verbose, save_on_train=save_on_train, serializer=serializer):
            if verbose: print(f"All models Trained, re-training all models.\n")
            if not force_true:
                self._recursive_set_train_flags(force_true=True, verbose=verbose, save_on_train=save_on_train, serializer=serializer)

    def _recursive_set_train_flags(self, force_true:bool=False, verbose:bool=False, save_on_train:bool=True, serializer:callable=None, **kwargs) -> set:
        # Sets own Ensemble() <train_flag> to True:
        if force_true: 
            self.ensemble.train_flag = True

        # Save Ensemble:
        if save_on_train:
            if serializer == None:
                self.ensemble.to_pickle(file_path=self.ensemble_path, overwite=True, verbose=verbose, **kwargs)
            else:
                serializer(self.ensemble, self.ensemble_path, **kwargs)

        # Get remaining Ensemble()s <train_flag>s
        flag_list = [self.ensemble.train_flag]
        for branch in self.branches.keys():
            for node in self.branches[branch]:
                for item in node._recursive_set_train_flags(force_true=force_true, verbose=verbose, save_on_train=save_on_train, serializer=serializer, **kwargs):
                    flag_list.append(item)
       
        # return <train_flag>s
        return flag_list


    def all_ensembles_ready(self) -> bool:
        '''
        Will return True if all Ensemble()s are ready for prediction, and False otherwise.
        '''
        return False not in self._recursive_get_ready_flags()

    def _recursive_get_ready_flags(self):
        # Get remaining Ensemble()s <train_flag>s
        flag_list = [self.ensemble.ready]
        for branch in self.branches.keys():
            for node in self.branches[branch]:
                for item in node._recursive_get_ready_flags():
                    flag_list.append(item)
       
        # return <train_flag>s
        return flag_list


    def update_from_json(self, file_path:str, verbose:bool=False) -> None:
        '''
        This allows updating of the design file without needing re-train the already present nodes.
        This does not handle new 'root' nodes, it will detect the entire structure as new.
        '''
        # Exceptions:
        if not os.path.exists(file_path):
            raise FileExistsError(f"File '{file_path}' does not exist")

        # Verbose Messages:
        if verbose:
                print(f"Building New Parts of Structure from '{file_path}'...", flush=True)
                start_time = time.time()

        temp_node = ClassificationNode.build_from_json(file_path)

        self._recursize_update(temp_node=temp_node, verbose=verbose)

        # Verbose Messages:
        if verbose: 
                print(f"Done. Operations took {time.time()-start_time}s", flush=True)

    def _recursize_update(self, temp_node:'ClassificationNode', verbose:bool=False) -> None:
        # This method could probably be done better

        for branch in list(self.branches.keys()):
            if branch not in temp_node.branches.keys():
                if verbose: print(f"Branch '{branch}' no longer present, removing")
                self.branches.pop(branch)

        for branch in temp_node.branches.keys():
            if verbose: print(f"\nBranch: '{branch}'", end=' ')

            if branch not in self.branches.keys():
                if verbose: print('new branch')
                self.branches[branch] = temp_node.branches[branch]

            else:
                if verbose: print('not new. Checking sub nodes:')
                current_node_titles = {}
                for node in self.branches[branch]:
                    current_node_titles[node.prediction_title] = self.branches[branch].index(node)

                temp_node_titles = {}
                for node in temp_node.branches[branch]:
                    temp_node_titles[node.prediction_title] = temp_node.branches[branch].index(node)

                # Adding new nodes in this branch, and recursive call for already present nodes:
                for title in temp_node_titles.keys():
                    if title not in current_node_titles.keys():
                        if verbose: print(f"new Sub node: {title}")
                        self.branches[branch].append(temp_node.branches[branch][temp_node_titles[title]])
                    else:
                        # Recursive Call:
                        if verbose: print(f"Not new Sub node: {title}")
                        self.branches[branch][current_node_titles[title]]._recursize_update(
                            temp_node.branches[branch][temp_node_titles[title]],
                            verbose=verbose
                        )

                # Removing nodes in this branch that are not present in new design:
                for node in self.branches[branch]:
                    if node.prediction_title not in temp_node_titles.keys():
                        if verbose: print(f"Sub node {title} no longer present. removing...")
                        self.branches[branch].pop(self.branches[branch].index(node))

    # Static Class Methods:
    @staticmethod
    def build_from_json(file_path:str, verbose:bool=False, save_ensembles:bool=False, serializer:callable=None) -> 'ClassificationNode':
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
    def _recursive_build(structure:dict, save_ensembles:bool=True, serializer:callable=None) -> 'ClassificationNode':
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


