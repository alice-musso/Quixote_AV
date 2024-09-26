import csv
import os
import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold

from features import HstackFeatureSet


class ForwardFeatureSelector:
    def __init__(self, dev_unstacked, test_unstacked, y_dev, y_test, groups_dev, groups_test, feature_sets):
        
        self.dev_unstacked = dev_unstacked #dict
        self.test_unstacked = test_unstacked #dict
        self.y_dev = y_dev
        self.y_test = y_test
        self.groups_dev = groups_dev
        self.groups_test = groups_test        
        
        self.feature_sets = feature_sets
        self.best_feature_set = [] # list of str
        self.best_score = (0.0, 1.0)
        self.hstacker = HstackFeatureSet()

    def forward_feature_sel(self):
        
        while len(self.best_feature_set) < len(self.feature_sets):
            feature_sets_scores = dict()
            best_new_score = (0.0, -1.0)
            best_new_feature = None
            
            for feature in self.feature_sets:
                if feature in self.best_feature_set:
                    continue
                
                candidate_sets = self.best_feature_set + [feature]
                X_dev = self.hstacker._hstack([self.dev_unstacked[fs] for fs in candidate_sets])
                X_test = self.hstacker._hstack([self.test_unstacked[fs] for fs in candidate_sets])

                self.clf = self.model_trainer(X_dev, self.y_dev, self.groups_dev)

                acc, posterior_proba, y_pred = self.get_model_scores(X_test)

                feature_sets_scores[tuple(candidate_sets)] = (acc, posterior_proba)

                print('Candidate set:', candidate_sets)
                print('Actual:', [label for i,(label, _) in enumerate(zip(self.y_test, self.groups_test)) if label==1])
                print('Predicted:', [pred for i, pred in enumerate(y_pred) if pred == 1])
                print('Posterior probability:', posterior_proba)
                print()
                
                
            best_new_feature, best_new_score = self.sort_scores(feature_sets_scores)
            best_new_score_is_greater = self.compare_scores(best_new_score, self.best_score)


            if best_new_score_is_greater:
                self.best_feature_set.append(best_new_feature)
                self.best_score = best_new_score
                print(f"Added feature: {best_new_feature}. New score: {best_new_score}")
            
            else:
                print('No better candidates found.')
                print('\nFinal feature set:',)
                for fset in self.best_feature_set:
                    print(fset)
                print('Best final score:', self.best_score)
                break

        
        return self

    def model_trainer(self, X_dev, y_dev, groups_dev, model=None):
        if not model:
            model = LogisticRegression(random_state=42, n_jobs=32)

        param_grid = {'C': np.logspace(-4,4,9)}
        cv = GroupKFold(n_splits=5)


        grid = GridSearchCV(model,
                            param_grid=param_grid,
                            cv=cv,
                            n_jobs=32,
                            scoring='accuracy',
                            verbose=True,
                            )    

        grid.fit(X_dev, y_dev, groups=groups_dev)

        print('Model fitted. Best params:', grid.best_params_)
        print()

        print('Model built. \n')
        return grid.best_estimator_
    
    def get_model_scores(self, X_test):
        y_pred = self.clf.predict(X_test)
        y_pred=[int(pred) for pred in y_pred]
        acc = accuracy_score(self.y_test, y_pred)
        probabilities = self.clf.predict_proba(X_test)
        posterior_proba = max(probabilities[0])
        return acc, posterior_proba, y_pred

        
    def sort_scores(self, scores_dict):
        neg_score = min(scores_dict.items(), key=lambda x: (x[1][0], x[1][1]))  # pred == 0, prob crescente
        pos_score = max(scores_dict.items(), key=lambda x: (x[1][0], x[1][1]))  # pred == 1, prob crescente

        if pos_score[1][0] == 1:
            best_score = pos_score
        else:
            best_score = neg_score

        best_new_feature = best_score[0][-1]  # last feature added
        best_new_score = best_score[1]

        print('Best feature set:', best_new_feature)
        print('Best score:', best_new_score)

        return best_new_feature, best_new_score

    
    def compare_scores(self, new_score, old_score):
        print('Old best score:', old_score)
        print('New best score:', new_score)
        if old_score[0] == 1 and new_score[0] == 0:
            return False
        
        elif new_score[0] == 1 and old_score[0] == 0:
            return True
        
        elif new_score[0] == 1 and old_score[0] == 1:
            if new_score[1] > old_score[1]:  # if posterior proba is greater
                return True
            
        elif new_score[0] == 0 and old_score[0] == 0:
            if new_score[1] < old_score[1]: # if posterior proba is lower (less confidence)
                return True
            
    def save_res(self, target_document, file_name='verifiers_res_forward_ablation.csv'):
        path= '/home/martinaleo/.ssh/Quaestio_AV/src/data/LOO_res'
        print(f'Saving results in {file_name}\n')

        accuracy, posterior_proba = self.best_score
        os.chdir(path)
        data = {
            'Target Document': target_document,
            'Accuracy':accuracy,
            'Proba': posterior_proba,
            'vectorizers': self.best_feature_set
        }
        with open(file_name, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            # Check if the file is empty, if so, write the header
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(data)
        print(f"Forward model selection res for document {target_document} saved in file '{file_name}'\n")
            

class BackwardFeatureSelector:
    def __init__(self, dev_unstacked, test_unstacked, y_dev, y_test, groups_dev, groups_test, feature_sets):
        
        self.dev_unstacked = dev_unstacked #dict
        self.test_unstacked = test_unstacked #dict
        self.y_dev = y_dev
        self.y_test = y_test
        self.groups_dev = groups_dev
        self.groups_test = groups_test        
        
        self.feature_sets = feature_sets
        self.best_feature_set = feature_sets # list of str
        self.best_score = (0.0, 1.0)
        self.hstacker = HstackFeatureSet()

    def backward_feature_sel(self):
        
        while len(self.best_feature_set) > 0:
            feature_sets_scores = dict()
            best_new_score = (0.0, -1.0)
            best_new_feature_set = None
            
            for feature in self.best_feature_set:
                candidate_sets = self.best_feature_set.copy()
                candidate_sets.remove(feature)

                X_dev = self.hstacker._hstack([self.dev_unstacked[fs] for fs in candidate_sets])
                X_test = self.hstacker._hstack([self.test_unstacked[fs] for fs in candidate_sets])

                self.clf = self.model_trainer(X_dev, self.y_dev, self.groups_dev)

                acc, posterior_proba, y_pred = self.get_model_scores(X_test)

                feature_sets_scores[tuple(candidate_sets)] = (acc, posterior_proba)

                print('Candidate set:', candidate_sets)
                print('Actual:', [label for i,(label, _) in enumerate(zip(self.y_test, self.groups_test)) if label==1])
                print('Predicted:', [pred for i, pred in enumerate(y_pred) if pred == 1])
                print('Posterior probability:', posterior_proba)
                print()
                
                
            best_new_feature_set, best_new_score = self.sort_scores(feature_sets_scores)
            best_new_score_is_greater = self.compare_scores(best_new_score, self.best_score)


            if best_new_score_is_greater or self.best_score[0]==0:
                for fset in self.best_feature_set:
                    if fset not in best_new_feature_set:
                        removed_feature = fset

                self.best_feature_set = list(best_new_feature_set)
                self.best_score = best_new_score
                print(f"Removed feature: {removed_feature}.\nNew score: {best_new_score}")
            
            else:
                print('No better candidates found.')
                print('\nFinal feature set:',)
                for fset in self.best_feature_set:
                    print(fset)
                print('\nBest final score:', self.best_score)
                break

        
        return self

    def model_trainer(self, X_dev, y_dev, groups_dev, model=None):
        if not model:
            model = LogisticRegression(random_state=42, n_jobs=32)

        param_grid = {'C': np.logspace(-4,4,9)}
        cv = GroupKFold(n_splits=5)


        grid = GridSearchCV(model,
                            param_grid=param_grid,
                            cv=cv,
                            n_jobs=32,
                            scoring='accuracy',
                            verbose=True,
                            )    

        grid.fit(X_dev, y_dev, groups=groups_dev)

        print('Model fitted. Best params:', grid.best_params_)
        print()

        print('Model built. \n')
        return grid.best_estimator_
    
    def get_model_scores(self, X_test):
        y_pred = self.clf.predict(X_test)
        y_pred=[int(pred) for pred in y_pred]
        acc = accuracy_score(self.y_test, y_pred)
        probabilities = self.clf.predict_proba(X_test)
        posterior_proba = max(probabilities[0])
        return acc, posterior_proba, y_pred

        
    def sort_scores(self, scores_dict):

        neg_score = min(scores_dict.items(), key=lambda x: (x[1][0], x[1][1]))  # pred == 0, prob crescente
        pos_score = max(scores_dict.items(), key=lambda x: (x[1][0], x[1][1]))  # pred == 1, prob crescente

        if pos_score[1][0] == 1:
            best_score = pos_score
        else:
            best_score = neg_score

        best_new_feature_set = best_score[0]  # whole feature set
        best_new_score = best_score[1]

        print('Best feature set:', best_new_feature_set)
        print('Best score:', best_new_score)

        return best_new_feature_set, best_new_score

    
    def compare_scores(self, new_score, old_score):
        print('Old best score:', old_score)
        print('New best score:', new_score)
        if old_score[0] == 1 and new_score[0] == 0:
            return False
        
        elif new_score[0] == 1 and old_score[0] == 0:
            return True
        
        elif new_score[0] == 1 and old_score[0] == 1:
            if new_score[1] >= old_score[1]:  # if posterior proba is greater or equal
                return True
            
        elif new_score[0] == 0 and old_score[0] == 0:
            if new_score[1] <= old_score[1]: # if posterior proba is lower (less confidence) or equal
                return True
            

    def save_res(self, target_document, file_name='verifiers_res_backward_ablation.csv'):
        path= '/home/martinaleo/.ssh/Quaestio_AV/src/data/LOO_res'
        print(f'Saving results in {file_name}\n')

        accuracy, posterior_proba = self.best_score
        os.chdir(path)
        data = {
            'Target Document': target_document,
            'Accuracy':accuracy,
            'Proba': posterior_proba,
            'vectorizers': self.best_feature_set
        }
        with open(file_name, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            # Check if the file is empty, if so, write the header
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(data)
        print(f"Backward model selection res for document {target_document} saved in file '{file_name}'\n")