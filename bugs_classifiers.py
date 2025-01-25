from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import cross_val_score

import pandas as pd
import random
import optuna

def split_dataset(dataset_path: str, target: str) -> list:
    df = pd.read_csv(dataset_path)

    all_features = df.columns.values.tolist()
    all_features.remove(target)
    df_features = df[all_features]
    df_target = df[target]

    df_features_train, df_feature_test, df_target_train, df_target_test = train_test_split(df_features, df_target, test_size=0.25, random_state=int(random.random()*100))

    return [df_features, df_features_train, df_feature_test, df_target, df_target_train, df_target_test]

def get_metrics(target, predicted) -> dict:
    return {
        "accuracy" : accuracy_score(target, predicted),
        "f1-score" : f1_score(target, predicted, average='weighted'),
        "recall" : recall_score(target, predicted, average='weighted'),
        "precision" : precision_score(target, predicted, average='weighted')
    }

def print_metrics(metrics: dict) -> None:
    len_key = max(map(lambda n: len(n), metrics.keys()))

    print(f'-'*(len_key+16))
    print(f'|{"█"*(len_key+4)}|  value  |')
    print(f'-'*(len_key+16))

    for k in metrics.keys():
        print(f'|  {k.rjust(len_key)}  |   {metrics[k]:.2f}  |')
        print(f'-'*(len_key+16))

def print_comparative_metrics(metrics: dict) -> None:
    methods_keys = metrics.keys()
    metrics_keys = metrics[list(methods_keys)[0]].keys()

    len_methods = max(map(lambda n : len(n), methods_keys))
    len_metrics = max(map(lambda n : len(n), metrics_keys))

    print(f'-'*(len_methods+((len_metrics+5)*len(metrics_keys))+6))
    print(f'|{"█"*(len_methods+4)}|', end='')
    for m in metrics_keys:
        print(f'  {m.rjust(len_metrics)}  |', end='')
    print()
    print(f'-'*(len_methods+((len_metrics+5)*len(metrics_keys))+6), end='')

    for m in methods_keys:
        print(f'\n|  {m.rjust(len_methods)}  |', end='')
        for me in metrics_keys:
            print(f'  {" ".rjust(len_metrics-4)}{metrics[m][me]:.2f}  |', end='')
        print()
        print(f'-'*(len_methods+((len_metrics+5)*len(metrics_keys))+6), end='')
    print()

def random_forest_classification(dfs) -> dict: 
    def objective(trial):
        
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)
        max_depth = trial.suggest_int('max_depth', 10, 100)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            min_samples_leaf=min_samples_leaf, 
            min_samples_split=min_samples_split, 
            max_features='log2',
            random_state=int(random.random()*100)
        )

        score = cross_val_score(model, dfs[0], dfs[3], n_jobs=-1, cv=3).mean()
        
        return score

    study = optuna.create_study(direction='maximize', study_name='Random_Forest')
    study.optimize(objective, n_trials=100)
    best_params = study.best_trial.params

    best_model = RandomForestClassifier(**best_params)
    best_model.fit(dfs[1], dfs[4])

    target_predicted = best_model.predict(dfs[2])

    return get_metrics(dfs[5], target_predicted)

def decision_tree_classification(dfs) -> dict:
    def objective(trial):
        
        max_depth = trial.suggest_int('max_depth', 10, 100)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)

        model = DecisionTreeClassifier(
            max_depth=max_depth, 
            min_samples_leaf=min_samples_leaf, 
            min_samples_split=min_samples_split, 
            max_features='log2',
            random_state=int(random.random()*100)
        )

        score = cross_val_score(model, dfs[0], dfs[3], n_jobs=-1, cv=3).mean()
        
        return score

    study = optuna.create_study(direction='maximize', study_name='Decision_Tree')
    study.optimize(objective, n_trials=100)
    best_params = study.best_trial.params

    best_model = DecisionTreeClassifier(**best_params)
    best_model.fit(dfs[1], dfs[4])

    target_predicted = best_model.predict(dfs[2])
    
    return get_metrics(dfs[5], target_predicted)

def logistic_regression_classification(dfs) -> dict:
    def objective(trial):
        
        max_iter = trial.suggest_int('max_iter', 50, 100)
        C = trial.suggest_float('C', 1, 2)

        model = LogisticRegression(
            max_iter=max_iter, 
            penalty='l2',
            solver='lbfgs',
            C=C,
            random_state=int(random.random()*100)
        )

        score = cross_val_score(model, dfs[0], dfs[3], n_jobs=-1, cv=3).mean()
        
        return score

    study = optuna.create_study(direction='maximize', study_name='Logistic_Regression')
    study.optimize(objective, n_trials=100)
    best_params = study.best_trial.params

    best_model = LogisticRegression(**best_params)
    best_model.fit(dfs[1], dfs[4])

    target_predicted = best_model.predict(dfs[2])

    return get_metrics(dfs[5], target_predicted)

def k_nearest_neighbors_classification(dfs) -> dict:
    def objective(trial):
        
        n_neighbors = trial.suggest_int('n_neighbors', 1, 20)

        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            algorithm='auto'
        )

        score = cross_val_score(model, dfs[0], dfs[3], n_jobs=-1, cv=3).mean()
        
        return score

    study = optuna.create_study(direction='maximize', study_name="K_Nearest_Neighbors")
    study.optimize(objective, n_trials=20)
    best_params = study.best_trial.params

    best_model = KNeighborsClassifier(**best_params)
    best_model.fit(dfs[1], dfs[4])

    target_predicted = best_model.predict(dfs[2])

    return get_metrics(dfs[5], target_predicted)

def naive_bayes_classification(dfs) -> dict:
    def objective(trial):
        

        var_smoothing = trial.suggest_float('var_smoothing', 0.000000001, 0.0000001)

        model = GaussianNB(
            var_smoothing=var_smoothing
        )

        score = cross_val_score(model, dfs[0], dfs[3], n_jobs=-1, cv=3).mean()
        
        return score

    study = optuna.create_study(direction='maximize', study_name='Naive_Bayes')
    study.optimize(objective, n_trials=100)
    best_params = study.best_trial.params

    best_model = GaussianNB(**best_params)
    best_model.fit(dfs[1], dfs[4])

    target_predicted = best_model.predict(dfs[2])

    return get_metrics(dfs[5], target_predicted)
