from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

import pandas as pd
import random

def split_dataset(dataset_path: str, features: list, target: str) -> list:
    df = pd.read_csv(dataset_path)

    df_features = df[features]
    df_target = df[target]

    df_features_train, df_feature_test, df_target_train, df_target_test = train_test_split(df_features, df_target, test_size=0.25, random_state=random.random()*100)

    return [df_features, df_features_train, df_feature_test, df_target, df_target_train, df_target_test]

def get_metrics(target, predicted) -> dict:
    return {
        "accuracy" : accuracy_score(target, predicted),
        "f1-score" : f1_score(target, predicted),
        "recall" : recall_score(target, predicted),
        "precision" : precision_score(target, predicted)
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


def random_forest_classification(dataset_path: str, features: list, target: str) -> dict: 
    dfs = split_dataset(dataset_path, features, target)

    classifier = RandomForestClassifier(n_estimators=200, random_state=random.random()*100)

    classifier.fit(dfs[1], dfs[4])
    target_predicted = classifier.predict(dfs[2])

    return get_metrics(dfs[5, target_predicted])

def decision_tree_classification(dataset_path: str, features: list, target: str) -> dict:
    dfs = split_dataset(dataset_path, features, target)

    classifier = DecisionTreeClassifier(random_state=random.random()*100)

    classifier.fit(dfs[1], dfs[4])
    target_predicted = classifier.predict(dfs[2])

    return get_metrics(dfs[5, target_predicted])

def logistic_regression_classification(dataset_path: str, features: list, target: str) -> dict:
    dfs = split_dataset(dataset_path, features, target)

    classifier = LogisticRegression(max_iter=200, random_state=random.random()*100)

    classifier.fit(dfs[1], dfs[4])
    target_predicted = classifier.predict(dfs[2])

    return get_metrics(dfs[5, target_predicted])

def k_nearest_neighbors_classification(dataset_path: str, features: list, target: str) -> dict:
    dfs = split_dataset(dataset_path, features, target)

    classifier = KNeighborsClassifier(n_neighbors=5, algorithm='auto')

    classifier.fit(dfs[1], dfs[4])
    target_predicted = classifier.predict(dfs[2])

    return get_metrics(dfs[5, target_predicted])

def naive_bayes_classification(dataset_path: str, features: list, target: str) -> dict:
    dfs = split_dataset(dataset_path, features, target)

    classifier = GaussianNB()

    classifier.fit(dfs[1], dfs[4])
    target_predicted = classifier.predict(dfs[2])

    return get_metrics(dfs[5, target_predicted])
