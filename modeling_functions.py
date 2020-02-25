import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)


def create_confusion_matrix(ytest: np.array, ypred: np.array) -> (int, int, int, int):
    cm = confusion_matrix(ytest, ypred)
    tp = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[1, 1]
    return tp, fp, fn, tn

def get_precision_recall(tp: int, fp: int, fn: int, tn: int) -> float:
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall


# Logistic Regression wrapper function
def get_logistic_regression_results(xtrain: np.array, xtest: np.array, ytrain: np.array, \
                                    ytest: np.array) -> (float, float, float):
    lr = LogisticRegression(C=1000, max_iter=1000).fit(xtrain, ytrain)
    ypred = lr.predict(xtest)
    accuracy = lr.score(xtest, ytest)
    tp, fp, fn, tn = create_confusion_matrix(ytest, ypred)
    precision, recall = get_precision_recall(tp, fp, fn, tn)
    return ypred, (accuracy, precision, recall)


# Function for random forest hyperparameter tuning
def plot_random_forest_hyperparameters(xtrain: np.array, xtest: np.array, ytrain: np.array, \
                                       ytest: np.array, genre_type: str) -> None:
    # Find optimal number of trees
    numTrees = np.arange(50, 201, 30)
    accuracy_t = []
    for n in numTrees:
        a = 0
        for i in range(5):
            rf = RandomForestClassifier(n, oob_score=True, n_jobs=-1).fit(xtrain, ytrain)
            # ypred = rf.predict(xtest)
            a += rf.score(xtest, ytest) / 5
        accuracy_t.append(a)
    fig, ax = plt.subplots()
    ax.plot(numTrees, accuracy_t)
    ax.set_title("RF accuracy by number of trees ({})".format(genre_type))

    # Find optimal max depth
    maxDepth = np.arange(3, 13)
    accuracy_d = []
    for d in maxDepth:
        a = 0
        for i in range(5):
            rf = RandomForestClassifier(max_depth=d, oob_score=True, n_jobs=-1). \
                    fit(xtrain, ytrain)
            # ypred = rf.predict(xtest)
            a += rf.score(xtest, ytest) / 5
        accuracy_d.append(a)
    fig, ax = plt.subplots()
    ax.plot(maxDepth, accuracy_d)
    ax.set_title("RF accuracy by max depth ({})".format(genre_type))

    # Find optimal number of features
    maxFeatures = np.arange(5, 11)
    accuracy_f = []
    for f in maxFeatures:
        a = 0
        for i in range(5):
            rf = RandomForestClassifier(max_features=f, oob_score=True, n_jobs=-1). \
                    fit(xtrain, ytrain)
            # ypred = rf.predict(xtest)
            a += rf.score(xtest, ytest) / 5
        accuracy_f.append(a)
    fig, ax = plt.subplots()
    ax.plot(maxFeatures, accuracy_f)
    ax.set_title("RF accuracy by max features ({})".format(genre_type))

# Random Forest wrapper function
def get_random_forest_results(num_trees: int, max_depth: int, max_features: int, \
                              xtrain: np.array, xtest: np.array, ytrain: np.array, \
                              ytest: np.array) -> (float, float, float):
    rf = RandomForestClassifier(num_trees, max_features=max_features, oob_score=True,\
                                n_jobs=-1).fit(xtrain, ytrain)
    ypred = rf.predict(xtest)
    accuracy = rf.score(xtest, ytest)
    tp, fp, fn, tn = create_confusion_matrix(ytest, ypred)
    precision, recall = get_precision_recall(tp, fp, fn, tn)
    return ypred, (accuracy, precision, recall)


# Function for gradient boosting hyperparameter tuning
def plot_gradient_boosting_hyperparameters(xtrain: np.array, xtest: np.array, ytrain: np.array, \
                                           ytest: np.array, genre_type: str) -> None:
    # Find optimal learning rate
    learningRate = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
    accuracy_l = []
    for l in learningRate:
        gbr = GradientBoostingClassifier(learning_rate=l).fit(xtrain, ytrain)
        # ypred = gbr.predict(xtest)
        a = gbr.score(xtest, ytest)
        accuracy_l.append(a)
    fig, ax = plt.subplots()
    ax.plot(learningRate, accuracy_l)
    ax.set_title("GBR accuracy by learning rate ({})".format(genre_type))

    # Find optimal number of trees
    numTrees = np.arange(50, 201, 30)
    accuracy_t = []
    for n in numTrees:
        gbr = GradientBoostingClassifier(n_estimators=n).fit(xtrain, ytrain)
        # ypred = gbr.predict(xtest)
        a = gbr.score(xtest, ytest)
        accuracy_t.append(a)
    fig, ax = plt.subplots()
    ax.plot(numTrees, accuracy_t)
    ax.set_title("GBR accuracy by number of trees ({})".format(genre_type))

    # Find optimal subsample rate
    subsampleRate = np.arange(0.2, 1.1, 0.2)
    accuracy_s = []
    for s in subsampleRate:
        gbr = GradientBoostingClassifier(subsample=s).fit(xtrain, ytrain)
        # ypred = gbr.predict(xtest)
        a = gbr.score(xtest, ytest)
        accuracy_s.append(a)
    fig, ax = plt.subplots()
    ax.plot(subsampleRate, accuracy_s)
    ax.set_title("GBR accuracy by subsample rate ({})".format(genre_type))

    # Find optimal max depth
    maxDepth = np.arange(3, 13)
    accuracy_d = []
    for d in maxDepth:
        gbr = GradientBoostingClassifier(max_depth=d).fit(xtrain, ytrain)
        # ypred = gbr.predict(xtest)
        a = gbr.score(xtest, ytest)
        accuracy_d.append(a)
    fig, ax = plt.subplots()
    ax.plot(maxDepth, accuracy_d)
    ax.set_title("GBR accuracy by max depth ({})".format(genre_type))

# Gradient boosting wrapper function
def get_gradient_boosting_results(learning_rate: float, num_trees: int, subsample_rate: float, \
                                  max_features: int, xtrain: np.array, xtest: np.array, \
                                  ytrain: np.array, ytest: np.array) -> (float, float, float):
    gbr = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=num_trees, \
                                     subsample=subsample_rate, max_features=max_features).fit(xtrain, ytrain)
    ypred = gbr.predict(xtest)
    accuracy = gbr.score(xtest, ytest)
    tp, fp, fn, tn = create_confusion_matrix(ytest, ypred)
    precision, recall = get_precision_recall(tp, fp, fn, tn)
    return ypred, (accuracy, precision, recall)


# Function for plotting ROC curve
def plot_roc_curve(ytest: np.array, ypred: np.array, ax: plt.axes) -> None:
    tpr, fpr, thresholds = roc_curve(ytest, ypred)
    ax.plot([0, 1], [0, 1])
    #ax.plot()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
