import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
from collections import Counter
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                            GradientBoostingClassifier, GradientBoostingRegressor)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout


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
    return accuracy, precision, recall


# Function for random forest classifier hyperparameter tuning
def plot_random_forest_class_hyperparameters(xtrain: np.array, xtest: np.array, ytrain: \
                        np.array, ytest: np.array, genre_type: str) -> None:
    # Find optimal number of trees
    start = time.time()
    numTrees = np.arange(100, 201, 20)
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
    end = time.time()
    print("num trees time", end-start)

    # Find optimal max depth
    start = time.time()
    maxDepth = np.arange(3, 11)
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
    end = time.time()
    print("max depth time", end-start)

    # Find optimal number of features
    start = time.time()
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
    end = time.time()
    print("max features time", end-start)

# Random Forest classifier wrapper function
def get_random_forest_class_results(num_trees: int, max_depth: int, max_features: int, \
                        xtrain: np.array, xtest: np.array, ytrain: np.array, ytest: \
                        np.array) -> (float, float, float):
    rf = RandomForestClassifier(num_trees, max_features=max_features, oob_score=True, \
                                n_jobs=-1).fit(xtrain, ytrain)
    ypred = rf.predict(xtest)
    accuracy = rf.score(xtest, ytest)
    tp, fp, fn, tn = create_confusion_matrix(ytest, ypred)
    precision, recall = get_precision_recall(tp, fp, fn, tn)
    return accuracy, precision, recall


# Function for gradient boosting classifier hyperparameter tuning
def plot_gradient_boost_class_hyperparameters(xtrain: np.array, xtest: np.array, ytrain: \
                            np.array, ytest: np.array, genre_type: str) -> None:
    # Find optimal learning rate
    start = time.time()
    learningRate = [0.01, 0.025, 0.05, 0.1, 0.2]
    accuracy_l = []
    for l in learningRate:
        gbc = GradientBoostingClassifier(learning_rate=l, subsample=0.5).fit(xtrain, ytrain)
        # ypred = gbc.predict(xtest)
        a = gbc.score(xtest, ytest)
        accuracy_l.append(a)
    fig, ax = plt.subplots()
    ax.plot(learningRate, accuracy_l)
    ax.set_title("gbc accuracy by learning rate ({})".format(genre_type))
    end = time.time()
    print("learning rate time", end-start)

    # Find optimal number of trees
    start = time.time()
    numTrees = np.arange(100, 201, 20)
    accuracy_t = []
    for n in numTrees:
        gbc = GradientBoostingClassifier(n_estimators=n, subsample=0.5).fit(xtrain, ytrain)
        # ypred = gbc.predict(xtest)
        a = gbc.score(xtest, ytest)
        accuracy_t.append(a)
    fig, ax = plt.subplots()
    ax.plot(numTrees, accuracy_t)
    ax.set_title("gbc accuracy by number of trees ({})".format(genre_type))
    end = time.time()
    print("num trees time", end-start)

    # Find optimal max depth
    start = time.time()
    maxDepth = np.arange(3, 11)
    accuracy_d = []
    for d in maxDepth:
        gbc = GradientBoostingClassifier(subsample=0.5, max_depth=d).fit(xtrain, ytrain)
        # ypred = gbc.predict(xtest)
        a = gbc.score(xtest, ytest)
        accuracy_d.append(a)
    fig, ax = plt.subplots()
    ax.plot(maxDepth, accuracy_d)
    ax.set_title("gbc accuracy by max depth ({})".format(genre_type))
    end = time.time()
    print("max depth time", end-start)

# Gradient boosting classifier wrapper function
def get_gradient_boost_class_results(learning_rate: float, num_trees: int, max_depth: \
                        int, xtrain: np.array, xtest: np.array, ytrain: np.array, \
                        ytest: np.array) -> (float, float, float):
    gbc = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=num_trees, \
                                     subsample=0.5, max_depth=max_depth).fit(xtrain, ytrain)
    ypred = gbc.predict(xtest)
    accuracy = gbc.score(xtest, ytest)
    tp, fp, fn, tn = create_confusion_matrix(ytest, ypred)
    precision, recall = get_precision_recall(tp, fp, fn, tn)
    return accuracy, precision, recall


# Grid search on gradient boosting regressor 
def grid_search_gradient_boost(xtrain: np.array, xtest: np.array, ytrain: np.array, \
                    ytest: np.array) -> (dict, float):
    start = time.time()
    gbr = GradientBoostingRegressor()
    parameters = {"learning_rate": [0.01, 0.025, 0.05, 0.1, 0.2], "n_estimators": \
                  np.arange(100, 201, 20), "max_depth": np.arange(3, 11)}
    gridSearch = GridSearchCV(gbr, parameters, "neg_mean_squared_error", n_jobs=-1, \
                              cv=5, verbose=1)
    gridSearch.fit(xtrain, ytrain)
    end = time.time()
    print("grid search time", end-start)
    return gridSearch


# Function for gradient boosting regressor hyperparameter tuning
def plot_gradient_boost_reg_hyperparameters(xtrain: np.array, xtest: np.array, ytrain: \
                            np.array, ytest: np.array, genre_type: str) -> None:
    # Find optimal learning rate
    start = time.time()
    learningRate = [0.01, 0.025, 0.05, 0.1, 0.2]
    rmse_l = []
    for l in learningRate:
        gbc = GradientBoostingRegressor(learning_rate=l, subsample=0.5).fit(xtrain, ytrain)
        ypred = gbc.predict(xtest)
        r = np.sqrt(mean_squared_error(ytest, ypred))
        rmse_l.append(r)
    fig, ax = plt.subplots()
    ax.plot(learningRate, rmse_l)
    ax.set_title("gbr rmse by learning rate ({})".format(genre_type))
    end = time.time()
    print("learning rate time", end-start)

    # Find optimal number of trees
    start = time.time()
    numTrees = np.arange(100, 201, 20)
    rmse_t = []
    for n in numTrees:
        gbc = GradientBoostingRegressor(n_estimators=n, subsample=0.5).fit(xtrain, ytrain)
        ypred = gbc.predict(xtest)
        r = np.sqrt(mean_squared_error(ytest, ypred))
        rmse_t.append(r)
    fig, ax = plt.subplots()
    ax.plot(numTrees, rmse_t)
    ax.set_title("gbr rmse by number of trees ({})".format(genre_type))
    end = time.time()
    print("num trees time", end-start)

    # Find optimal max depth
    start = time.time()
    maxDepth = np.arange(3, 11)
    rmse_d = []
    for d in maxDepth:
        gbc = GradientBoostingRegressor(subsample=0.5, max_depth=d).fit(xtrain, ytrain)
        ypred = gbc.predict(xtest)
        r = np.sqrt(mean_squared_error(ytest, ypred))
        rmse_d.append(r)
    fig, ax = plt.subplots()
    ax.plot(maxDepth, rmse_d)
    ax.set_title("gbr rmse by max depth ({})".format(genre_type))
    end = time.time()
    print("max depth time", end-start)

# Gradient boosting regressor wrapper function
def get_gradient_boost_reg_results(learning_rate: float, num_trees: int, max_depth: int, \
                        xtrain: np.array, xtest: np.array, ytrain: np.array, ytest: \
                        np.array) -> ((float, float), np.array):
    gbc = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=num_trees, \
                                    subsample=0.5, max_depth=max_depth).fit(xtrain, ytrain)
    ypred = gbc.predict(xtest)
    score = gbc.score(xtest, ytest)
    rmse = np.sqrt(mean_squared_error(ytest, ypred))
    featureImportances = gbc.feature_importances_
    return (score, rmse), featureImportances    


# Multilayer perceptron wrapper function
def get_mlp_score(epochs: int, xtrain: np.array, xtest: np.array, ytrain: np.array, \
                ytest: np.array):
    mlp = Sequential()
    mlp.add(Dense(32, input_dim=xtrain.shape[1]))
    mlp.add(Activation("tanh"))
    mlp.add(Dropout(0.5))
    mlp.add(Dense(32))
    mlp.add(Activation("tanh"))
    mlp.add(Dropout(0.5))
    mlp.add(Dense(1))
    mlp.add(Activation("softmax"))
    mlp.compile(optimizer="adadelta", loss="mean_squared_error", metrics=["mse"])
    mlp.fit(xtrain, ytrain, epochs=epochs, verbose=1, validation_data=(xtest, ytest))
    score = mlp.evaluate(xtest, ytest)
    return score
