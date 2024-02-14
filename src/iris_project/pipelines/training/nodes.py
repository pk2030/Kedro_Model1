"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.19.2
"""
"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.9
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import seaborn as sns


def train_model(x_train, y_train,params:dict):
    model = LogisticRegression(penalty=params["penality"],class_weight=params["class_weight"])
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Accuracy Of Model is: ",accuracy)
    print(report)

    labels = model.classes_
    cm = confusion_matrix(y_test, y_pred)
    #plt.figure()
    #plot_confusion_matrix(cm, figsize=(16, 12), hide_ticks=True, cmap=plt.cm.Blues)
    # plt.xticks(range(4), labels, fontsize=12)
    # plt.yticks(range(4), labels, fontsize=12)
    
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')

    return plt