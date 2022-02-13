# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 10:46:00 2022

@author: Ravi Mistry
"""

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset
# Sourced from UCI Machine Learning repository
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# for dimensions of dataset 
# shape
print(dataset.shape)

# first 20 rows of data (peek at data)
# head
print(dataset.head(20))

# Summary of each attribute
# count, mean, min, max, percentiles
# descriptions
print(dataset.describe())