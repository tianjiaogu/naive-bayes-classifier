#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 00:37:07 2020

@author: gutianjiao
"""
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import urllib
from urllib import request

import sklearn

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import CategoricalNB

from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def rand_jitter(arr):
    stdev = .035*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

accuracy = 0

def trial(url):

    headers = ['ID', 'Clump Thickness', 'Uniform Cell Size', 
                'Uniform Cell Shape', 'Marginal Adhesion',
                'Single Epithelial Cell Size', 'Bare Nuclei',
                'Bland Chromatin', 'Normal Nucleoli',
                'Mitoses', 'class']

    df = pd.read_csv(url, header=None)


    df.columns = headers

    df = df.drop(df.columns[0], axis=1)
    
   
    print(df)


    df_encode = df.apply(LabelEncoder().fit_transform)

    print(df_encode)

    training_set, testing_set = train_test_split(df_encode, test_size=0.5)

    nb = CategoricalNB()


    #fit X (attribute values) to Y (class values)
    nb.fit(training_set.drop('class', axis=1).values.tolist(), training_set['class'].tolist())

    y_predict = nb.predict(testing_set.drop('class', axis=1))


    print(f"Accuracy score is: {accuracy_score(testing_set['class'].tolist(), y_predict)}")

    global accuracy

    accuracy = accuracy +  accuracy_score(testing_set['class'].tolist(), y_predict)


    df.hist()
    import seaborn as sns

    
    ax1 = sns.catplot(x='class', y='Normal Nucleoli', data=df_encode, jitter=0.35)
    ax2 = sns.catplot(x='class', y='Uniform Cell Size', data=df_encode, jitter=0.35)
    ax3 = sns.catplot(x='class', y='Bare Nuclei', data=df_encode, jitter=0.35) 
    
    xplot = df_encode['Bland Chromatin'].values.tolist()
    yplot = df_encode['Uniform Cell Shape'].values.tolist()
    
    xplot = df_encode['Clump Thickness'].values.tolist()
    yplot = df_encode['Bland Chromatin'].values.tolist()

    plt.scatter(rand_jitter(xplot), rand_jitter(yplot), s=13, c=df_encode['class'].values.tolist())

    plt.show()






def showGraph(url):
    headers = ['ID', 'Clump Thickness', 'Uniform Cell Size', 
            'Uniform Cell Shape', 'Marginal Adhesion',
            'Single Epithelial Cell Size', 'Bare Nuclei',
            'Bland Chromatin', 'Normal Nucleoli',
            'Mitoses', 'class']

    df = pd.read_csv(url, header=None)


    df.columns = headers

    df = df.drop(df.columns[0], axis=1)

   # print(df)


    import seaborn as sns

 
    # Basic 2D density plot
    sns.set_style("white")

    #sns.plt.show()
    
    # Custom it with the same argument as 1D density plot
    sns.kdeplot(df['Clump Thickness'])
    

    #plt.show()


def main():
    url1 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
    
    showGraph(url1)

    
    total = 0
    global accuracy
    
    for i in range(1):
        trial(url1)
        total += 1
    
    print('\n\nAverage accuracy of {} trials is {}'.format(total, accuracy / total))
    
    
    
main()