#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 00:44:27 2020

@author: gutianjiao
"""
import numpy as np
import pandas as pd

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


accuracy = 0
def trial(url):

    headers = ['color', 'size', 'act', 'age', 'class']
    #url = "https://archive.ics.uci.edu/ml/machine-learning-databases/balloons/adult+stretch.data"

    df = pd.read_csv(url, header=None)

    df.columns = headers
    #print(df.head)

    df = df.drop('color', axis=1)
    df = df.drop('size', axis=1)
    df_encode = df.apply(LabelEncoder().fit_transform)

    #print(df_encode)

    training_set, testing_set = train_test_split(df_encode, test_size=0.50)

    nb = CategoricalNB()
    
    #for dropping attributes 
    nb.fit(training_set.drop('class', axis=1).values.tolist(), training_set['class'].tolist())

    y_predict = nb.predict(testing_set.drop('class', axis=1))


    print(f"Accuracy score is: {accuracy_score(testing_set['class'].tolist(), y_predict)}")

    global accuracy

    accuracy = accuracy +  accuracy_score(testing_set['class'].tolist(), y_predict)


def main():
    url1 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/balloons/adult+stretch.data'
    url2 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/balloons/adult-stretch.data'
    url3 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/balloons/yellow-small+adult-stretch.data'
    url4 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/balloons/yellow-small.data'

    total = 0
    global accuracy
    
    for i in range(10):
        trial(url1)
        total += 1
    
    print('total is {}'.format(total))
    print('Average accuracy is {}'.format(accuracy/ total))
    
main()