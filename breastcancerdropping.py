#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 00:41:14 2020

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
    stdev = .05*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev




accuracy = 0

def trial(url, trials):

    headers = ['ID', 'Clump Thickness', 'Uniform Cell Size', 
                'Uniform Cell Shape', 'Marginal Adhesion',
                'Single Epithelial Cell Size', 'Bare Nuclei',
                'Bland Chromatin', 'Normal Nucleoli',
                'Mitoses', 'class']

    df = pd.read_csv(url, header=None)


    df.columns = headers

    df = df.drop(df.columns[0], axis=1)
    df = df.drop('Mitoses', axis=1)
    df = df.drop('Normal Nucleoli', axis=1)
    df = df.drop('Single Epithelial Cell Size', axis=1)
    df = df.drop('Uniform Cell Size', axis=1)
    df = df.drop('Marginal Adhesion', axis=1)
    #df = df.drop('Uniform Cell Shape' , axis=1)

    df = df.drop('Bare Nuclei', axis=1)

    #df = df.drop('Bland Chromatin', axis=1)
    df = df.drop('Clump Thickness', axis=1)
   # print(df)
    


    df_encode = df.apply(LabelEncoder().fit_transform)

    #print(df_encode)
    nb = CategoricalNB()


    

    #fit X (attribute values) to Y (class values)





    global accuracy
    total = 0
    for i in range(trials):

        total+=1
        training_set, testing_set = train_test_split(df_encode, test_size=0.80)
        nb.fit(training_set.drop('class', axis=1).values.tolist(), training_set['class'].tolist())

        y_predict = nb.predict(testing_set.drop('class', axis=1))
        current_acc = accuracy_score(testing_set['class'].tolist(), y_predict)

        print('The accuracy score for trial {} is : {}'.format(total, current_acc))

        accuracy = accuracy +  current_acc
        
    

    print('\n\nAverage accuracy of {} trials is {}'.format(total, accuracy / total))

    #plt.xlabel('Bland Chromatin')
    #plt.ylabel('Uniform Cell Shape')
    #plt.show()

    #print(xplot)
    #print(yplot)
   



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
    

    plt.show()


def main():
    url1 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
    
    #showGraph(url1)

    trial(url1, 10)
    

    
    
    
main()