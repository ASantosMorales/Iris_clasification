#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:48:16 2017

@author: a_santos

My proposal to solve the classification problem of
Iris classification using a support vector machine (SVM)
with Radial Basis Function ('rbf') kernel.

Before this script it is necessary to access to data_analysis
script because there is a pre-processing step.

"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

#Import .npy data
path = 'write the path here!!!'
data = np.load(path)

#%% Random splitting of data
# In this script I try with different splitings
data_train_1, data_test_1 = train_test_split(data, test_size=0.4)
data_train_2, data_test_2 = train_test_split(data, test_size=0.3)
data_train_3, data_test_3 = train_test_split(data, test_size=0.2)
data_train_4, data_test_4 = train_test_split(data, test_size=0.1)

# Data assigning to variables
X_train_1 = data_train_1[:, 0:4]
X_train_2 = data_train_2[:, 0:4]
X_train_3 = data_train_3[:, 0:4]
X_train_4 = data_train_4[:, 0:4]

X_test_1 = data_test_1[:, 0:4]
X_test_2 = data_test_2[:, 0:4]
X_test_3 = data_test_3[:, 0:4]
X_test_4 = data_test_4[:, 0:4]

Y_train_1 = data_train_1[:, 4]
Y_train_2 = data_train_2[:, 4]
Y_train_3 = data_train_3[:, 4]
Y_train_4 = data_train_4[:, 4]

Y_test_1 = data_test_1[:, 4]
Y_test_2 = data_test_2[:, 4]
Y_test_3 = data_test_3[:, 4]
Y_test_4 = data_test_4[:, 4]

# Variable lists creation
X_train_list = ['X_train_1', 'X_train_2', 'X_train_3', 'X_train_4']
Y_train_list = ['Y_train_1', 'Y_train_2', 'Y_train_3', 'Y_train_4']

X_test_list = ['X_test_1', 'X_test_2', 'X_test_3', 'X_test_4']
Y_test_list = ['Y_test_1', 'Y_test_2', 'Y_test_3', 'Y_test_4']

#%% Working with the training data
# Running the different splits

success_rate_train = np.zeros([4, 1])
mse_train = np.zeros([4, 1])
cnf_matrix_train = np.zeros([3, 3, 4])
elapsed_train = np.zeros([4, 1])
k = 0
for i in range(4):
    print('New running')
    X_train = eval(X_train_list[i])
    Y_train = eval(Y_train_list[i])
    SVM_rbf = SVC(C=1, kernel='rbf')
    t = time.time()
    SVM_rbf.fit(X_train, Y_train)
    print('Training ok!')
    Y_train_out = SVM_rbf.predict(X_train)
    elapsed_train[i] = time.time() - t
    cnf_matrix_train[:, :, i] = confusion_matrix(Y_train, Y_train_out)
    success_rate_train[i] = ((cnf_matrix_train[0, 0, i] + cnf_matrix_train[1, 1, i] + cnf_matrix_train[2, 2, i]) / \
            sum(sum(cnf_matrix_train[:, :, i])))
    mse_train[i] = (mean_squared_error(Y_train, Y_train_out))
    del(SVM_rbf)
    print('running = ', i+1, '  time = ', elapsed_train[i])
    k = k + 1
   
#************************ Plot matrices confusion *************************************************

class_names = ['YES', 'NO']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
#*****************************************************************************
plt.figure()
plt.subplot(221)
plot_confusion_matrix(cnf_matrix_train[:, :, 0].astype(int), classes=class_names,
                      title='Default payments / Train data / run 1')

plt.subplot(222)
plot_confusion_matrix(cnf_matrix_train[:, :, 1].astype(int), classes=class_names,
                      title='Default payments / Train data / run 2')

plt.subplot(223)
plot_confusion_matrix(cnf_matrix_train[:, :, 2].astype(int), classes=class_names,
                      title='Default payments / Train data / run 3')

plt.subplot(224)
plot_confusion_matrix(cnf_matrix_train[:, :, 3].astype(int), classes=class_names,
                      title='Default payments / Train data / run 4')

#%% Working with the testing data
success_rate_test = np.zeros([4, 1])
mse_test = np.zeros([4, 1])
cnf_matrix_test = np.zeros([3, 3, 4])
elapsed_test = np.zeros([4, 1])
k = 0
for i in range(4):
    print('New running')
    X_train = eval(X_train_list[i])
    Y_train = eval(Y_train_list[i])
    X_test = eval(X_test_list[i])
    Y_test = eval(Y_test_list[i])
    SVM_rbf = SVC(C=1, kernel='rbf')
    t = time.time()
    SVM_rbf.fit(X_train, Y_train)
    print('Training ok!')
    Y_test_out = SVM_rbf.predict(X_test)
    elapsed_test[i] = time.time() - t
    cnf_matrix_test[:, :, i] = confusion_matrix(Y_test, Y_test_out)
    success_rate_test[i] = ((cnf_matrix_test[0, 0, i] + cnf_matrix_test[1, 1, i] + cnf_matrix_test[2, 2, i]) / \
            sum(sum(cnf_matrix_test[:, :, i])))
    mse_test[i] = (mean_squared_error(Y_test, Y_test_out))
    del(SVM_rbf)
    print('running = ', i+1, '  time = ', elapsed_test[i])
    k = k + 1
    
#*************** Plot matrices confusion for the testing data *********************************
class_names = ['YES', 'NO']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
#*****************************************************************************
plt.figure()
plt.subplot(221)
plot_confusion_matrix(cnf_matrix_test[:, :, 0].astype(int), classes=class_names,
                      title='Default payments / Test data / run 1')

plt.subplot(222)
plot_confusion_matrix(cnf_matrix_test[:, :, 1].astype(int), classes=class_names,
                      title='Default payments / Test data / run 2')

plt.subplot(223)
plot_confusion_matrix(cnf_matrix_test[:, :, 2].astype(int), classes=class_names,
                      title='Default payments / Test data / run 3')

plt.subplot(224)
plot_confusion_matrix(cnf_matrix_test[:, :, 3].astype(int), classes=class_names,
                      title='Default payments / Test data / run 4')
