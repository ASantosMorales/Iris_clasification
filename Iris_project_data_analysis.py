#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:52:02 2017

@author: a_santos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_open = open('/home/a_santos/Documents/TEC de Monterrey/Semestre_3/Receonocimiento de patrones/Proyecto_Final_Iris/iris.data', 'r')
data_open = data_open.readlines()

data = []
data_open_1 = []
for i in range(len(data_open)-1):
    data_open_1.append(data_open[i].split(','))
for i in range(len(data_open)-1):
    data.append(np.asarray(data_open_1[i][0:4], dtype=np.float))
data = np.asarray(data)

outs = np.zeros([len(data), 1])
outs[50:100] = 1
outs[100:150] = 2    

data = np.hstack((data, outs))

del(outs)
del(i)
del(data_open)
del(data_open_1)
#%% Descripción de los datos originales. Máximos, mínimos, medias, etc
dataframe = pd.DataFrame(data[:, 0:4], columns=['Sepal_length_(cm)', 'Sepal_width_(cm)', 
                                        'Petal_length_(cm)', 'Petal_width_(cm)'])
analysis = dataframe.describe()

#%% Diagramas de caja usando librerías propias de Python con los datos originales
plt.figure()
plt.boxplot(data[:, 0:4], labels = ['X_1', 'X_2', 'X_3', 'X_4'])
plt.ylabel('centimeters')
plt.title('Characteristics')
plt.grid(True)

#%% Ploteo de histogramas de los datos originales
f, axes = plt.subplots(2, 2)
labels = ['Sepal_length_(cm)', 'Sepal_width_(cm)',
          'Petal_length_(cm)', 'Petal_width_(cm)']

k = 0
for i in range(2):
    for j in range(2):
        axes[i, j].hist(data[:, k], bins = 'auto')
        axes[i, j].set_title(labels[k])
        k = k + 1

#%% MATRIZ DE COVARIANZA
corr_matrix = np.corrcoef(data.T)

#%% Dispersión de las variables
f, axes = plt.subplots(5, 5)

for i in range(5):
    for j in range(5):
        axes[i, j].plot(data[:, i], data[:, j], 'k.')

#%%***************************************************************************
#**************************** Guardado de datos ******************************
#*****************************************************************************

np.save('data.npy', data)