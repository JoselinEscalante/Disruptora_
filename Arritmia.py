# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 18:22:01 2019

@author: 52469
"""


import pandas as pd
import numpy as np

#Lee el archivo y lo coloca tipo dataframe pero es un diccionarioi
data=pd.read_table("C:/Users/52469/Desktop/Data Science curso Disruptora/Proyecto 2/Arritmia con vacios.csv", sep=",")
#Identificando columnas con valores vacíos
data.columns[data.isna().any()].tolist()

#Haciendo binarias las variables con valores vacíos
#data=np.isnan(data.T)
#data=np.isnan(data.P)
#data=np.isnan(data.QRST)
#data=np.isnan(data.J)
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="mean")
imp.fit_transform(data.Frecuencia_cardiaca)
imp.fit_transform(data.T)
imp.fit_transform(data.P)
imp.fit_transform(data.QRST)
imp.fit_transform(data.J)



#Definir las etiquetas y las caracteristicas: labels and features
b=data.Tipo_de_arritmia_cardiaca
a=data.drop('Tipo_de_arritmia_cardiaca', axis=1)


#Split data para el modelo
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(a,b, test_size=0.3, random_state=42)
#print (len(X_train))
#print (len(X_test))



#CLASIFICADOR 1
from sklearn.neighbors import KNeighborsClassifier
vecinos = 3
knn = KNeighborsClassifier(n_neighbors=vecinos)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)


#CLASIFICADOR 2
from sklearn import svm
clf_svm = svm.SVC(gamma='auto', kernel='rbf')
clf_svm.fit(X_train, y_train)
predictions = clf_svm.predict(X_test)


#CLASIFICADOR 3
from sklearn.naive_bayes import GaussianNB
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)
predictions = naive_bayes.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
Accuracy_score= accuracy_score(y_test, predictions)
Precision_score=precision_score(y_test, predictions,  average='weighted')
Recall_score=recall_score(y_test, predictions,average='weighted')

confmatrix=confusion_matrix(y_test, predictions)
