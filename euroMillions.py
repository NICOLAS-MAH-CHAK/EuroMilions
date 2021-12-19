#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 14:14:44 2021

@author: eisti
"""
import numpy as np
import pandas as pd
import csv
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from datetime import date
file = "EuroMillions_numbers.csv"


classifier = RandomForestClassifier(n_estimators=100, random_state=0)
results = []



def tirages_gagnants(file):
    #lecture du fichier
    f=open(file,"r")
    lecteur = csv.reader(f,delimiter=";")
    
    # chargement des éléments dans une liste
    tirages = []
    for ligne in lecteur :
        tirages.append(ligne)
      
    # fermeture du lecteur de fichier
    f.close()
    
    # formatisation (on enlève l'entête et les colonnes inintéressantes)
    tirages = np.array(tirages)
    tirages = tirages[1:,[1,2,3,4,5,6,7]].astype('int')
    # On trie les nombres dans l'ordre en séparant balls et stars
    sorted_tirages = np.sort(tirages[:,[0,1,2,3,4]])
    sorted_tirages = np.append(sorted_tirages, np.sort(tirages[:,[5,6]]), axis = 1)

    tirages = np.append(tirages,[[1]]*len(sorted_tirages),axis = 1)

    return tirages
    

def simule_tirage():
    ballPossibleValuesList = list(range(1,51))
    starPossibleValuesList = list(range(1,13))
    
    random.shuffle(ballPossibleValuesList)
    random.shuffle(starPossibleValuesList)
            # Now the list is shuffled, we'll take the ball values from these positions
    lotteryDrawBalls = ballPossibleValuesList[:5]
            # Now the list is shuffled, we'll take the star values from these positions
    lotteryDrawStars = starPossibleValuesList[:2]
            # Sort ball and star values chosen in the draw for easier matching
    lotteryDrawBalls = sorted(lotteryDrawBalls)
    lotteryDrawStars = sorted(lotteryDrawStars)
    
    return lotteryDrawBalls + lotteryDrawStars + [0]


def remplissage_BDD(tirages):
    
    L=[]
    n = 4*len(tirages)
    for i in range(n):
        L.append(simule_tirage())
    
    new_tirages = np.append(tirages, L, axis = 0)
    
    return new_tirages

def formatingDataset():
    winner = tirages_gagnants("EuroMillions_numbers.csv")
    bdd = remplissage_BDD(winner)
    np.random.shuffle(bdd)
    df = pd.DataFrame(data=bdd, columns=["B1", "B2", "B3", "B4", "B5", "E1", "E2", "W", ])
    X = df.iloc[:, 0:7]
    y = df.iloc[:, 7]

    return X,y

def trainingRF(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict_proba(X_test)
    predictions_proba = predictions[:, 1]
    predictions = predictions[:,1]

    for i in range(len(predictions)):
        if (predictions[i] == 1):
            results.append(X_test.iloc[i].to_numpy())
        if (predictions[i] < 0.5):
            predictions[i] = 0
        else:
            predictions[i] = 1

    #accuracy_model = accuracy_score(y_test, predictions.astype(int))

    #return predictions_proba,predictions

def predictionRF(X):

    df = pd.DataFrame(data=X, columns=["B1", "B2", "B3", "B4", "B5", "E1", "E2" ])
    predictions = classifier.predict_proba(df)
    predictions = predictions[:, 1]


    print("Probabilité  de gagner = ",predictions[0])

    return  predictions

def best_to_play():
    print("La meilleur combinaison à jouer est :")
    return results[0]

def addBase(combinaison):

    today = date.today()
    d1 = today.strftime("%d/%m/%Y")


    with open('EuroMillions_numbers.csv', 'a', newline='',) as fichiercsv:
        writer = csv.writer(fichiercsv,delimiter=";")
        writer.writerow([d1,combinaison[0],combinaison[1],combinaison[2],combinaison[3],combinaison[4],combinaison[5],combinaison[6]])
        fichiercsv.close()


def trainingModel():
    X,y = formatingDataset()
    trainingRF(X,y)

def infoModel():
    nom_model = " Random Forest Classifier"
    param = " n_estimators=100 "

    return nom_model,param
if __name__ == '__main__':

    #addBase([1,2,25,12,11,5,1])
    trainingModel()
    #predictionRF([[1,2,25,12,11,5,1]])
    #print(best_to_play())



