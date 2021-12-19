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
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from datetime import date, datetime
file = "EuroMillions_numbers.csv"

nb_estimators = 100
classifier = RandomForestClassifier(n_estimators=nb_estimators, random_state=0)

results = []



def tirages_gagnants(file):
    """Formats in a list the winning combinations from a .


    Args:
      file: A string which gives the path of the csv with winning combinations.


    Returns:
      A list newly formatted with the values of combinations.


    Raises:
      FileExistsError: If 'file' does not indicate an available path for a csv.
    """
    #lecture du fichier
    try:
        f=open(file,"r")
    except:
        raise FileExistsError()
        
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
    """Simulate a lotto draw with tools using random.


    Args:
      None


    Returns:
      A list of numbers which contains 5 different numbers between 1 and 50, 2 different numbers between 1 and 12 and a 0 to indicate that the combination is not a part of the winning ones.


    Raises:
      None  
    """
    
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
    """Fills in the correct format a list of combinations containing the winners and losers simulated by simule_tirage().


    Args:
      tirages: the returned list of tirages_gagnants which contains a good format for the value of the winning combinations.


    Returns:
      An array containing the winning combinations and the losing ones, simulated with simule_tirage, with 1 (resp 0) at the end of the row if the combination is a winning (resp losing) one.


    Raises:
      None  
    """
    
    L=[]
    n = 4*len(tirages)
    for i in range(n):
        L.append(simule_tirage())
    
    new_tirages = np.append(tirages, L, axis = 0)
    
    return new_tirages


def formating_dataset():
    """Formats a dataset separating an array of combinations and the array corresponding to the results (1 for win, 0 for lose).


    Args:
      None


    Returns:
      Two arrays: the first one containing the combinations, the second one containing the results, matching with the combinations from the first array.


    Raises:
      None  
    """
    
    winner = tirages_gagnants(file)
    bdd = remplissage_BDD(winner)
    np.random.shuffle(bdd)
    df = pd.DataFrame(data=bdd, columns=["B1", "B2", "B3", "B4", "B5", "E1", "E2", "W", ])
    X = df.iloc[:, 0:7]
    y = df.iloc[:, 7]

    return X,y


def training_RF(X,y):
    """Trains a Random Forest model with the dataset returned by formating_dataset.


    Args:
      X,y: X is an array of combinations, y is an array of 0 and 1 corresponding to the win (1) or the lose (0) of the combinations from X .


    Returns:
      predictions_proba: the array of the probabilities of the data of test to be winning combinations according to the model.
      predictions: the array of 1 and 0 which are the results of the decision of predictions by the model.
      accuracy_model: a float indicating the accuracy of the model.
      X_test: the combination data-test used.
          

    Raises:
      None  
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict_proba(X_test)
    predictions_proba = deepcopy(predictions[:, 1])
    predictions = predictions[:,1]

    for i in range(len(predictions)):
        if (predictions[i] == 1):
            results.append(X_test.iloc[i].to_numpy())
        if (predictions[i] < 0.5):
            predictions[i] = 0
        else:
            predictions[i] = 1

    accuracy_model = accuracy_score(y_test, predictions.astype(int))

    return predictions_proba,predictions,accuracy_model,X_test


def prediction_RF(x):
    """Print the probability to win for a combination according with the model.


    Args:
      x: a list meaning a combination of a lotto draw.


    Returns:
      The probability to win with the combination in enter.
    

    Raises:
      None  
    """
    combination = [x]
    df = pd.DataFrame(data=combination, columns=["B1", "B2", "B3", "B4", "B5", "E1", "E2"])
    prediction = classifier.predict_proba(df)
    prediction = prediction[:, 1]

    print("Probabilité de gagner = ", prediction[0])

    return prediction


def best_to_play(predictions_proba, X_test):
    """Adds a combination in the file where the model tooks his data.


    Args:
        predictions_proba: An array with the probabilities to win about X_test combinations, predicted by the model.
        X_test: An array with the combinations of tests


    Returns:
      The list of the combination which has the most probability to win with.
    

    Raises:
      None  

    """
    max_proba = max(predictions_proba)
    probas = list(predictions_proba)
    index_max = probas.index(max_proba) # index_max_probability_to_win_in_data_tests
    best_combi_in_data_tests = X_test.iloc[index_max].to_numpy()
    print("La meilleure combinaison à jouer du data test est :",list(best_combi_in_data_tests))
    
    return best_combi_in_data_tests



def add_base(combinaison):
    """Adds a combination in the file where the model tooks his data.


    Args:
      combination: a list meaning a combination which won a lotto draw.


    Returns:
      None
    

    Raises:
      None  
    """

    today = date.today()
    d1 = today.strftime("%d/%m/%Y")


    with open(file, 'a', newline='',) as fichiercsv:
        writer = csv.writer(fichiercsv,delimiter=";")
        writer.writerow([d1,combinaison[0],combinaison[1],combinaison[2],combinaison[3],combinaison[4],combinaison[5],combinaison[6]])
        fichiercsv.close()


def training_model():
    """Formats a dataset from the file and with tools using random, and  trains a random forest with it.


    Args:
      None


    Returns:
      The returns of training_RF with a formatted dataset in entry.
    

    Raises:
      None  
    """
    X,y = formating_dataset()
    return training_RF(X,y)



def info_model(accuracy_model):
    """Indicates information about the model.


    Args:
      accuracy_model : the accuracy of the model


    Returns:
      A tuple with the name of the model, the number of estimators in parameters of the model and the accuracy of the model calculated on the test data.
    

    Raises:
      None  
    """
    nom_model = "Random Forest Classifier"
    param = "n_estimators = {}".format(nb_estimators)
    accuracy = "accuracy = {}".format(accuracy_model)

    return nom_model,param,accuracy



if __name__ == '__main__':
    d1 = datetime.now()
    #addBase([1,2,25,12,11,5,1])
    predictions_proba,predictions,accuracy_model,X_test = training_model()
    
    ####
    # Preuve de notre conscience que notre modele ne fonctionne pas convenablement
    # n = len(predictions_proba)
    
    # compteur_0 = 0
    # compteur_1 = 0
    # compteur_else = 0
    # for i in range(n):
    #     if predictions_proba[i] == 0:
    #         compteur_0 += 1
    #     elif predictions_proba[i] == 1:
    #         compteur_1 += 1
    #     else :
    #         compteur_else +=1
            
    # print("Sur",n,"données, il y a",compteur_0,"probas égales à 0,",compteur_1,"probas égales à 1, et",compteur_else,"différentes.")
    ####
    
    # print(prediction_RF([1,2,25,12,11,5,1]))
    
    #print("accuracy =",accuracy_model)
    d2=datetime.now()
    duree = d2-d1
    #print("Temps d'execution :",duree)


