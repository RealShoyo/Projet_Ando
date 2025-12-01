# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

"""
Comment faire k-mean : 
Faire un code qui récupère les données des colonnes souhaités dans le tableur
Faire un objet pkm avec toutes les donées à analyser.
Faire une fonction position entre deux pkm 
Faire la generation aléatoire de n centre de masses.
Coder la boucle de gestion de modification des centres de masses
garde chaque groupe en calculant la variation des groupes, 
coder les calculs de varations
on fait une boucle sur la boucle, on garde celui avec le moins des variation au sein de chaque cluster
choisis le bon.
On fait un boucle en lançant ça pour plein de k différent, on fait le graphique des k
Coder le truc pour savoir quel k est le meilleure avec le coude, garder 
"""

import matplotlib as plt
import numpy as np
import pandas as pd 


data = pd.read_csv("dataset.csv", sep=",", header=0)
print(data)
print(data.columns)

data = data.drop(['Primary Type' , "Secondary Type" , "Female Ratio" , "Game(s) of Origin" , "Experience Growth" , "Primary Egg Group" , "Secondary Egg Group"])

print(data)