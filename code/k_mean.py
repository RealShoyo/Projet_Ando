# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

"""
Comment faire k-mean : 
*Faire un code qui récupère les données des colonnes souhaités dans le tableur
* Faire undico avec le nom du pkm qui renvoie une liste de ses attributs
* Faire une fonction position entre deux pkm 
*Faire la generation aléatoire de n centre de masses.
Coder la boucle de gestion de modification des centres de masses
garde chaque groupe en calculant la variation des groupes, 
coder les calculs de varations
on fait une boucle sur la boucle, on garde celui avec le moins des variation au sein de chaque cluster
choisis le bon.
On fait un boucle en lançant ça pour plein de k différent, on fait le graphique des k
Coder le truc pour savoir quel k est le meilleure avec le coude, garder 
"""






"le code est pas du tout opti, je pourrais probablement faire des np arrays et faire avec full opérationo vectoriels et passer en progra objets "
"et remplacer probablement chacun des for mais pas le temps et pas trop l'utilité de faire un truc giga opti"









import matplotlib as plt
import numpy as np
import pandas as pd 
import random






def recupereDuCsv(listeDrops):
    data = pd.read_csv("raw/final_dataset.csv", sep=",", header=0)
    data = data.drop(columns = listeDrops , errors = "ignore")
    return data

def creationDictionnaire(data):
    dico = data.set_index(data.columns[0]).apply(list, axis=1).to_dict()
    return dico


##Là en vrai je fais le rapport entre leur truc, puis pythagore avec les rapports ? car sinon dif d'echelle entre le poids et les stats
##Je pense que c'est le mieux
##Listetaille Plage est la diff entre la plus grande valeur de cette stat et la plus petite.
##Je divise par ListeTaille, j'obtiens tjr qqch entre -1 et 1
def DistanceEntre2Pkms(informationPkm1 , informationPkm2 , listeTaillePlageMaximums  ):
    tailleInfos = len(informationPkm1)
    distances = []
    for i in range (tailleInfos):
        distances.append((informationPkm1[i] - informationPkm2[i]) / listeTaillePlageMaximums[i])
    distance = 0

    for i in range (tailleInfos):
        distance += distances[i]**2 
    distance = np.sqrt(distance)
    return distance


def recupereListeTaillePlageMaximums(data):
    data_num = data.select_dtypes(include=["number"])
    max_vals = data_num.max()
    min_vals = data_num.min()
    diffs = (max_vals - min_vals).tolist()
    return diffs , min_vals , max_vals

def makeCentroidesListes(size , nombre , minlist , maxlist):
    listesCentroides = []
    for i in range (nombre):
        centroide = []
        for j in range (size):
            centroide.append(random.uniform(maxlist[j], minlist[j]))
        listesCentroides.append(centroide)

    return listesCentroides

        
def findCloserCentroide(point , centroides , diffs):
    listeDistances = []
    for e in centroides :
        listeDistances.append(DistanceEntre2Pkms(point, e , diffs))
    

    return (listeDistances.index(min(listeDistances))) ##renvoie l'indice du centroides avec la distance minimum
  

def repartitionPointsCentroides(listeClees , listesCentroides , dico , diffs) :
    nombreCentroide  = len(listesCentroides)

    dictionnaireAppartenanceCentroides = {}
    for i in range (nombreCentroide) :
        dictionnaireAppartenanceCentroides[i] = []

    for cle in listeClees.iloc[1:]: ##ca permet de faire un forEach sans prendre le nom de la colonne du dataframe qui n'étant pas une clée, fait crash
        dictionnaireAppartenanceCentroides[findCloserCentroide(dico[cle] , listesCentroides , diffs)].append(cle) ##Ajoute à la liste du centroide le plus proche du point sa clée dans le dictionnaire

    return dictionnaireAppartenanceCentroides


##Moche
def actualisationCentroides(dictionnaireAppartenance , nombreCentroides , listesCentroides , nombreStats , dicoStatsPokemon):
    newListesCentroides = [[] for _ in range(nombreCentroides)] 
    for i in range(nombreCentroides): ##je fais une matrice avec des listes qui contiennent les valeurs de tous les pts qui sont rattachés à ce centroides 
        nombrePoints = len(dictionnaireAppartenance[i])
        matriceStats = [[] for _ in range(nombreStats)] ##Crée une liste vide contenant 
        for indexPoint in range(nombrePoints):
            for indexStat in range(nombreStats):
                matriceStats[indexStat][indexPoint] = dicoStatsPokemon[indexPoint][indexStat]

        for indexStats in range(nombreStats):
            newListesCentroides[i][indexStats] = np.mean(matriceStats[indexStats])


    return (listesCentroides , (newListesCentroides == listesCentroides))


def main():
    nombreCentroides = 5 ##hardcode pour le moment
    data = recupereDuCsv(["Alternate Form Name" , "Legendary Type" , "Primary Egg Group" , "Secondary Egg Group","Tiers" ])
    diffs , minlist , maxlist = recupereListeTaillePlageMaximums(data)
    nombreStats = len(diffs)
    dicoStatsPokemon = creationDictionnaire(data)
    listesCentroides = makeCentroidesListes(nombreStats , nombreCentroides , minlist , maxlist)
    isCentroidsFixed = False
    dictionnaireAppartenance = {}
    compteurTour = 0
    while not (isCentroidsFixed):
        compteurTour += 1
        print("\ntour numero" + str(compteurTour))
        dictionnaireAppartenance = repartitionPointsCentroides(data["Pokemon"] ,listesCentroides  , dicoStatsPokemon , diffs)
        listesCentroides , isCentroidsFixed = actualisationCentroides(dictionnaireAppartenance , nombreCentroides , listesCentroides , nombreStats , dicoStatsPokemon)

    print("fin de programme")
    print(dictionnaireAppartenance)

    return (dictionnaireAppartenance)


main()

