from array import array
from asyncio.windows_events import NULL
import math
from tabnanny import check
import warnings
import numpy as np
import pandas as pd
from data.data import process_data
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# Data class that holds values for route creation
class coordData:
    def __init__(self, scatsNumber, streets, latitude, longitude, id):
        neighbours = []

        self.scatsNumber = scatsNumber
        self.street1 = streets.split(" ")[0]
        self.street2 = streets.split(" ")[len(streets.split(" "))-1]
        self.latitude = latitude
        self.longitude = longitude
        self.id = id

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ - setting up neighbours

def findNeighbours(dataList):
    for x in dataList:
        x.neighbours = getScatsOnSameRoad(dataList, x.street1, x.street2, x.id)

    for x in dataList:
        x.neighbours = getClosestNeighbours(x)

    return dataList

def getScatsOnSameRoad(dataList, street1, street2, id):
    scatsOnSameRoad = []
    for x in dataList:
        if (x.street1 == street1 or x.street2 == street2) and x.id != id:
            scatsOnSameRoad.append(x)
    return scatsOnSameRoad

def getClosestNeighbours(node):
    neighbours = []
    verticalNeighbours = []
    horizontalNeighbours = []

    for x in node.neighbours:
        result = getNeighbourDirection(node, x)
        if result:
            horizontalNeighbours.append(x)
        else:
            verticalNeighbours.append(x)

    neighbours = reduceHorizontalNeighbours(horizontalNeighbours, node.latitude)
    temp = reduceVerticalNeighbours(verticalNeighbours, node.longitude)
    for x in temp:
        neighbours.append(x)

    return neighbours

def getNeighbourDirection(node, nodeNeighbour):
    dif1 = nodeNeighbour.latitude - node.latitude
    dif2 = nodeNeighbour.longitude - node.longitude

    if abs(dif1) > abs(dif2):
        return True #horizontal
    else:
        return False #vertical

def reduceHorizontalNeighbours(arrayOfNeighbours, nodeLatitude):
    left = []
    right = []
    closestLeft = NULL
    closestRight = NULL
    for x in arrayOfNeighbours:
        if x.latitude < nodeLatitude:
            left.append(x)
        else:
            right.append(x)
    for x in left:
        if closestLeft == NULL or abs(nodeLatitude) - abs(x.latitude) < abs(nodeLatitude) - abs(closestLeft.latitude):
            closestLeft = x
    for x in right:
        if closestRight == NULL or abs(x.latitude) - abs(nodeLatitude) < abs(closestRight.latitude) - abs(nodeLatitude):
            closestRight = x
    return checkIfNullForArray(closestLeft, closestRight)
    
def reduceVerticalNeighbours(arrayOfNeighbours, nodeLongitutde):
    down = []
    up = []
    closestDown = NULL
    closestUp = NULL
    for x in arrayOfNeighbours:
        if x.longitude < nodeLongitutde:
            down.append(x)
        else:
            up.append(x)
    for x in down:
        if closestDown == NULL or abs(nodeLongitutde) - abs(x.longitude) < abs(nodeLongitutde) - abs(closestDown.longitude):
            closestDown = x
    for x in up:
        if closestUp == NULL or abs(x.longitude) - abs(nodeLongitutde) < abs(closestUp.longitude) - abs(nodeLongitutde):
            closestUp = x
    return checkIfNullForArray(closestDown, closestUp)

def checkIfNullForArray(value1, value2):
    array = []
    if value1 != NULL:
        array.append(value1)
    if value2 != NULL:
        array.append(value2)
    return array

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def main():
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ - Tim's code for setting up data for navigation
    # set up data for generating route
    data = pd.read_csv('data/BoroondaraData.csv', encoding='utf-8').fillna(0)
    scatsNumber = data['SCATS Number'].to_numpy()
    streets = data['Location']
    latitude = data['NB_LATITUDE'].to_numpy()
    longitude = data['NB_LONGITUDE'].to_numpy()

    #Create list of scats ids so dont need to read in all 4000+ rows
    dataList = []
    scatsList = []
    for x in scatsNumber:
        if not x in scatsList:
            scatsList.append(x)

    #loop through data to create list of objects containing relevant data
    i = 0
    j = 0
    while i < len(scatsNumber):
        if scatsNumber[i] in scatsList:
            dataList.append(coordData(scatsNumber[i], streets[i], round(latitude[i] + 0.00155, 5), round(longitude[i] + 0.00113, 5), j))
            scatsList.remove(scatsNumber[i])
            j+=1
        i+=1

    # print(idList)
    # print(len(idList))
    print(len(dataList))
    for x in dataList:
        print('-----------------------')
        print(x.scatsNumber)
        print(f'{x.street1} : {x.street2}')
        print(x.latitude)
        print(x.longitude)   
        print(f'id: {x.id}')

    dataList = findNeighbours(dataList)
    for x in dataList:
        print('--------------------------')
        if len(x.neighbours) == 0:
            print(f'{x.scatsNumber}: NULL')
        for y in x.neighbours:
            print(f'{x.scatsNumber}: {y.scatsNumber}')
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

if __name__ == '__main__':
    main()