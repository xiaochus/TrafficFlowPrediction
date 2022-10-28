from array import array
from asyncio.windows_events import NULL
from codecs import unicode_escape_decode
import heapq
import math
import random
from tabnanny import check
from turtle import distance
from typing import overload
import warnings
import numpy as np
import pandas as pd
from data.data import process_data
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
warnings.filterwarnings("ignore")

# Data class that holds values for route creation
class coordData:
    def __init__(self, scatsNumber, streets, latitude, longitude, id):
        self.neighbours = []

        self.scatsNumber = scatsNumber
        self.street1 = streets.lower().split(" of ")[0][:-2]
        self.street2 = streets.lower().split(" of ")[len(streets.split(" of "))-1]
        self.latitude = latitude
        self.longitude = longitude
        self.id = id
        self.visited = False
        self.distance = sys.maxsize
        self.previous = None
    
    def setVisited(self, bool):
        self.visited = bool

    def getDistance(self):
        return self.distance

    def setDistance(self, dist):
        self.distance = dist

    def getWeight(self, node):
        for x in self.neighbours:
            if x == node:
                return calculateWeight(self, node)

    def setPrevious(self, prev):
        self.previous = prev

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ - setting up neighbours

def calculateWeight(current, neighbour):
    if current.street1 == neighbour.street1 or current.street1 == neighbour.street2 or current.street2 == neighbour.street1 or current.street2 == neighbour.street2:
        weight = distanceBetweenVectors(current, neighbour)
    else:
        vector = (current.latitude, neighbour.longitude)
        value1 = distanceBetweenVectorsOverload(vector[0], vector[1], current.latitude, current.longitude)
        value2 = distanceBetweenVectorsOverload(vector[0], vector[1], neighbour.latitude, neighbour.longitude)
        weight = value1+value2
    return weight


def findNeighbours(dataList):
    for x in dataList:
        x.neighbours = getScatsOnSameRoad(dataList.copy(), x.street1, x.street2, x.id)

    # for x in dataList:
    #     x.neighbours = getClosestNeighbours(x)

    nullNodes = []
    for x in dataList:        
        if len(x.neighbours) == 0:
            nullNodes.append(x)
            x.neighbours = neighboursForNull(dataList, x)

    for x in dataList:
        for y in nullNodes:
            if x in y.neighbours:
                x.neighbours.append(y)

    return dataList

def getScatsOnSameRoad(dataList, street1, street2, id):
    scatsOnSameRoad = []
    for x in dataList:
        if (x.street1 == street1 or x.street2 == street2) and x.id != id:
            scatsOnSameRoad.append(x)
    return scatsOnSameRoad

"""def getClosestNeighbours(node):
    neighbours = []
    verticalNeighbours = []
    horizontalNeighbours = []

    for x in node.neighbours:
        result = getNeighbourDirection(node, x)
        if result:
            verticalNeighbours.append(x)            
        else:
            horizontalNeighbours.append(x)

    neighbours = reduceHorizontalNeighbours(horizontalNeighbours, node.longitude)
    temp = reduceVerticalNeighbours(verticalNeighbours, node.latitude)
    for x in temp:
        neighbours.append(x)

    return neighbours

def getNeighbourDirection(node, nodeNeighbour):
    dif1 = nodeNeighbour.latitude - node.latitude
    dif2 = nodeNeighbour.longitude - node.longitude

    if abs(dif1) > abs(dif2):
        return True #vertical
    else:
        return False #horizontal

def reduceHorizontalNeighbours(arrayOfNeighbours, nodeLongitutde):
    left = []
    right = []
    closestLeft = NULL
    closestRight = NULL
    for x in arrayOfNeighbours:
        if x.longitude < nodeLongitutde:
            left.append(x)
        else:
            right.append(x)
    for x in left:
        if closestLeft == NULL or abs(nodeLongitutde - x.longitude) < abs(nodeLongitutde - closestLeft.longitude):
            closestLeft = x
    for x in right:
        if closestRight == NULL or abs(x.longitude - nodeLongitutde) < abs(closestRight.longitude - nodeLongitutde):
            closestRight = x
    return checkIfNullForArray(closestLeft, closestRight)
    
def reduceVerticalNeighbours(arrayOfNeighbours, nodeLatitude):
    down = []
    up = []
    closestDown = NULL
    closestUp = NULL
    for x in arrayOfNeighbours:
        if x.latitude < nodeLatitude:
            down.append(x)
        else:
            up.append(x)
    for x in down:
        if closestDown == NULL or abs(nodeLatitude - x.latitude) < abs(nodeLatitude - closestDown.latitude):
            closestDown = x
    for x in up:
        if closestUp == NULL or abs(x.latitude - nodeLatitude) < abs(closestUp.latitude - nodeLatitude):
            closestUp = x
    return checkIfNullForArray(closestDown, closestUp)

def checkIfNullForArray(value1, value2):
    array = []
    if value1 != NULL:
        array.append(value1)
    if value2 != NULL:
        array.append(value2)
    return array"""

def neighboursForNull(dataList, node):
    closestNeighbour = dataList[0]
    closestDistance = distanceBetweenVectors(node, closestNeighbour)
    for x in dataList:
        tempDistance = distanceBetweenVectors(node, x)
        if tempDistance < closestDistance and tempDistance != 0:
            closestNeighbour = x
            closestDistance = tempDistance

    secondClosestNeighbour = NULL
    secondClosestDistance = NULL
    for x in dataList:
        tempDistance = distanceBetweenVectors(node, x)
        if (tempDistance < secondClosestDistance or secondClosestDistance == NULL) and x != closestNeighbour and tempDistance != 0:
            secondClosestNeighbour = x
            secondClosestDistance = tempDistance
    
    return [closestNeighbour, secondClosestNeighbour]

def distanceBetweenVectors(node1, node2):
    distance = math.sqrt((node1.latitude - node2.latitude)**2 + (node1.longitude - node2.longitude)**2)
    return distance

def distanceBetweenVectorsOverload(v1lat, v1lon, v2lat, v2lon):
    distance = math.sqrt((v1lat - v2lat)**2 + (v1lon - v2lon)**2)
    return distance

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def shortest(target, path):
    if target.previous:
        print(f'{target.previous.street1}, {target.previous.street2}')
        path.append(target.previous)
        shortest(target.previous, path)
    return

def dijkstra(dataList, startNode):
    #set distance of start node to 0
    startNode.setDistance(0)

    # Put tuple pair into the priority queue
    unvisitedQueue = [(x.getDistance(), x.id, x) for x in dataList]
    heapq.heapify(unvisitedQueue)

    while len(unvisitedQueue):
        # Pops a vertex with the smallest distance 
        uv = heapq.heappop(unvisitedQueue)
        current = uv[2]
        current.setVisited(True)

        #for next in node's neighbours
        for next in current.neighbours:            
            #if visited, skip
            if next.visited:
                continue
            newDist = current.getDistance() + current.getWeight(next)
                            
            if newDist < next.getDistance():
                next.setDistance(newDist)
                next.setPrevious(current)
                    # print ('updated : current = %s next = %s new_dist = %s' \
                    #         %(current.scatsNumber, next.scatsNumber, next.getDistance()))
                    #else:
                    # print ('not updated : current = %s next = %s new_dist = %s' \
                    #         %(current.scatsNumber, next.scatsNumber, next.getDistance()))

        #rebuild heap
        #pop every item
        while len(unvisitedQueue):
            heapq.heappop(unvisitedQueue)
        #put all vertices not visited into the queue
        unvisitedQueue = [(v.getDistance(), v.id, v) for v in dataList if not v.visited]
        heapq.heapify(unvisitedQueue)   
    

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def prepareRoutes():
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
            dataList.append(coordData(scatsNumber[i], streets[i], round(latitude[i], 5), round(longitude[i], 5), j))
            scatsList.remove(scatsNumber[i])
            j+=1
        i+=1

    """ print(idList)
    print(len(idList))
    print(len(dataList))
    for x in dataList:
        print('-----------------------')
        print(x.scatsNumber)
        print(f'{x.street1} : {x.street2}')
        print(x.latitude)
        print(x.longitude)   
        print(f'id: {x.id}')"""

    dataList = findNeighbours(dataList.copy())
    """for x in dataList:
        print('--------------------------')
        if len(x.neighbours) == 0:
            print(f'{x.scatsNumber}: NULL')
        for y in x.neighbours:
            print(f'{x.scatsNumber}: {y.scatsNumber}')
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"""

    return dataList
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def prepareDatalist(dataList):
    for location in dataList:
        location.setVisited(False)
        location.setPrevious(None)
        location.setDistance(sys.maxsize)

    return dataList

def pathFind(dataList, start, end):
    startNode = dataList[start]
    endNode = dataList[end]

    dataList = prepareDatalist(dataList)

    dijkstra(dataList, startNode)
    #print(dataList[25].scatsNumber, dataList[15].scatsNumber)

    target = endNode
    path = [target]
    print(f'{path[0].street1}, {path[0].street2}')
    shortest(target, path)
    #print(('The shortest path : %s' %(path[::-1])))

    # for x in dataList:
    #     print(f'{x.id} : {x.scatsNumber}')
    return path

def main():
    dataList = prepareRoutes()
    pathFind(dataList)

if __name__ == '__main__':
    main()