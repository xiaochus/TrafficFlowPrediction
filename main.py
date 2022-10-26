"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
from asyncio.windows_events import NULL
import math
from multiprocessing.spawn import prepare
from tabnanny import check
from typing import Literal
import warnings
import numpy as np
import pandas as pd
from data.data import process_data
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
warnings.filterwarnings("ignore")


lstm = None
gru = None
saes = None
my_model = None
y_scaler = None
lat_scaler = None
long_scaler = None
flask_app = Flask(__name__)
dataList = []

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

def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)


def plot_results(y_true, y_preds, names):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = '2006-3-4 00:00'
    x = pd.date_range(d, periods=96, freq='15min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()


def plot_results_no_true(y_preds, names):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = '2006-3-4 00:00'
    x = pd.date_range(d, periods=96, freq='15min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()

def createTestDay():
    latlong = [0.00165656, 0.99987643]
    x_test = []

    for i in range(1440):
        x_test.append(np.append([i * 15 / 1440], latlong))
    
    return np.array(x_test)

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

# Time is expected in minutes past 12am
def getTrafficData(locationData: coordData, time: float, model: Literal['lstm', 'gru', 'saes', 'my_model']):
    print(locationData.latitude)
    print(locationData.longitude)

    correctedTime = time / 1440
    preparedLatitude = lat_scaler.transform(np.array(locationData.latitude).reshape(1, -1))[0][0]
    preparedLongitude = long_scaler.transform(np.array(locationData.longitude).reshape(1, -1))[0][0]
    print('====================')
    print(preparedLatitude)
    print(preparedLongitude)
    X_test = np.array([[correctedTime, preparedLatitude, preparedLongitude]])
    if model == 'SAEs' or model == 'My_model':
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    else:
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    selectedModel = NULL
    if model == 'lstm':
        selectedModel = lstm
    elif model == 'gru':
        selectedModel = gru
    elif model == 'saes':
        selectedModel= saes
    elif model == 'my_model':
        selectedModel = my_model

    predicted = selectedModel.predict(X_test)

    predicted_new = np.zeros(shape=(len(predicted), 96))
    predicted_new[:,0] = predicted.reshape(-1, 1)[:, 0]
    predicted = y_scaler.inverse_transform(predicted_new)[:, 0].reshape(1, -1)[0][0]
    
    return predicted

def prepareLocations():
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ - Tim's code for setting up data for navigation
    # set up data for generating route
    data = pd.read_csv('data/BoroondaraData.csv', encoding='utf-8').fillna(0)
    scatsNumber = data['SCATS Number'].to_numpy()
    streets = data['Location']
    latitude = data['NB_LATITUDE'].to_numpy()
    longitude = data['NB_LONGITUDE'].to_numpy()

    #Create list of scats ids so dont need to read in all 4000+ rows
    global dataList
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
    # print(len(dataList))
    # for x in dataList:
    #     print('-----------------------')
    #     print(x.scatsNumber)
    #     print(f'{x.street1} : {x.street2}')
    #     print(x.latitude)
    #     print(x.longitude)   
    #     print(f'id: {x.id}')
    dataList = findNeighbours(dataList)
    # for x in dataList:
    #     print('--------------------------')
    #     if len(x.neighbours) == 0:
    #         print(f'{x.scatsNumber}: NULL')
    #     for y in x.neighbours:
    #         print(f'{x.scatsNumber}: {y.scatsNumber}')
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



def main():
    lstm = load_model('model/lstm.h5')
    # gru = load_model('model/gru.h5')
    # saes = load_model('model/saes.h5')
    # my_model = load_model('model/my_model.h5')
    # models = [lstm, gru, saes, my_model]
    # names = ['LSTM', 'GRU', 'SAEs', 'My_model']
    models = [lstm]
    names = ['LSTM']

    lag = 4
    file1 = 'data/BoroondaraData.csv'
    file2 = 'data/test.csv'
    _, _, X_test, y_test, scaler = process_data(file1, file2, lag)
    # print(y_test)
    # print(y_test.shape)
    # y_test = np.pad(y_test.reshape(-1, 1), (0, 95), 'empty')[:y_test.shape[0], :]
    # print(y_test)
    # print(y_test.shape)
    y_test_new = np.zeros(shape=(len(y_test), 96))
    y_test_new[:,0] = y_test.reshape(-1, 1)[:,0]
    y_test = scaler.inverse_transform(y_test_new).reshape(1, -1)[0]

    X_test = createTestDay()

    y_preds = []

    for name, model in zip(names, models):
        if name == 'SAEs' or name == 'My_model':
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        else:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        file = 'images/' + name + '.png'
        plot_model(model, to_file=file, show_shapes=True)

        predicted = model.predict(X_test)

        # print(predicted)
        predicted_new = np.zeros(shape=(len(predicted), 96))
        predicted_new[:,0] = predicted.reshape(-1, 1)[:, 0]
        predicted = scaler.inverse_transform(predicted_new)[:, 0].reshape(1, -1)[0]

        y_preds.append(predicted[:96])
        # print(name)
        # eva_regress(y_test, predicted)

    # print(y_test[: 96])
    # print(y_preds)
    plot_results_no_true(y_preds, names)
    # plot_results(y_test[: 96], y_preds, names)

def initialiseModels():
    global lstm
    global gru
    global saes
    global my_model
    global y_scaler
    global lat_scaler
    global long_scaler

    print ("Loading models")
    lstm = load_model('model/lstm.h5')
    gru = load_model('model/gru.h5')
    # saes = load_model('model/saes.h5')
    my_model = load_model('model/my_model.h5')
    print("Models loaded! Processing data")

    file1 = 'data/BoroondaraData.csv'
    file2 = ''
    _, _, _, _, y_scaler, lat_scaler, long_scaler = process_data(file1, '', 0)

    print("Data processed!")


    # location = dataList[0]
    # time = 10 * 60

    # print(f"Testing traffic at location: {location.street1} : {location.street2} | {location.latitude}, {location.longitude} at {time} mins")

    # print(getTrafficData(location, time, 'LSTM'))
    # print(getTrafficData(location, time, 'My_model'))
    # print(getTrafficData(location, time, 'GRU'))
    # print(getTrafficData(location, time, 'SAEs'))


# /?street1=WARRIGAL_RD&street2=STREET_RD&time=600&model=My_model
@flask_app.route("/api")
def traffic_api():
    street1 = request.args.get('street1', default="", type= str)
    street2 = request.args.get('street2', default="", type= str)
    time = request.args.get('time', default = 12*60, type = int)
    model_type = request.args.get('model', default = "lstm", type = str)

    foundLocation = next((x for x in dataList if x.street1 == street1 and x.street2 == street2), None)

    if foundLocation == None:
        return f"<p>Could not find the street data</p>"

    prediction = getTrafficData(foundLocation, time, model_type)

    return jsonify(prediction)

@flask_app.route("/locations")
def locations_api():
    return jsonify(dataList)

@flask_app.route("/")
def index():
    return render_template('index.html', locations = dataList)

initialiseModels()
prepareLocations()

if __name__ == '__main__':
    initialiseModels()
    # print(createTestDay())
    # print("aaaa")
    

    # main()

