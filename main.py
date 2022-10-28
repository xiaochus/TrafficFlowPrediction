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
from sympy import symbols, solve
from flask import Flask, request, jsonify, render_template
import json
warnings.filterwarnings("ignore")

from route import prepareRoutes, pathFind, calculateWeight, coordData


lstm = None
gru = None
saes = None
my_model = None
y_scaler = None
lat_scaler = None
long_scaler = None
flask_app = Flask(__name__)
dataList = []


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

# Time is expected in minutes past 12am
def getTrafficData(locationData: coordData, time: float, model: Literal['lstm', 'gru', 'saes', 'my_model']):
    correctedTime = time / 1440
    preparedLatitude = lat_scaler.transform(np.array(locationData.latitude).reshape(1, -1))[0][0]
    preparedLongitude = long_scaler.transform(np.array(locationData.longitude).reshape(1, -1))[0][0]
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
    # lstm = load_model('model/lstm.h5')
    gru = load_model('model/gru.h5')
    # saes = load_model('model/saes.h5')
    my_model = load_model('model/my_model.h5')
    print("Models loaded! Processing data")

    file1 = 'data/BoroondaraData.csv'
    file2 = ''
    _, _, _, _, y_scaler, lat_scaler, long_scaler = process_data(file1, '', 0)

    print(lat_scaler.data_min_)
    print(lat_scaler.data_max_)
    print(lat_scaler.data_range_)

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
    origin = request.args.get('origin', default=0, type = int)
    destination = request.args.get('destination', default=0, type = int)
    time = request.args.get('time', default = 12*60, type = int)
    model_type = request.args.get('model', default = "lstm", type = str)

    # foundOrigin = next((x for x in dataList if x.id == origin), None)
    # foundDestination = next((x for x in dataList if x.id == destination), None)

    # if foundOrigin == None or foundDestination == None:
    #     print(f'ERROR: Could not find origin or destination: {origin}, {foundOrigin} | {destination}, {foundDestination}')
    #     error = {error: "Could not find location"}
    #     return jsonify(error)

    path = pathFind(dataList, origin, destination)
    totalTimeAndDistance = findRouteTime(path, time, model_type)

    # print(path)
    # print(totalTimeAndDistance)

    streetList = []
    for loc in path:
        streetList.append(f'{loc.street1}, {loc.street2}')

    print(path)
    print(streetList)

    streetList.reverse()

    data = {
        "route": streetList,
        "totalTime": str(totalTimeAndDistance[0]),
        "distance": str(totalTimeAndDistance[1])
    }

    return jsonify(data)

@flask_app.route("/locations")
def locations_api():
    return jsonify(dataList)

@flask_app.route("/")
def index():
    return render_template('index.html', locations = dataList)

A = -(200/32**2)
B = -2 * 32 * A

def findRouteTime(path, startTime, model):
    times = []
    distances = []
    currentTime = startTime
    i = 0

    print(f'path length: {len(path)}')
    while i < (len(path)-1):
        flow = getTrafficData(path[i], currentTime, model)
        y = symbols('y')
        expr = ((A*(y**2)) + (B*y)) - flow
        sol = solve(expr)

        tempDistance = calculateWeight(path[i], path[i+1]) #gets distance in degrees
        distance = tempDistance * 111 #gives distance in kilometers (roughly)
        distances.append(distance)
        timeTaken = (distance/sol[1]) * 60 #gives time in minutes, uses speed assuming low traffic
        currentTime += int(timeTaken)
        times.append(timeTaken)
        i += 1

    totalTime = 0
    for x in times:
        totalTime += x

    totalDistance = 0
    for x in distances:
        totalDistance += x

    print(f'time v dist: {totalTime} : {totalDistance}')
    return (totalTime, totalDistance) #returns time in minutes and distance in km

initialiseModels()
dataList = prepareRoutes()
# path = pathFind(dataList, 34, 19)
# totalTimeAndDistance = findRouteTime(path, 12*60, 'my_model')



# if __name__ == '__main__':
#     initialiseModels()
    # print(createTestDay())
    # print("aaaa")
    

    # main()

