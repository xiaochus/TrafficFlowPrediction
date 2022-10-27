"""
Processing the data
"""
from concurrent.futures import process
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils.np_utils import to_categorical
from math import floor


def process_data(train, test, timeFrame):
    """Process data
    Reshape and split train\test data.

    # Arguments
        train: String, name of .csv train file.
        test: String, name of .csv test file.
        lags: integer, time lag.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        X_test: ndarray.
        y_test: ndarray.
        scaler: StandardScaler.
    """

    # attr = "Lane 1 Flow (Veh/5 Minutes)"

    # df1 = pd.read_csv(train, encoding='utf-8').fillna(0)
    # df2 = pd.read_csv(test, encoding='utf-8').fillna(0)

    # scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
    # flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    # flow2 = scaler.transform(df2[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    # train, test = [], []
    # for i in range(lags, len(flow1)):
    #     train.append(flow1[i - lags: i + 1])
    # for i in range(lags, len(flow2)):
    #     test.append(flow2[i - lags: i + 1])

    # train = np.array(train)
    # test = np.array(test)
    # np.random.shuffle(train)

    # X_train = train[:, :-1]
    # y_train = train[:, -1]
    # X_test = test[:, :-1]
    # y_test = test[:, -1]

    # return X_train, y_train, X_test, y_test, scaler

    df1 = pd.read_csv(train, encoding='utf-8').fillna(0)
    firstColPos = df1.columns.get_loc("V00")
    
    flowData = df1.to_numpy()[:, firstColPos:]

    # highest = 0
    # for i in range(len(flowData)):
    #     for j in range(len(flowData[i])):
    #         if flowData[i][j] > highest:
    #             highest = flowData[i][j]
    #             print(f'{i} {j} = {highest}')

    latitude = df1['NB_LATITUDE'].to_numpy().reshape(-1, 1)
    longnitude = df1['NB_LONGITUDE'].to_numpy().reshape(-1, 1)
    
    flowScaler = MinMaxScaler(feature_range=(0, 1)).fit(flowData)
    latScaler = MinMaxScaler(feature_range=(0, 1)).fit(latitude)
    longScaler = MinMaxScaler(feature_range=(0, 1)).fit(longnitude)
    flowValues = flowScaler.transform(flowData)
    latitude = latScaler.transform(latitude)
    longnitude = longScaler.transform(longnitude)
    
    latlong = np.concatenate((latitude, longnitude), axis=1)

    # Convert to inputs
    # [[-60min, -45min, -30min, -15min, LAT, LONG, EXPECTED OUTPUT/0min], ...]
    # [[Time in mins, LAT, LONG, EXPECTED OUTPUT/0min], ...]

    # train = []
    # for i in range(0, len(flowValues)):
    #     for j in range(timeFrame, len(flowValues[i])):
    #         # print(flowValues[i, j - timeFrame : j + 1])
    #         # print(latitude[i])
    #         # print(longnitude[i])
    #         k = j
    #         if k >= len(flowValues[i]) - 1:
    #             k = -1
    #         inputs = np.append(flowValues[i, j - timeFrame : k], latlong[i])
    #         inputs = np.append(inputs, flowValues[i, k + 1])
    #         train.append(inputs)

    train = []
    for i in range(0, len(flowValues)):
        for j in range(96):
            k = j
            if k == 95:
                k = -1
            inputs = np.append([j * 15 / 1440], latlong[i])
            inputs = np.append(inputs, flowValues[i, k + 1])
            train.append(inputs)

    train = np.array(train)

    # splitIndex = floor(len(train) * 0.7)
    X_train = train[:, :-1]
    y_train = train[:, -1]

    # print(y_test)
    # highest = 0
    # for i in range(len(y_test)):
    #     if y_test[i] > highest:
    #         highest = y_test[i]
    #         print(f'{i} = {highest}')

    # np.random.shuffle(train[:splitIndex, :])

    # X_train = train[:splitIndex, :-1]
    # y_train = train[:splitIndex, -1]
    X_test = None
    y_test = None

    return X_train, y_train, X_test, y_test, flowScaler, latScaler, longScaler

# dfOrig = pd.read_csv('data/train.csv', encoding='utf-8').fillna(0)
# attr = "Lane 1 Flow (Veh/5 Minutes)"

# df1 = pd.read_csv('data/BoroondaraData.csv', encoding='utf-8').fillna(0)
# firstColPos = df1.columns.get_loc("V00")
# lastColPos = df1.columns.get_loc("V95")

# print(dfOrig[attr].to_numpy())
# print(dfOrig[attr].to_numpy().reshape(-1, 1))
# print(MinMaxScaler(feature_range=(0, 1)).fit(dfOrig[attr].values.reshape(-1, 1)).data_min_)
# print("-----------------")

# latitude = df1['NB_LATITUDE'].to_numpy().reshape(-1, 1)
# longnitude = df1['NB_LONGITUDE'].to_numpy().reshape(-1, 1)

# flowScaler = MinMaxScaler(feature_range=(0, 1)).fit(df1.to_numpy()[:, firstColPos:])
# latScaler = MinMaxScaler(feature_range=(0, 1)).fit(latitude)
# longScaler = MinMaxScaler(feature_range=(0, 1)).fit(longnitude)
# flowValues = flowScaler.transform(df1.to_numpy()[:, firstColPos:])
# latitude = latScaler.transform(latitude)
# longnitude = longScaler.transform(longnitude)


# flowValues = df1.to_numpy()[:, firstColPos:]
# latitude = df1['NB_LATITUDE'].to_numpy()
# longnitude = df1['NB_LONGITUDE'].to_numpy()
# latlong = np.concatenate((latitude, longnitude), axis=1)
# weeknum = df1['Weeknum'].to_numpy()

# Convert weeknum to one-hot encoding
# weeknum = to_categorical(weeknum)

# print(flowValues)
# print(latitude)
# print(latlong)
# print(len(flowValues))
# print(len(latitude))
# print(len(longnitude))
# print('---------------------------------------------------')

# Convert to inputs
# [[-60min, -45min, -30min, -15min, LAT, LONG, EXPECTED OUTPUT/0min], ...]

# timeFrame = 4;

# train = []
# for i in range(0, len(flowValues)):
#     for j in range(timeFrame, len(flowValues[i])):
#         # print(flowValues[i, j - timeFrame : j + 1])
#         # print(latitude[i])
#         # print(longnitude[i])
#         k = j
#         if k >= len(flowValues[i]) - 1:
#             k = -1
#         inputs = np.append(flowValues[i, j - timeFrame : k], latlong[i])
#         inputs = np.append(inputs, flowValues[i, k + 1])
#         train.append(inputs)

# train = np.array(train)
# np.random.shuffle(train)

# split = 0.7
# splitIndex = floor(len(train) * 0.7)

# X_train = train[:splitIndex, :-1]
# y_train = train[:splitIndex, -1]
# X_test = train[splitIndex:, :-1]
# y_test = train[splitIndex:, -1]

# print(weeknum.shape)
# print(df1.iloc[0, firstColPos:lastColPos + 1].to_numpy().reshape(-1, 1))
# print("-----------------")
# print(df1.iloc[:, firstColPos:lastColPos + 1].to_numpy())
# print(df1.iloc[:, firstColPos:lastColPos + 1].to_numpy().reshape(-1, 2))


# scaler = MinMaxScaler(feature_range=(0, 1)).fit(dfOrig[attr].values.reshape(-1, 1))
# scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1.iloc[:, firstColPos:].values.reshape(-1, 1))
# flow1 = scaler.transform(dfOrig[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

# lags = 12

# train = []
# for i in range(lags, len(flow1)):
#     train.append(flow1[i - lags: i + 1])

# train = np.array(train)
# np.random.shuffle(train)

# print(len(train[0]))

# X_train = train[:, :-1]
# y_train = train[:, -1]

# print(train[:, :-1])
# print(train[:, -1])
# print(dfOrig[attr].values.reshape(-1, 1))

# process_data('data/BoroondaraData.csv', '', 0)