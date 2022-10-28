"""
Train the NN model.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data
from model import model
from keras.models import Model
from keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")


def train_model(model, X_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    model.save('model/' + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/' + name + ' loss.csv', encoding='utf-8', index=False)


def train_seas(models, X_train, y_train, name, config):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    temp = X_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(p.input,
                                       p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to train.")
    args = parser.parse_args()

    lag = 4
    config = {"batch": 256, "epochs": 600}
    file1 = 'data/BoroondaraData.csv'
    file2 = 'data/test.csv'
    X_train, y_train, _, _, _, _, _ = process_data(file1, file2, lag)

    # print(X_train)
    # print(X_train[0])
    # print(X_train.shape)

    # return

    # if args.model == 'my_model':
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    m = model.get_my_model([3, 32, 64, 1])
    train_model(m, X_train, y_train, 'my_model', config)
    # if args.model == 'lstm':
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    m = model.get_lstm([3, 64, 64, 1])
    train_model(m, X_train, y_train, 'lstm', config)
    # if args.model == 'gru':
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # m = model.get_gru([3, 64, 64, 1])
    # train_model(m, X_train, y_train, args.model, config)
    # if args.model == 'saes':
    #     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    #     m = model.get_saes([3, 400, 400, 400, 1])
    #     train_seas(m, X_train, y_train, args.model, config)


if __name__ == '__main__':
    main(sys.argv)
