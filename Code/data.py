import numpy as np

def Format_Data(Dataframe):
    train_data = np.array(Dataframe)
    M, N = train_data.shape

    train_data_T = train_data.T

    Y = train_data_T[0]
    X = train_data_T[1:N]

    X = X / 255.

    return X, Y