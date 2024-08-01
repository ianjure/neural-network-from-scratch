import numpy as np

def MSE(Y_Prediction, Y_True):
    # Mean Squared Error (Single) = ∑ (ŷ - Y)^2
    Output = np.sum((Y_Prediction - Y_True) ** 2, axis=0)

    return Output[0]

def MSE_Prime(Y_Prediction, Y_True):
    # MSE' (Single) = ŷ - Y
    Output = Y_Prediction - Y_True

    return Output