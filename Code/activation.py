import numpy as np

def Sigmoid(Input):
    # Sigmoid = 1 / 1 + e^-z
    Output = 1 / (1 + np.exp(-Input))

    return Output

def Sigmoid_Prime(Input):
    # Sigmoid' = Sigmoid(Z) * (1 - Sigmoid(Z))
    Output = (Sigmoid(Input) * (1 - Sigmoid(Input)))

    return Output

def ReLU(Input):
    """
    ReLU = Input if Input > 0
    ReLU = 0 if Input <= 0
    """
    Output = np.maximum(0, Input)

    return Output

def ReLU_Prime(Input):
    """
    ReLU' = 1 if Input > 0
    ReLU' = 0 if Input <= 0
    """
    Output = np.where(Input > 0, 1, 0)
    
    return Output