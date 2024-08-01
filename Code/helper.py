import numpy as np

def One_Hot_Encode(Label):
    Output = np.zeros((Label.size, Label.max() + 1))
    Output[np.arange(Label.size), Label] = 1
    Output = Output.T

    return Output