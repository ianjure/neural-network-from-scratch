import numpy as np

def Initialize(In_Neuron, Hidden_Neuron, Out_Neuron):
    # LAYER 1
    W1 = np.random.uniform(-0.5, 0.5, (Hidden_Neuron, In_Neuron))
    B1 = np.zeros((Hidden_Neuron, 1))

    # LAYER 2
    W2 = np.random.uniform(-0.5, 0.5, (Out_Neuron, Hidden_Neuron))
    B2 = np.zeros((Out_Neuron, 1))

    return W1, B1, W2, B2