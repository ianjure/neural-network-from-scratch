from dense import Dense
from activation import Sigmoid

def Forward(X, W1, B1, W2, B2):
    # INPUT -> HIDDEN (LAYER 1)
    Z1 = Dense(W1, B1, X) # DENSE LAYER
    A1 = Sigmoid(Z1)  # ACTIVATION LAYER

    # HIDDEN -> OUTPUT (LAYER 2)
    Z2 = Dense(W2, B2, A1) # DENSE LAYER
    A2 = Sigmoid(Z2) # ACTIVATION LAYER

    return Z1, A1, Z2, A2