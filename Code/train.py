import numpy as np
from initialize import Initialize
from loss import MSE
from forward import Forward
from backward import Backward

def Train(X, Y, Epochs, Alpha):
    # INITIALIZE WEIGHTS AND BIASES
    W1, B1, W2, B2 = Initialize(784, 20, 10)

    # TRACK ACCURACY AND LOSS
    accuracy = 0
    loss = 0

    ## HYPERPARAMETER
    A = Alpha
    E = Epochs

    for epoch in range(E):
        for IMG, LABEL in zip(X, Y):

            # ADD A DIMENSION TO IMG AND LABEL
            IMG.shape += (1,) # (784,) -> (784, 1)
            LABEL.shape += (1,) # (10,) -> (10, 1)

            """
            GRADIENT DESCENT:

            FORWARD PASS --> ERROR CALCULATION
            --> BACKWARD PASS --> UPDATE WEIGHTS AND BIASES
            """

            # FORWARD PASS
            Z1, A1, Z2, A2 = Forward(IMG, W1, B1, W2, B2)

            # LOSS AND ACCURACY CALCULATION
            loss = MSE(A2, LABEL)
            accuracy += int(np.argmax(A2) == np.argmax(LABEL))

            # BACKWARD PASS
            W2, B2, W1, B1 = Backward(IMG, LABEL, Z1, A1, Z2, A2, W1, B1, W2, B2, A)

        # TRACK ACCURACY AND LOSS EVERY EPOCH
        print(f"Epoch {epoch} | Accuracy: {round((accuracy / X.shape[0]) * 100, 2)}% | Loss:", "{:.5f}".format(loss))

        # RESET VARIABLES
        accuracy = 0
        loss = 0
    
    return W1, B1, W2, B2