from activation import Sigmoid_Prime
from loss import MSE_Prime

def Backward(X, Y, Z1, A1, Z2, A2, W1, B1, W2, B2, LR):
    # COMPUTE THE GRADIENTS
    dE2 = MSE_Prime(A2, Y) * Sigmoid_Prime(Z2)
    dW2 = dE2 @ A1.T
    dB2 = dE2
    dA1 = W2.T @ dE2
    dE1 = dA1 * Sigmoid_Prime(Z1)
    dW1 = dE1 @ X.T
    dB1 = dE1

    # UPDATE THE WEIGHTS AND BIASES
    W2 = W2 - LR * dW2
    B2 = B2 - LR * dB2
    W1 = W1 - LR * dW1
    B1 = B1 - LR * dB1
    
    return W2, B2, W1, B1