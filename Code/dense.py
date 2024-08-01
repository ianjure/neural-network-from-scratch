def Dense(Weights, Bias, Input):
    # Z = W * X + B
    Output = Weights @ Input + Bias

    return Output