import random
from matplotlib import pyplot as plt
from forward import Forward

def Test(X, W1, B1, W2, B2):
    index = random.randint(0, X.shape[0]-1)
    img = X[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")
    img.shape += (1,)

    # FORWARD PASS FOR TESTING
    Z1, A1, Z2, A2 = Forward(img, W1, B1, W2, B2)

    # PREDICTION
    plt.title(f"PREDICTION: {A2.argmax()}")
    plt.show()