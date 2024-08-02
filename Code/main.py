"""
SIMPLE NEURAL NETWORK (3 LAYERS)

INPUT -> HIDDEN LAYER [ DENSE LAYER -> ACTIVATION LAYER ]
-> OUTPUT LAYER [ DENSE LAYER -> ACTIVATION LAYER ]

Author: Ian Jure Macalisang
Email: ianjuremacalisang2@gmail.com
Accuracy: 90-95%
"""

import pandas as pd
from data import Format_Data
from helper import One_Hot_Encode
from train import Train
from test import Test

data = pd.read_csv('mnist_train.csv') # NOTE: Put data in the same directory.

X_train, Y_train = Format_Data(data)
images = X_train.T
labels = One_Hot_Encode(Y_train).T

W1, B1, W2, B2 = Train(images, labels, 20, 0.01)
Test(images, W1, B1, W2, B2)
