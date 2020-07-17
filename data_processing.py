import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler



def read_and_preprocess_data(data_file):
    df = pd.read_csv(data_file)
    df.drop(["objid", "specobjid"], axis=1)

    y = df["class"].values
    X = df.drop(["class"], axis=1).values
    return X, y
    


def preprocess(dataset):
    X = dataset[:, :-1]
    y = dataset[:, -1].reshape(-1, 1)
    return X, y


def standardize(X1, *args):
    sc = StandardScaler()
    Xs_standard = sc.fit_transform(X1)
    if args != ():
        Xs_standard = [Xs_standard]
        Xs_standard.extend([sc.transform(Xi) for Xi in args])
    return Xs_standard


def make_batches(X_train_std, y_train, mini_batch_size):
    m = X.shape[1]
    X_batches = []
    y_batches = []
    
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(num_complete_minibatches):
        mini_batch_X = X_train_std[:, mini_batch_size * k : mini_batch_size * (k + 1)]
        mini_batch_Y = y_train[:, mini_batch_size * k : mini_batch_size * (k + 1)]
        X_batches.append(mini_batch_X)
        y_batches.append(mini_batch_y)

    if m % mini_batch_size != 0:
        mini_batch_X = X_train_std[:, mini_batch_size * num_complete_minibatches : mini_batch_size * num_complete_minibatches + (m - mini_batch_size * math.floor(m / mini_batch_size))]
        mini_batch_Y = y_train[:, mini_batch_size * num_complete_minibatches : mini_batch_size * num_complete_minibatches + (m - mini_batch_size * math.floor(m / mini_batch_size))]
        X_batches.append(mini_batch_X)
        y_batches.append(mini_batch_y)

    return X_batches, y_batches
    
