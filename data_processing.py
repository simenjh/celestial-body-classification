import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler



def read_and_preprocess_data(data_file):
    df = pd.read_csv(data_file)
    df["class"] = pd.factorize(df["class"])[0]

    temp_class_values = df["class"].values
    X = df.drop(["objid", "specobjid", "class"], axis=1).values

    # One-hot encoding
    y = np.zeros((temp_class_values.size, temp_class_values.max()+1))
    y[np.arange(temp_class_values.size), temp_class_values] = 1
    
    return X, y
   


def standardize(X1, *args):
    sc = StandardScaler()
    Xs_standard = sc.fit_transform(X1)
    if args != ():
        Xs_standard = [Xs_standard]
        Xs_standard.extend([sc.transform(Xi) for Xi in args])
    return Xs_standard



def make_batches(X_train_std, y_train, mini_batch_size):
    m = y_train.shape[1]
    X_batches = []
    y_batches = []
    
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(num_complete_minibatches):
        interval_first = mini_batch_size * k
        interval_last = mini_batch_size * (k + 1)
        
        mini_batch_X = X_train_std[:, interval_first : interval_last]
        mini_batch_y = y_train[:, interval_first : interval_last]
        X_batches.append(mini_batch_X)
        y_batches.append(mini_batch_y)

    if m % mini_batch_size != 0:
        interval_first = mini_batch_size * num_complete_minibatches 
        interval_last = mini_batch_size * num_complete_minibatches + (m - mini_batch_size * math.floor(m / mini_batch_size))
        
        mini_batch_X = X_train_std[:, interval_first : interval_last]
        mini_batch_y = y_train[:, interval_first : interval_last]
        X_batches.append(mini_batch_X)
        y_batches.append(mini_batch_y)

    return X_batches, y_batches
    
