import numpy as np
from scipy.special import expit


def init_params_V_and_S(activation_layers):
    params = {}
    V = {}
    S = {}
    L = len(activation_layers)
    
    for l in range(1, L):
        # Kaiming He initialization
        params[f"W{l}"] = np.random.randn(activation_layers[l], activation_layers[l-1]) * np.sqrt(2 / activation_layers[l-1])
        params[f"b{l}"] = np.zeros((activation_layers[l], 1))

        # Momentum params
        V[f"dW{l}"] = np.zeros((activation_layers[l], activation_layers[l-1]))
        V[f"db{l}"] = np.zeros((activation_layers[l], 1))

        # RMSprop params
        S[f"dW{l}"] = np.zeros((activation_layers[l], activation_layers[l-1]))
        S[f"db{l}"] = np.zeros((activation_layers[l], 1))

    return params, V, S



def train_mini_batch_model(X_batches, y_batches, parameters, V, S, epochs, learning_rate, reg_param):
    t = 0
    logging_frequency = 10
    costs_epochs = {"costs": [], "epochs": []}

    for i in range(epochs):
        cost = 0
        for j in range(len(X_batches)):
            batch_size = X_batches[j].shape[1]
            AL, caches = forward_propagation(X_batches[j], parameters)
            cost += compute_cost(AL, y_batches[j], parameters, reg_param)

            t +=1 
            gradients = backprop(AL, y_batches[j], caches)
            update_parameters(parameters, gradients, V, S, batch_size, t, learning_rate, reg_param)

        if i % logging_frequency == 0:
            costs_epochs["costs"].append(cost)
            costs_epochs["epochs"].append(i)
                     
    return costs_epochs, parameters
            
        




def backprop(AL, y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    y = y.reshape(AL.shape)
    epsilon = 1e-10

    current_cache = caches[L-1]
    grads[f"dA{L-1}"], grads[f"dW{L}"], grads[f"db{L}"] =  linear_activation_backward(current_cache, "softmax", AL=AL, y=y)

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads[f"dA{l}"], grads[f"dW{l+1}"], grads[f"db{l+1}"] = linear_activation_backward(current_cache, "relu", grads[f"dA{l+1}"])

    return grads



def linear_activation_backward(cache, activation, dA=None, AL=None, y=None):
    dA_prev, dZ, dW, db = None, None, None, None

    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "softmax":
        dZ = AL - y

    da_prev, dW, db = linear_backward(dZ, linear_cache)
    return da_prev, dW, db


def linear_backward(dZ, linear_cache):
    m = dZ.shape[1]
    dW = (1 / m) * np.dot(dZ, linear_cache["A"].T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(linear_cache["W"].T, dZ)
    return dA_prev, dW, db


def relu_backward(dA, activation_cache):
    dZ = dA * relu_deriv(activation_cache["Z"])
    return dZ

def relu_deriv(Z):
    return np.where(Z >= 0, 1, 0)

    






def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for i in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters[f"W{i}"], parameters[f"b{i}"], "relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters[f"W{L}"], parameters[f"b{L}"], "softmax")
    caches.append(cache)
    return AL, caches


def linear_activation_forward(A, W, b, activation):
    cache = None
    Z, linear_cache = linear_forward(A, W, b)
    
    if activation == "relu":
        A, activation_cache = relu_forward(Z)
        cache = (linear_cache, activation_cache)
    elif activation == "softmax":
        A, activation_cache = softmax_forward(Z)
        cache = (linear_cache, activation_cache)

    return A, cache


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    linear_cache = {"A": A, "W": W, "b": b}
    return Z, linear_cache


def relu_forward(Z):
    activation_cache = {"Z": Z}
    return Z * (Z > 0), activation_cache

def softmax_forward(Z):
    activation_cache = {"Z": Z}
    return softmax(Z), activation_cache

def softmax(Z):
    t = np.exp(Z)
    t_sum = t.sum(axis=0)
    return t / t_sum






def update_parameters(parameters, gradients, V, S, batch_size, t, learning_rate, reg_param):
    L = len(parameters) // 2
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-4
    
    for l in range(1, L+1):
        reg_term = (reg_param / batch_size) * parameters[f"W{l}"]

        # Momentum gradients
        V[f"dW{l}"] = beta1 * V[f"dW{l}"] + (1 - beta1) * (gradients[f"dW{l}"] + reg_term)
        V[f"db{l}"] = beta1 * V[f"db{l}"] + (1 - beta1) * (gradients[f"db{l}"])

        # RMSprop gradients
        S[f"dW{l}"] = beta2 * S[f"dW{l}"] + (1 - beta2) * np.square(gradients[f"dW{l}"] + reg_term)
        S[f"db{l}"] = beta2 * S[f"db{l}"] + (1 - beta2) * np.square(gradients[f"db{l}"])

        # Bias correction
        V_dW_corrected = V[f"dW{l}"] / (1 - (beta1**t))
        V_db_corrected = V[f"db{l}"] / (1 - (beta1**t))
        S_dW_corrected = S[f"dW{l}"] / (1 - (beta2**t))
        S_db_corrected = S[f"db{l}"] / (1 - (beta2**t))

        # Adam optimization
        parameters[f"W{l}"] -= learning_rate * (V_dW_corrected / (np.sqrt(S_dW_corrected) + epsilon))
        parameters[f"b{l}"] -= learning_rate * (V_db_corrected / (np.sqrt(S_db_corrected) + epsilon))

        






def compute_cost(AL, y, parameters, reg_param):
    m = AL.shape[1]
    epsilon = 1e-10
    reg_term = regularize_cost(parameters, m, reg_param)
    cost = -(1 / m) * np.sum(y * np.log(AL + epsilon))
    return cost + reg_term
    


def regularize_cost(parameters, m, reg_param):
    L = len(parameters) // 2
    temp = 0
    for i in range(1, L+1):
        temp += np.sum(parameters[f"W{i}"]**2)

    return (reg_param / (2 * m)) * temp



def compute_test_accuracy(X, y, parameters):
    AL, caches = forward_propagation(X, parameters)
    
    y_classes = y.argmax(axis=0).reshape(1, -1)
    y_pred = AL.argmax(axis=0).reshape(1, -1)

    test_accuracy = np.sum(np.where(y_classes == y_pred, 1, 0)) / y_pred.shape[1]

    y_stars = np.where(y_classes == 0, 1, 0)
    y_pred_stars = np.where(y_pred == 0, 1, 0)

    y_galaxies = np.where(y_classes == 1, 1, 0)
    y_pred_galaxies = np.where(y_pred == 1, 1, 0)

    y_quasars = np.where(y_classes == 2, 1, 0)
    y_pred_quasars = np.where(y_pred == 2, 1, 0)

    stars_accuracy = np.sum(y_stars * y_pred_stars) / np.sum(y_stars)
    galaxies_accuracy = np.sum(y_galaxies * y_pred_galaxies) / np.sum(y_galaxies)
    quasars_accuracy = np.sum(y_quasars * y_pred_quasars) / np.sum(y_quasars)
    
    return {"test": test_accuracy, "stars": stars_accuracy, "galaxies": galaxies_accuracy, "quasars": quasars_accuracy}
    
