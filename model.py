import numpy as np
from scipy.special import expit


def init_params_and_V(activation_layers):
    params = {}
    V = {}
    L = len(activation_layers)
    
    for l in range(1, L):
        params[f"W{l}"] = np.random.randn(activation_layers[l], activation_layers[l-1]) * np.sqrt(2 / activation_layers[l-1])
        params[f"b{l}"] = np.zeros((activation_layers[l], 1))

        # Momemntum params
        V[f"dW{l}"] = np.zeros((activation_layers[l], activation_layers[l-1]))
        V[f"db{l}"] = np.zeros((activation_layers[l], 1))

    return params, V



def train_mini_batch_model(X_batches, y_batches, parameters, V, epochs, learning_rate, reg_param):
    logging_frequency = 10
    costs_epochs = {"costs": [], "epochs": []}

    for i in range(epochs):
        cost = 0
        for j in range(len(X_batches)):
            AL, caches = forward_propagation(X_batches[j], parameters)
            cost += compute_cost(AL, y_batches[j], parameters, reg_param)

            gradients = backprop(AL, y_batches[j], caches)
            update_parameters(parameters, gradients, V, learning_rate, reg_param)

        if i % logging_frequency == 0:
            costs_epochs["costs"].append(cost)
            costs_epochs["epochs"].append(i)
                     
    return costs_epochs, parameters
            
        



def train_various_sizes(X_train, X_cv, y_train, y_cv, parameters, V, activation_layers, epochs, learning_rate, reg_param):
    costs_train, costs_cv, m_examples = [], [], []
    for i in range(1, X_train.shape[1], 20):
        parameters = init_params(X_train, activation_layers)
        costs_epochs, parameters = train_model(X_train[:, :i], y_train[:, :i], parameters, V, epochs, learning_rate, reg_param)

        AL_cv, caches = forward_propagation(X_cv, parameters)
        cost_cv = compute_cost(AL_cv, y_cv, parameters, reg_param)

        costs_train.append(costs_epochs["costs"][-1])
        costs_cv.append(cost_cv)
        m_examples.append(i)

    return costs_train, costs_cv, m_examples









        
        

    
    
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






def update_parameters(parameters, gradients, V, learning_rate, reg_param):
    L = len(parameters) // 2
    m = parameters["W1"].shape[1]
    beta = 0.9

    for l in range(1, L+1):
        reg_term = (reg_param / m) * parameters[f"W{l}"]

        # Momentum gradients
        V[f"dW{l}"] = beta * V[f"dW{l}"] + (1 - beta) * (gradients[f"dW{l}"] + reg_term)
        V[f"db{l}"] = beta * V[f"db{l}"] + (1 - beta) * (gradients[f"db{l}"])

        

        parameters[f"W{l}"] -= learning_rate * V[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * V[f"db{l}"]
        
        # parameters[f"W{l}"] -= learning_rate * (gradients[f"dW{l}"] + reg_term)
        # parameters[f"b{l}"] -= learning_rate * gradients[f"db{l}"]










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



def compute_accuracy(X, y, parameters):
    AL, caches = forward_propagation(X, parameters)
    y_pred = np.where(AL >= 0.5, 1, 0)

    comparison = np.where(y_pred == y, 1, 0)
    return np.sum(comparison) / y.shape[1]


def compute_accuracy(X, y, parameters):
    AL, caches = forward_propagation(X, parameters)
    
    y_ints = y.argmax(axis=0).reshape(1, -1)
    y_pred = AL.argmax(axis=0).reshape(1, -1)

    comparison = np.where(y_pred == y_ints, 1, 0)
    return np.sum(comparison) / y_ints.shape[1]
    
    
