import data_processing as dproc
from sklearn.model_selection import train_test_split
import model
import dataplot
import numpy as np
from hyperopt import hp
from hyperopt import tpe
from hyperopt import fmin
from hyperopt import STATUS_OK
from hyperopt.pyll.base import scope
from hyperopt import Trials



X_train_std = None
X_cv_std = None
X_test_std = None
y_train = None
y_cv = None
y_test = None

X_batches = None
y_batches = None




def celestial_body(data_file, epochs=10, learning_rate=0.1, reg_param=0.1, batch_size=64):
    global X_train_std, X_cv_std, X_test_std, y_train, y_cv, y_test, X_batches, y_batches

    # dataplot.plot_class_distribution(data_file)
    
    X, y = dproc.read_and_preprocess_data(data_file)
    
    X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=(1/4))

    # Standardize data
    X_train_std, X_cv_std, X_test_std = dproc.standardize(X_train, X_cv, X_test)

    # Make mini batches
    X_batches, y_batches = dproc.make_batches(X_train_std.T, y_train.T, batch_size)
    
    bayes_tune(plot_results)
    


    


    

def bayes_tune(plot_results):
    space = {
        "hidden_layers": hp.choice("options", [{"hidden_layers": 1, "network": (X_train_std.shape[1], scope.int(hp.quniform("1_hidden_1", 5, 32, 1)), 3)},
                                               {"hidden_layers": 2, "network": (X_train_std.shape[1], scope.int(hp.quniform("2_hidden_1", 5, 32, 1)), scope.int(hp.quniform("2_hidden_2", 5, 32, 1)), 3)}]),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.0001), np.log(0.2)),
        "epochs": scope.int(hp.quniform("epochs", 5, 50, 1)),
        "reg_param": hp.loguniform("reg_param", np.log(0.001), np.log(1))
    }

    bayes_trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=1, trials=bayes_trials)

    best_result = bayes_trials.best_trial["result"]

    test_accuracies = model.compute_test_accuracy(X_test_std.T, y_test.T, best_result["parameters"])

    print(f"Test accuracy: {test_accuracies['test']}")
    print(f"Stars accuracy: {test_accuracies['stars']}")
    print(f"Galaxies accuracy: {test_accuracies['galaxies']}")
    print(f"Quasars accuracy: {test_accuracies['quasars']}")

    if plot_results:
        dataplot.plot_results_bayes(bayes_trials)


    
        
def objective(hyper_params):
    activation_layers = tuple(int(a) for a in hyper_params["hidden_layers"]["network"])
    
    learning_rate = hyper_params["learning_rate"]
    epochs = int(hyper_params["epochs"])
    reg_param = hyper_params["reg_param"]
    parameters, V, S = model.init_params_V_and_S(activation_layers)

    model.train_mini_batch_model(X_batches, y_batches, parameters, V, S, epochs, learning_rate, reg_param)

    train_accuracy = model.compute_accuracy(X_train_std.T, y_train.T, parameters)
    cv_accuracy = model.compute_accuracy(X_cv_std.T, y_cv.T, parameters)

    loss = 1 - cv_accuracy
    
    print(f"CV accuracy: {cv_accuracy}")

    return {"loss": loss, "train_accuracy": train_accuracy, "cv_accuracy": cv_accuracy, "hyper_params": hyper_params, "status": STATUS_OK, "parameters": parameters,}


    
