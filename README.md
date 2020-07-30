# Predicting stars, galaxies and quasars with neural network

## Summary
In this project the celestial bodies stars, galaxies and quasars are predicted using a neural network implemented from scratch. The neural network uses mini-batch gradient descent with Adam optimization and Bayesian hyperparameter tuning.

Test accuracy is 98%, which is about the best you can get for the dataset. 

Dataset: [Sloan Digital Sky Survey DR14](https://www.kaggle.com/lucidlenn/sloan-digital-sky-survey)

## Introduction

### Dataset
From the dataset description: "The data consists of 10,000 observations of space taken by the SDSS. Every observation is described by 17 feature columns and 1 class column which identifies it to be either a star, galaxy or quasar."

#### Features

### Star
Short description

### Galaxy
Short description

### Quasar
Short description



## Distribution of celestial bodies
Insert visual representation



## Neural network architecture



## Results
Test accuracy: 0.985
Stars accuracy: 0.9906976744186047
Galaxies accuracy: 0.9827935222672065
Quasars accuracy: 0.9671052631578947



<br />

## Run program

<br /> <br />


### Results
![](images/random_results.png?raw=true)

Above, you can see the results from a single test run of 250 random selections of hyperparameters. 
Running the random search algorithm about 50 times has resulted in cross-validation accuracies of 84% - 90%. 

 <br />



## Bayesian hyperparameter optimization
Contrary to random search, this approach considers the performance of previously selected hyperparameters when selecting which hyperparameters to try next.

Bayesian optimization finds the loss that minimizes an objective function. It does this by building a surrogate function (probability model) that is built from past evaluation results of the objective function. The surrogate function is much cheaper to evaluate than the objective function. Values returned from the surrogate function are selected using an expected improvement criterion.

The process can be described like this:
1. Build a surrogate probability model of the objective function.
2. Find the hyperparameters that perform best on the surrogate.
3. Apply these hyperparameters to the true objective function.
4. Update the surrogate model incorporating the new results.
5. Repeat steps 2–4 until max iterations or time is reached.

I have used 1 - cross validation accuracy as the the value returned by the objective function, and the loss to be minimized. 

For this project, Bayesian optimization is done with the Hyperopt library, and the surrogate used is Tree Parzen Estimator (TPE). 

### Results
![](images/bayes_results.png?raw=true)

In the table above, you can see results from running the bayesian optimizer with random initialization of data to training and cross validation sets. The accuracy on the cross-validation set is about 90%. In general it ranges from 84% - 94% depending on random initialization of data. 

