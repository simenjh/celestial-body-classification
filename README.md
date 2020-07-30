# Predicting stars, galaxies and quasars with neural network

## Summary
In this project the celestial bodies stars, galaxies and quasars are predicted using a neural network implemented from scratch. The neural network uses mini-batch gradient descent with Adam optimization and Bayesian hyperparameter tuning.

The test accuracy is 98.5%, which is about the best you can get for the dataset. 

Dataset: [Sloan Digital Sky Survey DR14](https://www.kaggle.com/lucidlenn/sloan-digital-sky-survey)

## Introduction
### Dataset
From the dataset description: "The data consists of 10,000 observations of space taken by the SDSS. Every observation is described by 17 feature columns and 1 class column which identifies it to be either a star, galaxy or quasar."

#### Features
* objid = Object Identifier
* ra = J2000 Right Ascension (r-band)
* dec = J2000 Declination (r-band)
* u (ultraviolet)= better of DeV/Exp magnitude fit
* g (green) = better of DeV/Exp magnitude fit
* r (red) = better of DeV/Exp magnitude fit
* i (Near infrared) = better of DeV/Exp magnitude fit
* z (Infrared) = better of DeV/Exp magnitude fit
* run = Run Number
* rereun = Rerun Number
* camcol = Camera column
* field = Field number
* specobjid = Object Identifier
* class = object class (galaxy, star or quasar object)
* redshift = Final Redshift
* plate = plate number
* mjd = MJD of observation
* fiberid = fiber ID

### Star
A star is an astronomical object consisting of a luminous spheroid of plasma held together by its own gravity.
![Star](images/star.jpg?raw=true)

### Galaxy
A galaxy is a gravitationally bound system of stars, stellar remnants, interstellar gas, dust, and dark matter.
![Galaxy](images/galaxy.jpg?raw=true)

### Quasar
A quasar (also known as a quasi-stellar object abbreviated QSO) is an extremely luminous active galactic nucleus (AGN), in which a supermassive black hole with mass ranging from millions to billions of times the mass of the Sun is surrounded by a gaseous accretion disk.
![Quasar](images/quasar.jpg?raw=true)

## Distribution of celestial bodies
![Celestial body distribution](images/celestial_body_distribution.png?raw=true)


## Neural network architecture
The neural network has 1-2 hidden layers, each with 5-32 hidden units. It uses mini-batch gradient descent with Adam optimization. The hyperparameters, including units and layers are chosen by the Tree Parzen Estimator algorithm to build the surrogate model used in the Bayesien hyperparameter tuning approach. 


## Results
* Overall test accuracy: 98.5%
* Stars accuracy: 99.0%
* Galaxies accuracy: 98.3%
* Quasars accuracy: 96.7%


## Run program
1. import celestial_body_classification.py
2. Run the function celestial_body("stars_galaxies_quasars.csv")

