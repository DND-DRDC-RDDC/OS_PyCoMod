# OS_SIRplus

SIRplus is a Python package for building and running Susceptible-Infectious-Recovered (SIR) and similar compartment models derived from systems of differential equations.

The key components of the package are a set of Python classes:
 1) model element classes (pools, flows, parameters, equations, and random samples) are used to define the behavior of the system;
 2) users define a custom model class that contains these elements as well as nested sub-models;
 3) a run manager class is used to keep track of various models, initial conditions and saved output; and
 4) a plotter class is used for visualizing the results of model runs.

Internally, the package implements a numerical differential equation solver to solve the system of equations in discrete time-steps and generate time series output for the state of the system. Unlike a pure differential equation solver, SIRplus allows discrete stochastic flow equations to be created, and the model can be automatically run repeatedly to conduct Monte Carlo simulations and plot the distribution of model outputs.
