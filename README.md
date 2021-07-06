# OS_SIRplus

## Introduction

SIRplus is a Python package for building and running Susceptible-Infectious-Recovered (SIR) and similar compartment models derived from systems of differential equations.

The project is being released as part of Open Science (OS), an initiative of the Government of Canada to make the research products of federal scientists open to the public.

The key components of the package are a set of Python classes:
 1. model element classes (pools, flows, parameters, equations, and random samples) are used to define the behavior of the system;
 1. users define a custom model class that contains these elements as well as nested sub-models;
 1. a run manager class is used to keep track of various models, initial conditions and saved output; and
 1. a plotter class is used for visualizing the results of model runs.

Internally, the package implements a numerical differential equation solver to solve the system of equations in discrete time-steps and generate time series output for the state of the system. Unlike a pure differential equation solver, SIRplus allows discrete stochastic flow equations to be created, and the model can be automatically run repeatedly to conduct Monte Carlo simulations and plot the distribution of model outputs.

## A simple SIR model example

The basic SIR compartment model of disease spread moves the individuals in a population through three compartments: susceptible (S), infectious (I) and recovered (R). Initially, the vast majority of the population will be susceptible, but one or a few seed cases in the infectious compartment will spread the infection to those in the susceptible compartment moving them to the infectious compartment, potentially leading to an exponential growth in infections. Eventually, infectious individuals move to the recovered compartment and stop spreading the disease and are no longer susceptible. The depletion of the suscpetible individuals is what ultimately causes an outbreak to cease.

The flow of individuals from S to I is given by the rate

<img src="https://render.githubusercontent.com/render/math?math=b*I*\frac{S}{N}">

where *b* is the transmission rate and *N* is the total population, <img src="https://render.githubusercontent.com/render/math?math=S%2BI%2BR">.

The flow of individuals from I to R is given the rate

<img src="https://render.githubusercontent.com/render/math?math=g*I">

where *g* is the recovery rate.

This produces the following system of differential equations:

<img src="https://render.githubusercontent.com/render/math?math=\frac{dS}{dt}=-b*I*\frac{S}{N}">

<img src="https://render.githubusercontent.com/render/math?math=\frac{dI}{dt}=b*I*\frac{S}{N}-g*I">

<img src="https://render.githubusercontent.com/render/math?math=\frac{dR}{dt}=g*I">

If we have a population of 100 with 5 initial cases of infection and the remaining 95 being susceptible, we can model this system in SIRplus with the following code:

```python
import sirplus as sp

class simple_sir(sp.model):
  def _build(self):
    #pools
    self.S = sp.pool(95)
    self.I = sp.pool(5)
    self.R = sp.pool(0)
    
    #equations
    self.N = sp.equation(lambda: self.S() + self.I() + self.R())
    
    #parameters
    self.b = sp.parameter(0.2)
    self.g = sp.parameter(0.1)
    
    #flows
    self.Fsi = sp.flow(lambda: self.b()*self.I()*self.S()/self.N(), src=self.S, dest=self.I)
    self.Fir = sp.flow(lambda: self.g()*self.I(), src=self.I, dest=self.R)

m = simple_sir()
```

The first line, above, imports the sirplus package. We then define a custom model class inheriting from the sirplus *model* class and override the model's *_build* function to define the elements of the model. In this case, we create the three compartments using the sirplus *pool* class. We define the value *N* as a sirplus *equation*. Equations are defined by a function referencing other model elements. Lamda functions are syntactically compact for this purpose. To obtain the value of a model element, we simple call the object. For example, the current number of susceptible individuals is obtained by "self.S()". We then define the transmission rate, *b*, and recovery rate, *g*, using the sirplus *parameter* class. Finally, we define the two flows from *S* to *I*, *Fsi*, and from *I* to *R*, *Fir* using the sirplus *flow* class. Flows are defined by a flow equation, again conveniently written as lambda functions, which correspond to the flows in the mathematical form described above. Flows must also specify a source pool and a destination pool. Lastly, having defined the *simple_sir* model class, we create an instance of it.




