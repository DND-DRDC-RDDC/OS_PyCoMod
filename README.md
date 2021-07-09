# OS_SIRplus

## Introduction

SIRplus is a Python package for building and running Susceptible-Infectious-Recovered (SIR) and similar compartment models derived from systems of differential equations.

The project is being released under Open Science (OS), an initiative of the Government of Canada to make the research products of federal scientists open to the public. SIRplus was developed by scientists in the Centre for Operational Research and Analysis (CORA) within Defence Research and Development Canada (DRDC) in order to model the spread of COVID-19 in specific populations of interest to the Canadian Armed Forces.

The key components of SIRplus are a set of Python classes:
 1. model element classes (pools, flows, parameters, equations, and random samples) are used to define the behavior of the system;
 1. users define a custom model class that contains these elements as well as nested sub-models;
 1. a run manager class is used to keep track of various models, initial conditions and saved output; and
 1. a plotter class is used for visualizing the results of model runs.

Internally, the package implements a numerical differential equation solver to solve the system of equations in discrete time-steps and generate time series output for the state of the system. SIRplus also allows discrete stochastic flow equations to be created, and the model can be automatically run repeatedly to conduct Monte Carlo simulations and plot the distribution of model outputs.

## Installation

To install SIRplus to your local Python environment (requires Git version control system), run the following from the command line:

    pip install git+https://github.com/DND-DRDC-RDDC/OS_SIRplus.git

To install SIRplus in [Google Colab](https://colab.research.google.com), run the following in a code cell:

    ! pip install git+https://github.com/DND-DRDC-RDDC/OS_SIRplus.git

After installing the package, import SIRplus into your code.

```Python
import sirplus as sp
```

The examples that follow were tested in [Google Colab](https://colab.research.google.com) and assume that SIRplus has been installed and imported as above using the abreviated name *sp*.

## A simple SIR model example

The susceptible-infectious-recovered (SIR) model of disease spread consists of the three compartments (S, I and R), and two flows that move individuals from susceptible to infectious and from infectious to recovered.

The flow of individuals from S to I is given by the rate

<img src="https://render.githubusercontent.com/render/math?math=F_{si}=bI\frac{S}{N}">

where *b* is the transmission rate and *N* is the total population, <img src="https://render.githubusercontent.com/render/math?math=S%2BI%2BR">.

The flow of individuals from I to R is given the rate

<img src="https://render.githubusercontent.com/render/math?math=F_{ir}=gI">

where *g* is the recovery rate.

This produces the following system of differential equations:

<img src="https://render.githubusercontent.com/render/math?math=\frac{dS}{dt}=-bI\frac{S}{N}">

<img src="https://render.githubusercontent.com/render/math?math=\frac{dI}{dt}=bI\frac{S}{N}-gI">

<img src="https://render.githubusercontent.com/render/math?math=\frac{dR}{dt}=gI">

If we have a population of 100 with 5 initial cases of infection and the remaining 95 being susceptible, we can model this system in SIRplus with the following code:

```python
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
    
    #output
    self._set_output('S', 'I', 'R')
```

The first line, above, imports the sirplus package. We then define a custom class inheriting from the SIRplus *model* class and override the model's *_build* function to define the elements of the model. In this case, we create the three compartments using sirplus *pools*, specifying the initial value of each pool. We define the value *N* (the total population) as a sirplus *equation*. Equations are defined by a function referencing other model elements. Lamda functions are syntactically compact for this purpose. To obtain the value of a model element, we call the object (add open and close-brackets). For example, the current number of susceptible individuals is obtained by *self.S()*. We then create the transmission rate, *b*, and recovery rate, *g*, using sirplus *parameters*, and specify their values. Next, we define the movement between the compartments using sirplus *flows*. Flows are defined by a function, similar to the equation class. In this case, the flow functions corresponding to the rate equations, *Fsi* and *Fir*, defined above. Flows must also specify a source pool and a destination pool. Note that when specifying source and destination pools, reference the pool object, but don't call it (e.g. *src=self.S*, not *src=self.S()*). A final step in specifying the model is to let SIRplus know which outputs we want to capture for analysis. This is done by calling the model's *_set_output* function and providing the names of the model elements that we want to track.

Lastly, having defined the *simple_sir* model class, we create an instance of it.

```Python
m = simple_sir()
```

Having created the model, we use another SIRplus object called a *run_manager* to run it. The run manager keeps track of multiple models, run settings and output so that batches of runs can be automated. First we create an instance of the run manager.

```Python
mgr = sp.run_manager()
```

Now we can tell the run manager to run the simple_SIR model. We can supply run settings (such as the duration in this example), and we must provide a label as a key to access the run results later.

```Python
mgr.run(m, duration=150, label='My run')
```

Finally, we can plot the results of the run using the SIRplus *plotter*.  First we create an instance of the plotter, which creates a Matplotlib Figure, then we can plot the specific outputs from the run on the figure axes.

```Python
plt = sp.plotter(title='Infections', ylabel='Population', fontsize=14)
plt.plot(mgr['My run'],'S', color='blue', label = 'S')
plt.plot(mgr['My run'],'I', color='orange', label = 'I')
plt.plot(mgr['My run'],'R', color='green', label = 'R')
plt.plot(mgr['My run'],'S + I + R', color='black', label = 'Total')
```

Each call to the plotter's *plot* function must specify a run and an output. The run is identified by indexing the run manager with the run label used earlier. The output must be one of the outputs that was specified in the model using *_set_output*. Outputs can be summed together, e.g. *S + I + R* in the last line, above.


