# OS_SIRplus

## Introduction

SIRplus is a Python package for building and running Susceptible-Infectious-Recovered (SIR) and similar compartment models derived from systems of differential equations.

The project is being released under Open Science (OS), an initiative of the Government of Canada to make the research products of federal scientists open to the public. SIRplus was developed by scientists in the Centre for Operational Research and Analysis (CORA) within Defence Research and Development Canada (DRDC) in order to model the spread of COVID-19 in specific populations of interest to the Canadian Armed Forces.

The primary developers and contributors to this work are:
 - Mr. Stephen Okazawa
 - Ms. Josée van den Hoogen
 - Dr. Steve Guillouzic

SIRplus is composed of a set of Python classes:
 1. model element classes (pools, flows, parameters, equations, and random samples) are used to define the behavior of the system;
 1. users define a custom model class that contains these elements as well as nested sub-models;
 1. a run manager class is used to keep track of various models, initial conditions and saved output; and
 1. a plotter class is used for visualizing the results of model runs.

Internally, the package implements a numerical differential equation solver to solve the system of equations in discrete time-steps and to generate time series output for the state of the system. SIRplus also allows discrete stochastic flow equations to be created, and the model can be automatically run repeatedly to conduct Monte Carlo simulations and plot the distribution of model outputs.

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
plt = sp.plotter(title='SIR Time Series', ylabel='Population', fontsize=14)
plt.plot(mgr['My run'],'S', color='blue', label = 'S')
plt.plot(mgr['My run'],'I', color='orange', label = 'I')
plt.plot(mgr['My run'],'R', color='green', label = 'R')
plt.plot(mgr['My run'],'S + I + R', color='black', label = 'Total')
```

![image](https://user-images.githubusercontent.com/86741975/125519680-4f964905-8c1b-4565-acf9-fac73ea403f4.png)

Each call to the plotter's *plot* function must specify a run and an output. The run is identified by indexing the run manager with the run label used earlier. The output must be one of the outputs that was specified in the model using *_set_output*. Outputs can be summed together, e.g. *S + I + R* in the last line, above.


# Stochastic model elements

In SIRplus, we can also introduce stochastic model elements and run Monte Carlo simulations. For example, two realistic improvements to the simple SIR model would be to sample the transmission rate from a distribution reflecting the uncertainty in this parameter, and to make the flows stochastic and discrete. We show these changes below in a new model class called *mc_sir*.

```Python
import numpy as np
rng = np.random.default_rng()

class mc_sir(sp.model):
    def _build(self):
        #pools
        self.S = sp.pool(95)
        self.I = sp.pool(5)
        self.R = sp.pool(0)

        #equations
        self.N = sp.equation(lambda: self.S() + self.I() + self.R())
        
        #transmission rate parameters
        self.b_m = sp.parameter(0.2)
        self.b_s = sp.parameter(0.05)
        
        #transmission rate random sample
        self.b = sp.sample(lambda: rng.normal(self.b_m(), self.b_s()))

        #recovery rate parameter
        self.g = sp.parameter(0.1)
        
        #flows
        self.Fsi = sp.flow(lambda: rng.binomial(self.S(), self.b()*self.I()/self.N()), src=self.S, dest=self.I)
        self.Fir = sp.flow(lambda: rng.binomial(self.I(), self.g()), src=self.I, dest=self.R)

        #output
        self._set_output('S','I','R')

m2 = mc_sir()
```

The first lines, above, import numpy and initialize its random number generator (RNG). We now specify the transmission rate with two parameters, a mean value *b_m* and a standard deviation *b_s*. Then we create the transmission rate *b* as a SIRplus *sample*, defined by a lambda function that calls numpy's normal RNG, passing *b_m* and *b_s* as parameters. This will resample the transmission rate from the normal distribution at the start of each model run.

The flow *Fsi* has been updated such that, rather than being a deterministic rate, each susceptible person has a probability of being infected based on the number of infected people in the population and the transmission rate. Therefore, we use the binomial RNG to generate a discrete, random number of new infections that will move from the susceptible population to the infectious population. The flow *Fir* has similiary been updated such that each infected person has a probability of recovering in each time step, again using the binomial RNG to generate a discrete, random number of people to move from the infectious population to the recovered population.

Finally, we instantiate the new model. These modifications produce the same average behavior as the deterministic model, but introduce variability based on the uncertainty in the transmission rate and the randomness of transmission events.

We can now run the model in Monte Carlo mode using the run manager's *run_mc* function, passing the number of replications (reps) in the run settings, and giving the run a new label.

```Python
mgr.run_mc(m2, duration=150, reps=100, label='My run - mc')
```

We can plot the results of a Monte Carlo run using the plotter's *plot_mc* function. The optional *interval* parameter specifies the percentile range from the distribution of outputs to be displayed. An interval of 50 means the middle 50% of the distribution, or the inter-quartile range. An interval of 90 would display the region from the 5th to 95th percentile.

```Python
plt = sp.plotter(title='SIR Time Series - Monte Carlo', ylabel='Population', fontsize=14)
plt.plot_mc(mgr['My run - mc'],'S', color='blue', interval=50, label = 'S')
plt.plot_mc(mgr['My run - mc'],'I', color='orange', interval=50, label = 'I')
plt.plot_mc(mgr['My run - mc'],'R', color='green', interval=50, label = 'R')
plt.plot_mc(mgr['My run - mc'],'S + I + R', color='black', interval=50, label = 'Total')
```

![image](https://user-images.githubusercontent.com/86741975/125520231-accec0af-5762-4c4e-9002-7b179df6089f.png)


# Nested models

SIRplus models support nesting, so any SIRplus model can be used as an element inside another model. For example, if we have two sub-populations with different transmission dynamics and a certain degree of mixing between them, we can create a new model, *mix_sir*, that contains the two instances of of the *mc_sir* model defiend previously.

```Python
class mix_sir(sp.model):
    def _build(self):

        #sub models
        self.GrpA = mc_sir()
        self.GrpB = mc_sir()

        #transmission parameter between students and instructors
        self.b_mix = sp.parameter()
        
        #cross-infection flows
        self.Fsi_GrpA = sp.flow(lambda: rng.binomial(self.GrpA.S(), self.b_mix()*self.GrpB.I()/self.GrpB.N()), src=self.GrpA.S, dest=self.GrpA.I)
        self.Fsi_GrpB = sp.flow(lambda: rng.binomial(self.GrpB.S(), self.b_mix()*self.GrpA.I()/self.GrpA.N()), src=self.GrpB.S, dest=self.GrpB.I)
        
        #output
        self._set_output('GrpA','GrpB')

m3 = mix_sir()
```

In the code above, the two sub-populations, *GrpA* and *GrpB*, are both defined as instances of the *mc_sir* model. Each group behaves internally as before acording to its parameters and initial conditions, but we introduce the possibility of cross-infection between these groups. The cross-infections occur with a different transmission rate, *b_mix* defined as a parameter in the *mix_sir* model. The cross-infection flows result in new infections within each group caused by the infectious population in the other group.

While GrpA and GrpB are the same model, we will supply them with different parameter values and initial conditions. Previously, we specified these values while defining the model, but it is usually preferable to separate model inputs from the model itself. Therefore, we can supply the inputs for the model at run-time using a dictionary. For the above nested model, the dictionary would look something like the following.

```Python
init_GrpA = {'S':95, 'I':5, 'R':0, 'b_m':0.2, 'b_s':0.5, 'g':0.1}
init_GrpB = {'S':29, 'I':1, 'R':0, 'b_m':0.3, 'b_s':0.7, 'g':0.1}
init_mix_sir = {'b_mix':0.1, 'GrpA':init_GrpA, 'GrpB':init_GrpB, 'reps':100, 'end':150}
```

The dictionary keys are the names of the model elements, and the values are used to initialize the element. 






