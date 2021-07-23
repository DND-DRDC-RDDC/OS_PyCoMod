# OS_SIRplus

## Introduction

SIRplus is a Python package for building and running Susceptible-Infectious-Recovered (SIR) and similar compartment models derived from systems of differential equations.

The project is being released under Open Science (OS), an initiative of the Government of Canada to make the research products of federal scientists open to the public. SIRplus was developed by scientists in the Centre for Operational Research and Analysis (CORA) within Defence Research and Development Canada (DRDC) in order to model the spread of COVID-19 in specific populations of interest to the Canadian Armed Forces.

The primary developers and contributors to this work are:
 - Mr. Stephen Okazawa
 - Ms. Josée van den Hoogen
 - Dr. Steve Guillouzic

SIRplus is composed of a set of Python classes:
 - model element classes (pools, flows, parameters, equations, and random samples) are used to define the behavior of the system;
 - a model class is used to contain these elements as well as nested sub-models;
 - a run manager class is used to keep track of various models, initial conditions, and saved output; and
 - a plotter class is used for visualizing the results of model runs.

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

<img src="https://render.githubusercontent.com/render/math?math=F_{si}=bS\frac{I}{N}">

where *b* is the transmission rate and *N* is the total population, <img src="https://render.githubusercontent.com/render/math?math=S%2BI%2BR">.

The flow of individuals from I to R is given the rate

<img src="https://render.githubusercontent.com/render/math?math=F_{ir}=gI">

where *g* is the recovery rate.

This produces the following system of differential equations:

<img src="https://render.githubusercontent.com/render/math?math=\frac{dS}{dt}=-bS\frac{I}{N}">

<img src="https://render.githubusercontent.com/render/math?math=\frac{dI}{dt}=bS\frac{I}{N}-gI">

<img src="https://render.githubusercontent.com/render/math?math=\frac{dR}{dt}=gI">

Given a population of size 100, where 5 individuals are infected (I) and the remaining 95 individuals are susceptible (S), we can model this system in SIRplus with the following code:

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
    self.Fsi = sp.flow(lambda: self.b()*self.S()*self.I()/self.N(), src=self.S, dest=self.I)
    self.Fir = sp.flow(lambda: self.g()*self.I(), src=self.I, dest=self.R)
    
    #output
    self._set_output('S', 'I', 'R')
```

The first two lines begin the definition of a custom class inheriting properties from the SIRplus *model* class and overrides the model's *_build* function to define the elements of this simple SIR model. In this case, we create the three population compartments (S,I,R) using sirplus *pools* and specify the initial value of each pool. We define the value *N* (the total population) as a sirplus *equation*. Equations are defined by a function referencing other model elements; lamda functions are syntactically compact for this purpose. To obtain the value of a model element, we call the object by adding open and close parentheses *()*; for example, the current number of susceptible individuals is obtained by *self.S()*. Using sirplus *parameters* we create and specify values for the model's parameters: transmission rate, *b*, and recovery rate, *g*. Next, we define the movement between the compartments using sirplus *flows*; flows are defined by a function, similar to the equation class. In this case, the flow functions correspond to the rate equations, *Fsi* and *Fir*, defined above. Flows must also specify a source pool and a destination pool. Note that when specifying source and destination pools, we reference the pool object itself rather than calling it (e.g. *src=self.S*, not *src=self.S()*). A final step in specifying the model is to let SIRplus know which outputs we want to capture for analysis. This is done by calling the model's *_set_output* function and providing the names of the model elements that we want to track.

Having defined the *simple_sir* model class, we can now create an instance of it and call it *m*.

```Python
m = simple_sir()
```

We use another SIRplus object called a *run_manager* to run it. The run manager keeps track of multiple models, run settings and output so that batches of runs can be automated. First we create an instance of the run manager.

```Python
mgr = sp.run_manager()
```

Now we can tell the run manager to run the *simple_sir* model. We can supply run settings (such as the duration in this example), and we must provide a label as a key to access the run results later.

```Python
mgr.run(m, duration=150, label='My run')
```

Finally, we can plot the results of the run using the SIRplus *plotter*.  First we create an instance of the plotter, which creates a Matplotlib Figure, and then we can plot outputs from the run on the figure axes.

```Python
plt = sp.plotter(title='SIR Time Series', ylabel='Population', fontsize=14)
plt.plot(mgr['My run'],'S', color='blue', label = 'S')
plt.plot(mgr['My run'],'I', color='orange', label = 'I')
plt.plot(mgr['My run'],'R', color='green', label = 'R')
plt.plot(mgr['My run'],'S + I + R', color='black', label = 'Total')
```

![image](https://user-images.githubusercontent.com/86741975/125519680-4f964905-8c1b-4565-acf9-fac73ea403f4.png)

Each call to the plotter's *plot* function must specify a run and an output. The run is identified by indexing the run manager with the label that we specified when we ran the model. The output must be one of the outputs that was specified in the model using *_set_output*. Outputs can be summed together in a plot, e.g. *S + I + R* in the last line, above.

Note that the examples that follow are meant to provide simple demonstrations of the features of SIRplus; they are not necessarily appropriate models for real situations.

# Stochastic model elements

In SIRplus, we can also introduce stochastic model elements and run Monte Carlo simulations. For example, two improvements to the simple SIR model would be to sample the transmission rate from a distribution reflecting the uncertainty in this parameter, and to make the flows stochastic and discrete. We show these changes below in a new model class called *mc_sir*.

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

The flow *Fsi* has been updated such that, rather than being a deterministic rate, each susceptible person has a probability of being infected based on the number of infected people in the population and the transmission rate. Therefore, we use the binomial RNG to generate a discrete, random number of new infections that will move from the susceptible population to the infectious population. The flow *Fir* has similarly been updated such that each infected person has a probability of recovering in each time step, again using the binomial RNG to generate a discrete, random number of people to move from the infectious population to the recovered population.

Finally, we create an instance of the new model. These modifications produce the same average behavior as the deterministic model, but introduce variability based on the uncertainty in the transmission rate and the randomness of transmission events.

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


# Nested models and model initialization

SIRplus models support nesting, so any SIRplus model can be used as an element inside another model. For example, if we have two sub-populations with different transmission dynamics and a certain degree of mixing between them, we can create a new model, *mix_sir*, that contains two instances of the *mc_sir* model defined previously.

```Python
class mix_sir(sp.model):
    def _build(self):

        #sub models
        self.GrpA = mc_sir()
        self.GrpB = mc_sir()

        #transmission parameter between groups
        self.b_mix = sp.parameter()
        
        #cross-infection flows
        self.Fsi_GrpA = sp.flow(lambda: rng.binomial(self.GrpA.S(), self.b_mix()*self.GrpB.I()/self.GrpB.N()), src=self.GrpA.S, dest=self.GrpA.I)
        self.Fsi_GrpB = sp.flow(lambda: rng.binomial(self.GrpB.S(), self.b_mix()*self.GrpA.I()/self.GrpA.N()), src=self.GrpB.S, dest=self.GrpB.I)
        
        #output
        self._set_output('GrpA','GrpB')

m3 = mix_sir()
```

In the code above, the two sub-populations, *GrpA* and *GrpB*, are both defined as instances of the *mc_sir* model. Each group behaves internally as before according to its parameters and initial conditions, but we introduce the possibility of cross-infection between these groups. The cross-infections occur with a different transmission rate, *b_mix*, defined as a parameter in the *mix_sir* model. The cross-infection flows result in new infections within each group caused by the infectious population in the other group. Note that in order to save the output from a sub-model, the sub-model must be listed in the parent model's output list.

While GrpA and GrpB are the same model, we will supply them with different parameter values and initial conditions. Previously, we specified these values while defining the model, but it is usually preferable to separate model inputs from the model itself. Therefore, we can supply the inputs for the model at run-time using a dictionary. For the *mix_sir* model, above, the initialization dictionary would look something like *init_mix* below.

```Python
init_GrpA = {'S':95, 'I':5, 'R':0, 'b_m':0.2, 'b_s':0.05, 'g':0.1}
init_GrpB = {'S':30, 'I':0, 'R':0, 'b_m':0.3, 'b_s':0.05, 'g':0.1}
init_mix = {'b_mix':0.05, 'GrpA':init_GrpA, 'GrpB':init_GrpB, '_reps':100, '_end':150}
```

The dictionary keys are the names of the model elements, and the dictionary values are used to initialize the element. The only model elements that accept input are pools, parameters and sub-models. The entry value for a pool is the initial condition for the pool. The entry value for a parameter is the parameter's value which is constant. To initialize a sub-model, such as *GrpA* above, the entry value is another dictionary designed to initialize the sub-model, which is *init_GrpA* in this case. Hense, nested models are initialized with equivalently nested dictionaries. In this example, GrpA is given the same initialization values as before while GrpB is a smaller population with a higher mean transmission rate but with no initial infections.

The top-level initialization dictionary, *init_mix* in this case, can also contain some special entries to control the run. Here, we specify the number of replications with a *_reps* entry and the run duration with an *_end* entry. These special keys are prefixed with an underscore. This allows the entire model setup to be controlled from the initialization dictionary.

We can then perform a run using the dictionary to initialize the model.

```Python
mgr.run_mc(m3, init=init_mix, label='My run - mix')
```

And we can then plot what happens to GrpA.

```Python
plt = sp.plotter(title='SIR Time Series - Monte Carlo - GrpA', ylabel='Population', fontsize=14)
plt.plot_mc(mgr['My run - mix'],'GrpA.S', color='blue', interval=50, label = 'S')
plt.plot_mc(mgr['My run - mix'],'GrpA.I', color='orange', interval=50, label = 'I')
plt.plot_mc(mgr['My run - mix'],'GrpA.R', color='green', interval=50, label = 'R')
plt.plot_mc(mgr['My run - mix'],'GrpA.S + GrpA.I + GrpA.R', color='black', interval=50, label = 'Total')
```

![image](https://user-images.githubusercontent.com/86741975/125534046-615c52ce-7740-432c-ac19-3d233a9dda32.png)

And GrpB.

```Python
plt = sp.plotter(title='SIR Time Series - Monte Carlo - GrpB', ylabel='Population', fontsize=14)
plt.plot_mc(mgr['My run - mix'],'GrpB.S', color='blue', interval=50, label = 'S')
plt.plot_mc(mgr['My run - mix'],'GrpB.I', color='orange', interval=50, label = 'I')
plt.plot_mc(mgr['My run - mix'],'GrpB.R', color='green', interval=50, label = 'R')
plt.plot_mc(mgr['My run - mix'],'GrpB.S + GrpB.I + GrpB.R', color='black', interval=50, label = 'Total')
```

![image](https://user-images.githubusercontent.com/86741975/125534082-e84971f1-3436-4dda-b431-31ae293f42ba.png)

Note in the above code that to specify the output we want to plot in a nested model, we use dot-notation to navigate the sub-models. E.g. *GrpB.S* plots the susceptible population within GrpB.

# Initialization files

Initialization dictionaries are useful when we want to set up the model in Python code, but it is often practical to contain the initialization data in a file. This allows different model setups to be saved and edited by hand. For this purpose, SIRplus models can also be initialized from an Excel file. The Excel file template to initialize a particular model can be generated by the model itself by calling *_write_excel_init* and providing a file name.

```Python
m3._write_excel_init('init_mix.xlsx')
```

In Google Colab, the initialization file will be written to session storage and can be downloaded. In a local Python environment, the file is written to local storage.

The Excel initialization file is structured in a similar way to the initialization dictionary. The inputs for the model and each sub-model are contained in individual tabs. In this case, there are three tabs.

![image](https://user-images.githubusercontent.com/86741975/126229677-360af357-0a0b-4984-beef-59d634188f1b.png)

The first tab is always called *init* and it contains the top-level initialization inputs which are *GrpA*, *GrpB*, and *b_mix*.

![image](https://user-images.githubusercontent.com/86741975/126227350-953feb05-2c2c-4f55-a939-78d914ad0bbe.png)

We can edit the value for the *b_mix* parameter here.

The *init* tab also contains the special run control entries which are:
 - *_t* - the initial simulation time (usually 0)
 - *_date* - the initial simulation date
 - *_dt* - the simulation time step
 - *_end* - the simulation end time
 - *_reps* - the number of replications for Monte Carlo runs

Because *GrpA* and *GrpB* are sub-models, the value under these labels is the name of the tab that contains the initialization data for that sub-model. So under *GrpA*, the value is *init.GrpA* which is the name of the second tab. It should usually not be necessary to change the sheet name entry under a sub-model. In the *init.GrpA* tab we find the inputs for the elements of the GrpA sub-model: *S*, *I*, *R*, *b_m*, *b_s*, and *g*.

![image](https://user-images.githubusercontent.com/86741975/126227829-7080c6b4-a58c-475d-b9ae-dc8058473f00.png)

The same applies to the *GrpB* sub-model. Each tab also contains an *_out* entry which is used to list the outputs for the model or sub-model. This has the same function as calling *_set_output* within the model definition. Recall that the outputs of a sub-model will only be saved if the parent model includes the sub-model in its output list. 

We can edit the values in the Excel file, for example, by changing b_mix to 0.025 (cutting the transmission rate between the two populations in half) and then save it.

In Google Colab, we then have to upload the edited file to session storage.

Now we can run the model using the Excel file to initialize it.

```Python
mgr.run_mc(m3, init='init_mix.xlsx', label='My run - mix - xls')
```

Viewing the run output is the same as before.


# Dynamic model parameters
It is often necessary to adjust model parameters over time. In general this can be accomplished using SIRplus equations. For example, we might want to modify the *simple_SIR* model to make the transmission rate decay over time, reflecting increasing adherence to public health measures. So we could replace the parameter *b* with an equation implementing an exponential decay.

```Python
class mod_sir(sp.model):
  def _build(self):
    #pools
    self.S = sp.pool(95)
    self.I = sp.pool(5)
    self.R = sp.pool(0)
    
    #equations
    self.N = sp.equation(lambda: self.S() + self.I() + self.R())

    #parameters
    self.b = sp.equation(lambda: 0.2*(0.98)**self._t())
    self.g = sp.parameter(0.1)
    
    #flows
    self.Fsi = sp.flow(lambda: self.b()*self.I()*self.S()/self.N(), src=self.S, dest=self.I)
    self.Fir = sp.flow(lambda: self.g()*self.I(), src=self.I, dest=self.R)
    
    #output
    self._set_output('S', 'I', 'R', 'b')
    
m4 = mod_sir()
```

Note that the current simulation time can be accessed by calling the special variable *self._t*. We can view the modified transmission rate over time by adding *b* to the list of outputs, running the model and then plotting it.

```Python
mgr.run(m4, duration=150, label='Mod SIR')

plt = sp.plotter(title='Dynamic transmission rate', ylabel='Value', fontsize=14)
plt.plot(mgr['Mod SIR'],'b', color='blue', label = 'Transmission rate')
```

![image](https://user-images.githubusercontent.com/86741975/126204950-020d616b-22a4-45b7-94fd-88c2fcbd1108.png)


Sometimes we want a parameter to change to specific values at specific times, in other words, a step function. It is possible to implement a step function as a SIRplus equation, but this is not trivial. As this is is a common requirement in modelling and simulation, SIRplus provides a built-in equation sub-class called *step*. For example, we can change the *mod_sir* model such that the transmission rate increases and decreases at certain times, reflecting specific measures coming into and out of force.

```Python
self.b = sp.step([0.2, 0.13, 0.2], [0, 7, 21])
```

When initializing the SIRplus *step* object, we provide a list of values and a corresponding list of times. In this case, the transmission rate is initially 0.2 at time 0, it then reduces to to 0.13 on day 7 for a period of two weeks, after which it returns to 0.2. Note that the default time unit in SIRplus is 1 day.

![image](https://user-images.githubusercontent.com/86741975/126210132-60da1f39-f562-494c-8282-a25a3783157e.png)

In the above examples, the numerical constants used to define *b* could be replaced with SIRplus parameters which would register them as model inputs allowing them to be adjusted via an initialization dictionary or initialization file. This is the advantage of using parameters rather than literals in a model.

In the case of the *step* function, we need two vectors, and SIRplus parameters support vector inputs. So we can create a parameter *b_v* for the values of the transmission rate, and a parameter *b_t* for the times at which they will be applied.

```Python
self.b_v = sp.parameter([0.2, 0.13, 0.2])
self.b_t = sp.parameter([0, 7, 21])
self.b = sp.step(self.b_v(), self.b_t())
```
The initialization dictionary for this model would then specify lists for the values of *b_v* and *b_t*.

```Python
init_mod = {'S':95, 'I':5, 'R':0, 'b_v':[0.2, 0.13, 0.2], 'b_t':[0, 7, 21], 'g':0.1}
```

If we create an Excel initialization file for this model, we will see two vector inputs for the parameters *b_v* and *b_t*.

![image](https://user-images.githubusercontent.com/86741975/126209560-0027f6d4-5b18-4a6f-bd08-062a64975d36.png)

Whichever method is used, we can now edit the timing and magnitude of changes to the transmission rate. The size of the vector is not restricted to the initial dimension of three in this example. More values and times can be added so long as there is always a corresponding time for each value.

The SIRplus *impluse* is another type of dynamic value similar to *step*. Impulse generates specified values at specified times, but only at those times. In other words the impulse value is held only for the timestep that contains the impulse time, otherwise it returns 0 or an optional default value. For example, the transmission rate in our model could be 0.2 under normal circumstances, but on certain dates there may be events that are expected to result in elevated transmission.

```Python
self.b = sp.impulse([0.5, 0.5, 0.5], [10, 25, 45], 0.2)
```

When initializing a SIRplus *impulse* object, we provide a list of impulse values, a list of impulse times, and an optional default value. In this case, it produces an elevated transmission rate of 0.5 on days 10, 25 and 45, but it otherwise produces the nominal rate of 0.2.

![image](https://user-images.githubusercontent.com/86741975/126553442-feb00813-bec3-44f7-a5f2-fbbbef562e5a.png)

The same approach as described above can be used to set these values using an initialization dictionary or Excel file.


# Initial flows

In some cases, it may be useful to incorporate flows into establishing the initial state of the system. For example, we may not know that there are exactly 5 initial infections in the population, as in the preceding examples. Instead, we may only know that there is a 5% chance that any given person is infected, based on some larger population statistics. To model this situation, we can place the entire population in the S compartment, and use a stochastic initial flow to move a random number of them to the infectious compartment based on the 5% probability.

```Python
class mc_sir2(sp.model):
    def _build(self):
        #pools
        self.S = sp.pool(100)
        self.I = sp.pool(0)
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
        
        #initial flow
        self.Pi = sp.parameter(0.05)
        self.Fsi_init = sp.flow(lambda: rng.binomial(self.S(), self.Pi()), src=self.S, dest=self.I, init=True)

        #output
        self._set_output('S','I','R')

m5 = mc_sir2()
```

In the above code, note that the S pool is initialized to contain the whole population, and I and R are empty. Toward the end of the model definition, we have added a parameter *Pi*, for the 5% probability of initial infection, and the initial flow *Fsi_init*. This flow uses a binomial RNG to move a random number of individuals from S to I using the probability *Pi*. To flag this flow as an initial flow, we set the optional *init* parameter to *True*. This flow will now only be executed once at the start of each run.

If we run this model, we can see that the initial state of the system is now uncertain, and there is more variability in the outcome compared to the first *mc_sir* model.

```Python
mgr.run_mc(m5, duration=150, reps=100, label='My run - mc2')

plt = sp.plotter(title='SIR Time Series - Monte Carlo', ylabel='Population', fontsize=14)
plt.plot_mc(mgr['My run - mc2'],'S', color='blue', interval=50, label = 'S')
plt.plot_mc(mgr['My run - mc2'],'I', color='orange', interval=50, label = 'I')
plt.plot_mc(mgr['My run - mc2'],'R', color='green', interval=50, label = 'R')
plt.plot_mc(mgr['My run - mc2'],'S + I + R', color='black', interval=50, label = 'Total')
```

![image](https://user-images.githubusercontent.com/86741975/126559095-829eb933-08dc-442d-8e4e-60278e04e676.png)


<!--
# Vectorization


# Execution order


# Common pitfalls

In some cases, compartment models can experience unexpected errors caused by multiple flows exiting the same pool, especially if one or more of the flows is relatively large. For example, imagine that a certain number of people will be vaccinated on certain date. A very simple way to model this is to add a new flow that moves the specified number of people from S to R on that date. However, it is possible that the sum of the vaccination flow (from S to R) and the regular infection flow (from S to I) will exceed the total population in the susceptible pool. This will cause the S pool to have a negative value, and the total population in I and R will exceed the initial total population in the model.

Multinomial distribution as alternate fix.

-->










