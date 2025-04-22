import datetime

import numpy as np


# function to fix the 'other' parameter in operator methods
def f(other):
    if isinstance(other, BuildingBlock):
        return other()
    else:
        return other
    

# Building block class for model elements
# Handles the initial value, current value, and history of values for the
# element
class BuildingBlock:

    def __init__(self, value=1):

        if isinstance(value, list):
            value = np.array(value)

        self.init_value = value
        self.value = value
        self.value_hist = [value]  # History of values

    def reset(self, value=None):

        if value is not None:
            if isinstance(value, list):
                value = np.array(value)
            self.init_value = value

        self.value = self.init_value
        self.value_hist = [self.init_value]

    def save_hist(self):
        self.value_hist.append(self.value)


    # Calling the building block returns its most recent value
    # Optional idx parameter used to return past values, e.g. Block(-2) returns
    # value from two timesteps ago
    def __call__(self, idx=-1):
        if idx < 0:
            try:
                return self.value_hist[idx]
            except IndexError:
                return self.init_value
        else:
            raise Exception("Index must be negative to reference past value. "
                            "Can't reference present or future value.")

    # Get the time series data for this element as a numpy array
    def get_hist(self):
        return np.array(self.value_hist)



    # data model methods
    def __float__(self):
        return float(self())
        
    def __int__(self):
        return int(self())
    
    # comparators
    def __lt__(self, other):
        return self() < f(other)
        
    def __le__(self, other):
        return self() <= f(other)
        
    def __eq__(self, other):
        return self() == f(other)
        
    def __ne__(self, other):
        return self() != f(other)
        
    def __gt__(self, other):
        return self() > f(other)
        
    def __ge__(self, other):
        return self() >= f(other)
        
        
    # numeric
    def __add__(self, other):
        return self() + f(other)
        
    def __sub__(self, other):
        return self() - f(other)
        
    def __mul__(self, other):
        return self() * f(other)
        
    def __matmul__(self, other):
        return self() @ f(other)
        
    def __truediv__(self, other):
        return self() / f(other)
        
    def __floordiv__(self, other):
        return self() // f(other)
        
    def __mod__(self, other):
        return self() % f(other)
        
    def __pow__(self, other):
        return self() ** f(other)
        
        
    def __radd__(self, other):
        return f(other) + self()
        
    def __rsub__(self, other):
        return f(other) - self()
        
    def __rmul__(self, other):
        return f(other) * self()
        
    def __rmatmul__(self, other):
        return f(other) @ self()
        
    def __rtruediv__(self, other):
        return f(other) / self()
        
    def __rfloordiv__(self, other):
        return f(other) // self()
        
    def __rmod__(self, other):
        return f(other) % self()
        
    def __rpow__(self, other):
        return f(other) ** self()
        
        
    def __neg__(self):
        return -self()
        
    def __abs__(self):
        return abs(self())
        




# Sim time
class SimTime(BuildingBlock):

    def __init__(self, value=0):
        super().__init__(value)

    def reset(self):
        super().reset()

    def init_cond(self, value):
        super().reset(value)

    def update(self, dt):
        self.value = self.value + dt


# Sim time dates
class SimDate(BuildingBlock):

    def __init__(self, start_date=None):

        if start_date is None:
            start_date = np.datetime64('today')
        else:
            start_date = np.datetime64(start_date)

        super().__init__(start_date)

    def reset(self):
        super().reset()

    def init_cond(self, start_date):
        super().reset(np.datetime64(start_date))

    def update(self, dt, tunit):
        self.value = self.value + dt*tunit


# Class for arbitrary run info
class RunInfo(BuildingBlock):

    # Constructor
    def __init__(self, value=1):
        super().__init__(value)

    def reset(self):
        super().reset()

    # Parameters accept an initial condition
    def init_cond(self, value):
        super().reset(value)


# Class representing a pool of people, e.g. the S, I and R in SIR models
class Pool(BuildingBlock):

    # Constructor
    def __init__(self, value=1):
        super().__init__(value)
        self.delta = 0

    # Reset
    def reset(self):
        super().reset()
        self.delta = 0

    # Pools accept an initial condition
    def init_cond(self, value):
        super().reset(value)
        self.delta = 0

    # Reset flows
    def reset_flows(self):
        self.delta = 0

    # Add a flow volume to the pool
    def add_flow(self, volume):
        self.delta += volume

    # Update the value of the pool based on flows affecting the pool
    def update(self):
        # Be careful: numpy arrays treat += as self-modifying
        self.value = self.value + self.delta

        # Prevent negative values for pool (this needs more thought)
        self.value = np.maximum(self.value, 0)

        self.reset_flows()


# Class representing a flow between pools where the rate is a function of other
# values in the model
# If the flow equation defines a volume (as in discrete flows), the volume
# parameter is set to true
class Flow(BuildingBlock):

    # Constructor
    def __init__(self, rate_func=lambda: 1, src=None, dest=None, discrete=False):
        super().__init__(rate_func())
        self.rate_func = rate_func  # Function defining the flow
        self.src = src
        self.dest = dest
        #self.priority = priority
        #self.init = init
        self.discrete = discrete
        self.rem = 0

    # Reset rate values
    def reset(self, dt):
        self.rem = 0

        v = self.rate_func()*dt + self.rem
        
        if self.discrete:
            v_ = round(v,0)
            self.rem = v - v_
            v = v_
        
        super().reset(v)
        
    # Update the flow
    def update(self, dt):
        
        v = self.rate_func()*dt + self.rem
        
        if self.discrete:
            v_ = round(v,0)
            self.rem = v - v_
            v = v_
        
        self.value = v
        #if self.init:
        #    self.value = self.value * 0

    # Add flows to the src and dest pools
    def add_flows(self):
        if self.src is not None:
            self.src.add_flow(-self.value)

        if self.dest is not None:
            self.dest.add_flow(self.value)


# Class representing a model parameter that can change over time
# IDEA: if parameters can optionally accept a function, this can be called to
# set the parameter value which would accomplish what random samples do
class Parameter(BuildingBlock):

    # Constructor
    def __init__(self, value=1):
        super().__init__(value)

    def reset(self):
        super().reset()

    # Parameters accept an initial condition
    def init_cond(self, value):
        super().reset(value)


# Class representing a constant that is randomly sampled from a distribution at
# the start of the simulation
class Sample(BuildingBlock):

    # Constructor
    def __init__(self, sample_func=lambda: 1):
        super().__init__(sample_func())
        self.sample_func = sample_func

    def reset(self):
        super().reset(self.sample_func())


# Class representing an intermediate equation, e.g. N = S+E+I+R, that can be
# used in flow equations
class Equation(BuildingBlock):

    # Constructor
    def __init__(self, eq_func=lambda: 1):
        super().__init__(eq_func())
        self.eq_func = eq_func

    def reset(self):
        
        v = self.eq_func()
        if isinstance(v, BuildingBlock):
            v = v()
            
        super().reset(v)

    def update(self, t, dt):
        
        v = self.eq_func()
        if isinstance(v, BuildingBlock):
            v = v()
        
        self.value = v


class Step(Equation):

    def __init__(self, values, times, default=0):

        # Define the step function
        def eq_func(t=0):
            
            vals = values
            tims = times
            
            if isinstance(vals, Parameter):
                vals = vals()
                
            if isinstance(tims, Parameter):
                tims = tims()
            
            idx = len([x for x in tims if x <= t]) - 1
            if idx < 0:
                return default
            else:
                return vals[idx]

        super().__init__(eq_func)

    def update(self, t, dt):
        self.value = self.eq_func(t)


class Impulse(Equation):

    def __init__(self, values, times):

        # Define the impulse function
        def eq_func(t=0, dt=1):

            vals = values
            tims = times
            
            if isinstance(vals, Parameter):
                vals = vals()
                
            if isinstance(tims, Parameter):
                tims = tims()

            # Impulse times x where t-dt < x <= t
            y = [1 if x > t-dt and x <= t else 0 for x in tims]

            # If no impulse values fall within t-dt and t, return default
            if 1 not in y:
                return 0
            # Else return sum of impulse values that fall within the t-dt and t
            else:
                return sum(i*j for i, j in zip(vals, y))/dt

        super().__init__(eq_func)

    def update(self, t, dt):
        self.value = self.eq_func(t, dt)
