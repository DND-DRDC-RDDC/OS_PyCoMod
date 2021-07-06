import numpy as np
import datetime

#building block class for model elements
#handles the initial value, current value, and history of values for the element
class building_block:

  def __init__(self, value=1):

    if isinstance(value, list):
      value = np.array(value)

    self.init_value = value
    self.value = value
    self.value_hist = [value] # history of values

  def reset(self, value=None):
    
    if value is not None:  
      if isinstance(value, list):
        value = np.array(value)
      self.init_value = value

    self.value = self.init_value
    self.value_hist = [self.init_value]

  def save_hist(self, passno=1):
    #if first pass, append the current value to value_hist
    if passno == 1:
      self.value_hist.append(self.value)
    #if this is a second or subsequent pass, set last value in value hist (rather than appending)
    else:
      self.value_hist[-1] = self.value

  #calling the building block returns its most recent value 
  #optional idx parameter used to return past values, e.g. Block(-2) returns value from two timesteps ago
  def __call__(self, idx=-1):
    if idx < 0:
      try:
        return self.value_hist[idx]
      except IndexError:
        return self.init_value
    else:
      raise Exception("Index must be negative to reference past value. Can't reference present or future value.")

  #get the time series data for this element as a numpy array
  def get_hist(self):
    return np.array(self.value_hist)


#sim time
class sim_time(building_block):
  def __init__(self, value=0):
    super().__init__(value)

  def reset(self):
    super().reset()

  def init_cond(self, value):
    super().reset(value)

  def update(self, dt):
    self.value = self.value + dt

#sim time dates
class sim_date(building_block):
  def __init__(self, start_date=None, unit=None):
    
    if start_date is None:
      start_date = np.datetime64('today')
    else:
      start_date = np.datetime64(start_date) 
    
    if unit is None:
      unit = np.timedelta64(datetime.timedelta(days=1))
    else:
      unit = np.timedelta64(unit)
    
    super().__init__(start_date)
    self.unit = unit

  def reset(self):
    super().reset()

  def init_cond(self, start_date):
    super().reset(np.datetime64(start_date))

  def update(self, dt):
    self.value = self.value + self.unit*dt

#class for arbitrary run info
class run_info(building_block):
  #constructor
  def __init__(self, value=1):
    super().__init__(value)

  def reset(self):
    super().reset()

  #parameters accept an initial condition
  def init_cond(self, value):
      super().reset(value)


#class representing a pool of people, e.g. the S, I and R in SIR models
class pool(building_block):
  #constructor
  def __init__(self, value=1):
    super().__init__(value)
    self.delta = 0

  #reset
  def reset(self):
    super().reset()
    self.delta = 0

  #pools accept an initial condition
  def init_cond(self, value):
    super().reset(value)
    self.delta = 0

  #reset flows
  def reset_flows(self):
      self.delta = 0

  #add a flow volume to the pool
  def add_flow(self, volume):
    self.delta += volume

  #update the value of the pool based on flows affecting the pool
  def update(self):
    self.value = self.value + self.delta  #be careful: numpy arrays treat += as self-modifying

    #prevent negative values for pool (this needs more thought)
    self.value = np.maximum(self.value, 0)

    self.reset_flows()




#class representing a flow between pools where the rate is a function of other values in the model
#if the flow equation defines a volume (as in discrete flows), the volume parameter is set to true
class flow(building_block):
  #constructor
  def __init__(self, rate_func=lambda:1, src=None, dest=None, priority=False, init=False):
    super().__init__(rate_func())
    self.rate_func = rate_func  #function defining the flow
    self.src = src
    self.dest = dest
    self.priority = priority
    self.init = init
    
  #reset rate values
  def reset(self):
    super().reset(self.rate_func())

  #update the flow
  def update(self, dt):
    self.value = self.rate_func()*dt
    if self.init:
      self.value = self.value * 0
    
  #add flows to the src and dest pools
  def add_flows(self):
    if self.src is not None:
      self.src.add_flow(-self.value)

    if self.dest is not None:
      self.dest.add_flow(self.value)


#class representing a model parameter that can change over time
#IDEA: if parameters can optionally accept a function, this can be called to set the parameter value which would accomplish what random samples do 
class parameter(building_block):
  #constructor
  def __init__(self, value=1):
    super().__init__(value)

  def reset(self):
    super().reset()

  #parameters accept an initial condition
  def init_cond(self, value):
      super().reset(value)



#class representing a constant that is randomly sampled from a distribution at the start of the simulation
class sample(building_block):
  #constructor
  def __init__(self, sample_func=lambda:1):
    super().__init__(sample_func())
    self.sample_func = sample_func

  def reset(self):
    super().reset(self.sample_func())



#class representing an intermediate equation, e.g. N = S+E+I+R, that can be used in flow equations
class equation(building_block):
  #constructor
  def __init__(self, eq_func=lambda:1):
    super().__init__(eq_func())
    self.eq_func = eq_func

  def reset(self):
    super().reset(self.eq_func())

  def update(self, t, dt):
    self.value = self.eq_func()

class step(equation):
  def __init__(self, values, times):
    
    #define the step function
    def eq_func(t=0):
      idx = len([x for x in times if x <= t]) - 1
      if idx < 0:
        return values[0]
      else:
        return values[idx]
      
    super().__init__(eq_func)
    
  def update(self, t, dt):
    self.value = self.eq_func(t)
    
    
class impulse(equation):
  def __init__(self, values, times):
    
    #define the impulse function
    def eq_func(t=0, dt=1):
      
      #impulse times x where t-dt < x <= t
      y = [1 if x > t-dt and x <= t else 0 for x in times]
      
      #return sum of impulse values that fall within the t-dt and t
      return sum(i*j for i,j in zip(values,y))
      
    super().__init__(eq_func)
    
  def update(self, t, dt):
    self.value = self.eq_func(t, dt)
    

