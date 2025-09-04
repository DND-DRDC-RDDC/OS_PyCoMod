import datetime
import heapq
import math
import numpy as np
from types import GeneratorType



# function to fix the 'other' parameter in operator methods
def f(other):
    if isinstance(other, BuildingBlock):
        return other()
    elif isinstance(other, VirtualBuildingBlock):
        return other._target()
    else:
        return other
    

class PoolDict(dict):

    def _setparent(self, parent):
        self._parent = parent
        self._parent.update_value(len(self))
        self._parent.save_hist()

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        self._parent.update_value(len(self))
        self._parent.save_hist()

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        self._parent.update_value(len(self))
        self._parent.save_hist()

    def pop(self, key, default=None):
        dict.pop(self, key, default)
        self._parent.update_value(len(self))
        self._parent.save_hist()
        
    def popitem(self):
        dict.popitem(self)
        self._parent.update_value(len(self))
        self._parent.save_hist()
        
    def update(self, other):
        dict.update(self, other)
        self._parent.update_value(len(self))
        self._parent.save_hist()
        
    def clear(self):
        dict.clear(self)
        self._parent.update_value(len(self))
        self._parent.save_hist()
        

# Building block class for model elements
# Handles the initial value, current value, and history of values for the
# element
class BuildingBlock:

    def __init__(self, value=1, parent=None):

        if isinstance(value, list):
            value = np.array(value)

        self.init_value = value
        self.value = value
        self.time = 0
        self.value_hist = [value]  # History of values
        self.time_hist = [0]
        
        self.parent = parent
        
        

    def reset(self, value=None):

        if value is not None:
            if isinstance(value, list):
                value = np.array(value)
            self.init_value = value

        self.value = self.init_value
        self.value_hist = [self.init_value]
        self.time_hist = [self.parent.t.init_value]


    def update_value(self, value):
        self.value = value
        self.time = self.parent.t()


    def save_hist(self):
        self.value_hist.append(self.value)
        self.time_hist.append(self.time)


    # pushes a current value on the element including the most recent value in value_hist
    def push_value(self, value):
        self.value = value
        self.value_hist[-1] = value
        self.time = self.parent.t()
        self.time_hist[-1] = self.parent.t()

    # Calling the building block returns its most recent value
    # Optional idx parameter used to return past values, e.g. Block(-2) returns
    # value from two timesteps ago
    def __call__(self):
        return self.value_hist[-1]
        
        # if idx < 0:
            # try:
                # return self.value_hist[idx]
            # except IndexError:
                # return self.init_value
        # else:
            # raise Exception("Index must be negative to reference past value. "
                            # "Can't reference present or future value.")

    # Get the history of values for this element as a numpy array (true DES time)
    def get_hist(self):
        
        h = {}
        h['values'] = np.array(self.value_hist)
        h['times'] = np.array(self.time_hist)
        h['dates'] = np.array([self.parent.date() + t * self.parent.tunit() for t in h['times']])
        
        return h
    
    def __iter__(self):
        return iter(self())

    def __getitem__(self, index):
        return self()[index]
        
    def __len__(self):
        return len(self())


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
        



class VirtualBuildingBlock:
    def __init__(self, cls=None):
        self._cls = cls
        self._target = None


    def connect(self, target):
        
        assert isinstance(target, self._cls)
        
        self._target = target

        
    def __getattr__(self, name):
        return getattr(self._target, name)
        
        
    def __call__(self):
        return self._target()
        
        
    def __iter__(self):
        return iter(self._target())

    def __getitem__(self, index):
        return self._target()[index]
        
    def __len__(self):
        return len(self._target())


    # data model methods
    def __float__(self):
        return float(self._target())
        
    def __int__(self):
        return int(self._target())
    
    # comparators
    def __lt__(self, other):
        return self._target() < f(other)
        
    def __le__(self, other):
        return self._target() <= f(other)
        
    def __eq__(self, other):
        return self._target() == f(other)
        
    def __ne__(self, other):
        return self._target() != f(other)
        
    def __gt__(self, other):
        return self._target() > f(other)
        
    def __ge__(self, other):
        return self._target() >= f(other)
        
        
    # numeric
    def __add__(self, other):
        return self._target() + f(other)
        
    def __sub__(self, other):
        return self._target() - f(other)
        
    def __mul__(self, other):
        return self._target() * f(other)
        
    def __matmul__(self, other):
        return self._target() @ f(other)
        
    def __truediv__(self, other):
        return self._target() / f(other)
        
    def __floordiv__(self, other):
        return self._target() // f(other)
        
    def __mod__(self, other):
        return self._target() % f(other)
        
    def __pow__(self, other):
        return self._target() ** f(other)
        
        
    def __radd__(self, other):
        return f(other) + self._target()
        
    def __rsub__(self, other):
        return f(other) - self._target()
        
    def __rmul__(self, other):
        return f(other) * self._target()
        
    def __rmatmul__(self, other):
        return f(other) @ self._target()
        
    def __rtruediv__(self, other):
        return f(other) / self._target()
        
    def __rfloordiv__(self, other):
        return f(other) // self._target()
        
    def __rmod__(self, other):
        return f(other) % self._target()
        
    def __rpow__(self, other):
        return f(other) ** self._target()
        
        
    def __neg__(self):
        return -self._target()
        
    def __abs__(self):
        return abs(self._target())
        
        
        
        
        
        
        

# Sim time
class SimTime(BuildingBlock):

    def __init__(self, value=0, parent=None):
        super().__init__(value, parent)

    def reset(self):
        super().reset()

    def init_cond(self, value):
        super().reset(value)

    def update(self, t):
        #elf.value = self.value + dt
        
        self.update_value(t)
        
#    def event_update(self, t):
#        self.value = t


# Sim time dates
class SimDate(BuildingBlock):

    def __init__(self, start_date=None, parent=None):

        if start_date is None:
            start_date = np.datetime64('today')
        else:
            start_date = np.datetime64(start_date)

        super().__init__(start_date, parent)

    def reset(self):
        super().reset()

    def init_cond(self, start_date):
        super().reset(np.datetime64(start_date))

    def update(self, dt, tunit):
        self.value = self.value + dt*tunit


# Class for arbitrary run info
class RunInfo(BuildingBlock):

    # Constructor
    def __init__(self, value=1, parent=None):
        super().__init__(value, parent)

    def reset(self):
        super().reset()

    # Parameters accept an initial condition
    def init_cond(self, value):
        super().reset(value)


# Class representing a pool of people, e.g. the S, I and R in SIR models
# pool_type can be "float", "int", "discrete"
class Pool(BuildingBlock):

    # Constructor
    def __init__(self, value=1, allow_neg=False, pool_type="float", parent=None):
        
        self.pool_type = pool_type
        
        if pool_type == "int":
            value = int(value)
            
        elif pool_type == "discrete":
            value = int(value)

        super().__init__(value, parent)
        self.allow_neg = allow_neg
        self.delta = 0
        self.next_uid = 1
        
        if pool_type == "discrete":
            self.members = PoolDict()
            self.members._setparent(self)
            self.members.update(self.create_members(value))
        
    
    def create_members(self, num):
        pref = str(id(self))
        keys = [pref + '.' + str(self.next_uid + i) for i in range(num)]
        values = [{'uid': k} for k in keys]
        
        self.next_uid += num
        
        return dict(zip(keys, values))


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


    # Actions that can be applied to pools in processes
    def add(self, amount):
        v = self.value + amount
        
        if self.parent.t() == 0:
            self.push_value(v)
        else:
            self.update_value(v)
            self.save_hist()

        
    def remove(self, amount):
        v = self.value - amount

        if self.parent.t() == 0:
            self.push_value(v)
        else:
            self.update_value(v)
            self.save_hist()



    # Update the value of the pool based on flows affecting the pool
    def update(self):
        # Be careful: numpy arrays treat += as self-modifying
        #self.value = self.value + self.delta
        #self.time = t
        
        v = self.value + self.delta
        
        # Prevent negative values for pool (this needs more thought)
        if not self.allow_neg:
            v = np.maximum(v, 0)
        
        self.update_value(v)

        self.reset_flows()


# Class representing a flow between pools where the rate is a function of other
# values in the model
# If the flow equation defines a volume (as in discrete flows), the volume
# parameter is set to true
class Flow(BuildingBlock):

    # Constructor
    def __init__(self, rate_func=lambda: 1, src=None, dest=None, discrete=False, parent=None):
        self.rate_func = rate_func  # Function defining the flow
        self.src = src
        self.dest = dest
        
        if src is not None:
            if src.pool_type in ('int', 'discrete'):
                discrete = True
                
        if dest is not None:
            if dest.pool_type in ('int', 'discrete'):
                discrete = True
        
        self.discrete = discrete
        self.rem = 0
        
        
        v = self.rate_func() * parent.dt
        
        if self.discrete:
            v_ = round(v,0)
            self.rem = v - v_
            v = v_
        
        
        super().__init__(v, parent)


    # Reset rate values
    def reset(self):
        self.rem = 0

        v = self.rate_func() * self.parent.dt()
        
        if self.discrete:
            v_ = round(v,0)
            self.rem = v - v_
            v = v_
        
        super().reset(v)
        
    # Update the flow
    def update(self):
        
        v = self.rate_func()*self.parent.dt() + self.rem
        
        if self.discrete:
            v_ = round(v,0)
            self.rem = v - v_
            v = v_
        
        self.update_value(v)


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
    def __init__(self, value=1, parent=None):
        super().__init__(value, parent)

    def reset(self):
        super().reset()

    # Parameters accept an initial condition
    def init_cond(self, value):
        super().reset(value)
        
    # Can be called by processes to set the value of a parameter
    def set(self, value):
        
        if self.parent.t() == 0:
            self.push_value(value)
        else:
            self.update_value(value)
            self.save_hist()


# # Class representing a constant that is randomly sampled from a distribution at
# # the start of the simulation
# class Sample(BuildingBlock):

    # # Constructor
    # def __init__(self, sample_func=lambda: 1):
        # super().__init__(sample_func())
        # self.sample_func = sample_func

    # def reset(self):
        # super().reset(self.sample_func())


# Class representing an intermediate equation, e.g. N = S+E+I+R, that can be
# used in flow equations
class Equation(BuildingBlock):

    # Constructor
    def __init__(self, eq_func=lambda: 1, parent=None):
        v = eq_func()
        if isinstance(v, BuildingBlock):
            v = v()
        super().__init__(v, parent)
        
        self.eq_func = eq_func

    def reset(self):
        
        v = self.eq_func()
        if isinstance(v, BuildingBlock):
            v = v()
            
        super().reset(v)

    def update(self):
        
        v = self.eq_func()
        if isinstance(v, BuildingBlock):
            v = v()
        
        self.update_value(v)


class Step(Equation):

    def __init__(self, values, times, default=0, parent=None):

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

        super().__init__(eq_func, parent)
        

    def update(self):
        self.update_value(self.eq_func(self.parent.t()))


class Impulse(Equation):

    def __init__(self, values, times, parent=None):

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

        super().__init__(eq_func, parent)

    def update(self):
        self.update_value(self.eq_func(self.parent.t(), self.parent.dt()))


class TimeStep:
    def __init__(self, steps=1):
        self.steps = steps

class Delay:
    def __init__(self, delay):
        self.delay = delay

class Time:
    def __init__(self, time):
        self.time = time
        
class Date:
    def __init__(self, date):
        self.date = np.datetime64(date)
        




# individual event on the sim event queue, the routine could be a function or a generator
class Event:
    def __init__(self, routine, time=None, args=(), priority=0, origin=None, parent=None):
        self.routine = routine
        self.time = time
        self.args = args
        self.priority = priority
        self.origin = origin
        
        self.parent = parent
        
        self.finished = False
        
        
    
    def yield_return(self, y):
        
        if isinstance(y, TimeStep):
            self.time = self.parent.t() + self.parent.dt()*y.steps
            self.parent._push_event(self)
        
        elif isinstance(y, Time):
            self.time = y.time
            self.parent._push_event(self)
            
        elif isinstance(y, Delay):
            self.time = self.parent.t() + y.delay
            self.parent._push_event(self)
            
        elif isinstance(y, Date):
            self.time = (y.date - self.parent.date()) / self.parent.tunit() + self.parent.t.init_value
            self.parent._push_event(self)
            
        elif isinstance(y, Event):
            y.time = self.parent.t()
            y.origin = self
            y.run()
        
    
            
    def resume(self, value):
        if not self.finished:
            try:
                y = self.routine.send(value)
                self.yield_return(y)
                
            except StopIteration as e:
                self.finished = True
                if self.origin != None:
                    self.origin.resume(e.value)
            
            

        
    # run 
    def run_gen(self):
        if not self.finished:
            try:
                y = next(self.routine)
                self.yield_return(y)
                
            except StopIteration as e:
                self.finished = True
                if self.origin != None:
                    self.origin.resume(e.value)
        
        

            
    # run when event pops off sim queue
    def run(self):
        # if it's a generator
        if isinstance(self.routine, GeneratorType):
            self.run_gen()
                
        # else assume it is a function
        else:
            
            #run the function
            x = self.routine(*self.args)
            
            #if the function created a generator, run as a generator
            if isinstance(x, GeneratorType):
                self.routine = x
                self.run_gen()
            
            # else it's a simple function
            else:
                if self.origin != None:
                    self.origin.resume(x)
          

    def start(self, start=None):
        if start != None:
            
            
            if isinstance(start, TimeStep):
                self.time = self.parent.t() + self.parent.dt()*start.steps
                self.parent._push_event(self)
            
            elif isinstance(start, Time):
                self.time = start.time
                self.parent._push_event(self)
                
            elif isinstance(start, Delay):
                self.time = self.parent.t() + start.delay
                self.parent._push_event(self)
                
            elif isinstance(start, Date):
                self.time = (start.date - self.parent.date()) / self.parent.tunit() + self.parent.t.init_valu
                self.parent._push_event(self)
                
            elif isinstance(start, Event):
                
                #ev_sub = Event(self.routine, args=self.args, priority=self.priority, parent=self.parent)
                
                def routine():
                    yield start
                    x = yield self
                    return x
                
                ev = Event(routine, time=self.parent.t(), priority=self.priority, parent=self.parent)
                
                ev.run()
        else:
            self.run()
          
            
    # comparators
    def __lt__(self, other):
        if self.time == other.time:
            return self.priority > other.priority
        else:
            return self.time < other.time
        
    def __le__(self, other):
        if self.time == other.time:
            return self.priority >= other.priority
        else:
            return self.time <= other.time
        
    def __eq__(self, other):
        if isinstance(other, Event):
            if self.time == other.time:
                return self.priority == other.priority
            else:
                return self.time == other.time
        else:
            return False
        
    def __ne__(self, other):
        if isinstance(other, Event):
            if self.time == other.time:
                return self.priority != other.priority
            else:
                return self.time != other.time
        else:
            return True
        
    def __gt__(self, other):
        if self.time == other.time:
            return self.priority < other.priority
        else:
            return self.time > other.time
        
    def __ge__(self, other):
        if self.time == other.time:
            return self.priority <= other.priority
        else:
            return self.time >= other.time
            


# Processes are the user created elements that generate events
class Process:
    
    def __init__(self, routine=lambda:1, args=(), start=None, priority=0, parent=None):
        self.routine = routine
        self.args = args
        self.start = start
        self.priority = priority
        
        self.parent = parent
        
        
    # put the event on the queue if a time is specified
    def reset(self):
        if self.start != None:
            
            if isinstance(self.start, TimeStep):
                time = self.parent.t() + self.parent.dt()*self.start.steps
                ev = Event(self.routine, args=self.args, time=time, priority=self.priority, parent=self.parent)
                self.parent._push_event(ev)
                
            elif isinstance(self.start, Time):
                time = self.start.time
                ev = Event(self.routine, args=self.args, time=time, priority=self.priority, parent=self.parent)
                self.parent._push_event(ev)
                
            elif isinstance(self.start, Delay):
                time = self.parent.t() + self.start.delay
                ev = Event(self.routine, args=self.args, time=time, priority=self.priority, parent=self.parent)
                self.parent._push_event(ev)
                
            elif isinstance(self.start, Date):
                time = (self.start.date - self.parent.date()) / self.parent.tunit() + self.parent.t.init_value
                ev = Event(self.routine, args=self.args, time=time, priority=self.priority, parent=self.parent)
                self.parent._push_event(ev)
                
            elif isinstance(self.start, Event):
                
                ev_sub = Event(self.routine, args=self.args, priority=self.priority, parent=self.parent)
                
                def routine():
                    yield self.start
                    x = yield ev_sub
                    return x
                
                ev = Event(routine, time=self.parent.t.init_value, priority=self.priority, parent=self.parent)
                
                self.parent._push_event(ev)
                
            
            
    # calling (used when another process yields to this process) returns an event for immediate execution
    def __call__(self, *args):
        return Event(self.routine, args=args, time=self.parent.t(), priority=self.priority, parent=self.parent)

            
        
        

    