from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .elements import (BuildingBlock, SimTime, SimDate, RunInfo,
                              Pool, Flow, Parameter, Sample, Equation, Step, Impulse)



class event:
    def __init__(self, init=False, delay=0, priority=0):
        self.init = init
        self.delay = delay
        self.priority = priority
        
    def __call__(self, f):
        class wrapper:
            def __init__(self, f):
                #register the event
                self.f = f
                
            def __call__(self):
                self.f()
                
        return wrapper
                
    
        

# Class for building and running the model
class Model(ABC):

    def __init__(self):

        # Time info
        self._t = SimTime()
        self._date = SimDate()
        self._tunit = RunInfo(np.timedelta64(1, 'D'))

        # Run info
        self._dt = RunInfo(1)
        self._end = RunInfo(365)
        self._reps = RunInfo(100)

        # Model elements
        self._parameters = []
        self._samples = []
        self._equations = []
        self._flows = []
        self._pools = []

        # Sub-models
        self._models = []

        # Available
        self._available = {}

        # Output
        self._out = None  # Elements to track for output
        self._output = None  # Output from run
        self._output_mc = None  # Output from mc runs


        # Event queue for discrete events
        self._event_queue = []

        # Setup
        self.build()
        self._register()


    # Read-only properties
    @property
    def t(self):
        return self._t

    @property
    def date(self):
        return self._date
        
    @property
    def tunit(self):
        return self._tunit

    @property
    def dt(self):
        return self._dt

    @property
    def end(self):
        return self._end

    @property
    def reps(self):
        return self._reps

    @property
    def out(self):
        return self._out

    def set_output(self, *args):
        self._out = list(args)

    @abstractmethod
    def build(self):
        # Implemented by sub-class
        pass
        
    def set_available(self, names, output=None ):
        
        # self is not needed in the dict
        del names['self']
        
        for key, value in names.items():
            value.name = key
        
        self._available = names
        self._out = [o.name for o in output]
        
        
    def __getattr__(self, name):
        return self._available[name]
    
    # element creation functions
    def pool(self, value=1, allow_neg=False):
        e = Pool(value, allow_neg)
        self._pools.append(e)
        return e
        
    # def flow(self, rate_func=lambda: 1, src=None, dest=None, discrete=False):
        # e = Flow(rate_func, src, dest, discrete)
        # self._flows.append(e)
        # return e
        
    def flow(self, *args, **kwargs):
        # decorator without parameters or call with flow function but no parameters
        # a flow without src or dest args is useless, but syntactically allowed
        if len(args)==1 and len(kwargs)==0 and callable(args[0]):
            e = Flow(args[0])
            self._flows.append(e)
            return e
            
        # non-decorator call with flow function and optional parameters
        elif len(args)==1 and len(kwargs)>0 and callable(args[0]):
            
            src = None
            dest = None
            discrete = False
            
            if 'src' in kwargs:
                src = kwargs['src']
            if 'dest' in kwargs:
                dest = kwargs['dest']
            if 'discrete' in kwargs:
                discrete = kwargs['discrete']
            
            e = Flow(args[0], src, dest, discrete)
            self._flows.append(e)
            return e
        
        # else assume decorator with params
        else:
            src = None
            dest = None
            discrete = False
            
            if 'src' in kwargs:
                src = kwargs['src']
            if 'dest' in kwargs:
                dest = kwargs['dest']
            if 'discrete' in kwargs:
                discrete = kwargs['discrete']
                
            def inner(rate_func):
                e = Flow(rate_func, src, dest, discrete)
                self._flows.append(e)
                return e
                
            return inner
        
            
            
    def parameter(self, value=1):
        e = Parameter(value)
        self._parameters.append(e)
        return e
        
    def equation(self, eq_func=lambda: 1):
        e = Equation(eq_func)
        self._equations.append(e)
        return e
        
    def step(self, values, times, default=0):
        e = Step(values, times, default)
        self._equations.append(e)
        return e       
     
    def impulse(self, values, times):
        e = Impulse(values, times)
        self._equations.append(e)
        return e

    def submodel(self, m):
        self._models.append(m)
        m._event_queue = self._event_queue
        return m

    def _register(self):
        # Get all attributes that are an instance of BuildingBlock and
        # organize them into lists

        elements = [x for x in self.__dict__.values()
                    if isinstance(x, (BuildingBlock, Model))]

        for e in elements:
            if isinstance(e, Sample):
                self._samples.append(e)
            #elif isinstance(e, Parameter):
            #    self._parameters.append(e)
            #elif isinstance(e, Equation):
            #    self._equations.append(e)
            #elif isinstance(e, Flow):
            #    self._flows.append(e)
            #elif isinstance(e, Pool):
            #    self._pools.append(e)
            #elif isinstance(e, Model):
            #    self._models.append(e)
                #all sub-models share the root event queue
            #    e._event_queue = self._event_queue



    # Set any initial conditions for the model
    def _init_cond(self, init):
        # Recursively apply initial conditions
        for key, value in init.items():
            if key == 'out':
                # Store elements of this model to be tracked for output
                self._out = value
            elif key in ['dt', 't', 'tunit', 'end', 'date', 'reps']:
                # If time and run info, push init to submodels
                self._push_init(key, value)
            else:
                # Set initial condition
                e = getattr(self, key)

                # If it's a model
                if isinstance(e, Model):
                    e._init_cond(value)

                # If it's an element
                else:
                    e.init_cond(value)

    # Set the run and model initial conditions from a dictionary
    def set_init(self, init):
        self._init_cond(init['run'])
        self._init_cond(init['model'])

    # Get the initial condition dict for this model
    def _get_model_init(self):

        self._reset()

        d = {}
        elements = [(k, v) for k, v in self.__dict__.items()
                    if isinstance(v, (Pool, Parameter, Model))]

        for k, v in elements:
            if isinstance(v, Model):
                d[k] = v._get_model_init()
            else:
                d[k] = v()

        return d

    # Get the run init settings
    def _get_run_init(self):
        d = {}

        # Add run settings
        d = {}
        d['t'] = self.t()
        d['date'] = self.date()
        d['tunit'] = self.tunit()
        d['dt'] = self.dt()
        d['end'] = self.end()
        d['reps'] = self.reps()

        return d

    def get_init(self):
        d = {}

        d['run'] = self._get_run_init()
        d['model'] = self._get_model_init()

        return d

    # Get dataframes representing initial conditions for the model
    def _get_init_df(self, d=None, key=None):

        self._reset()

        # If this is the root, create the dict and add run settings
        if d is None:
            d = {}

            # Add run settings
            d['run'] = {}
            d['run']['t'] = [self.t()]
            d['run']['date'] = [self.date()]
            d['run']['tunit'] = [self.tunit()]
            d['run']['dt'] = [self.dt()]
            d['run']['end'] = [self.end()]
            d['run']['reps'] = [self.reps()]

            d['run'] = pd.DataFrame.from_dict(d['run'])

        # If this is the root, set the key to 'model'
        if key is None:
            key = 'model'

        # Create dict
        d[key] = {}

        # Add all elements to the dict
        elements = [(k, v) for k, v in self._available.items()
                    if isinstance(v, (Pool, Parameter, Model))]
        for k, v in elements:
            if isinstance(v, Model):
                next_key = key + '.' + k
                d[key][k] = [next_key]
                v._get_init_df(d, next_key)
            else:
                if type(v()) == np.ndarray:
                    d[key][k] = v()
                else:
                    d[key][k] = [v()]

        # Add output tracking
        if self.out is None:
            d[key]['out'] = [None]
        else:
            d[key]['out'] = self.out

        # Get max num rows
        rows = max([len(x) for x in d[key].values()])

        # Normalize column lengths
        for k in d[key].keys():
            add = rows - len(d[key][k])
            if add > 0:
                d[key][k] = np.append(d[key][k], [None]*add)

        # Convert to dataframe
        d[key] = pd.DataFrame.from_dict(d[key])

        return d

    # Write an excel file containing initial conditions for the model
    def write_excel_init(self, filename=None):
        d = self._get_init_df()

        if filename is None:
            filename = 'init.xlsx'

        with pd.ExcelWriter(filename) as writer:
            for k, v in d.items():
                v.to_excel(writer, sheet_name=k, index=False)

    # Set initial condition and push to submodels
    def _push_init(self, key, value):
        getattr(self, key).init_cond(value)
        for m in self._models:
            m._push_init(key, value)

    # UPDATE FUNCTIONS

    def _add_flows(self):

        # Recurse through sub-models
        for m in self._models:
            m._add_flows()

        # Add flows to pools
        for e in self._flows:
            e.add_flows()

    def _update_pools(self):

        # Recurse through sub-models
        for m in self._models:
            m._update_pools()

        # Update pools (in order)
        for e in self._pools:
            e.update()
            e.save_hist()

    def _update_equations(self):

        # Recurse through sub-models
        for m in self._models:
            m._update_equations()

        # Update equations (in order)
        for e in self._equations:
            e.update(self.t(), self.dt())
            e.save_hist()


    def _update_flows(self):

        # Recurse through sub-models
        for m in self._models:
            m._update_flows()

        # Update flows (order independent)
        for e in self._flows:
            e.update(self.dt())
        for e in self._flows:
            e.save_hist()

    def _update_time(self):

        # Recurse through sub-models
        for m in self._models:
            m._update_time()

        # Update time info
        self.t.update(self.dt())
        self.t.save_hist()

        self.date.update(self.dt(), self.tunit())
        self.date.save_hist()

    # Regular update sequence
    def _update_regular(self):

        self._add_flows()
        self._update_pools()
        self._update_equations()
        self._update_flows()



    # Update pass for all model elements
    def _update(self):

        # Update time
        self._update_time()

        # Update model elements
        self._update_regular()

    def _reset_pools(self):

        # Recurse through sub-models
        for m in self._models:
            m._reset_pools()

        # Reset pools
        for e in self._pools:
            e.reset()

    def _reset_parameters(self):

        # Recurse through sub-models
        for m in self._models:
            m._reset_parameters()

        # Reset parameters
        for e in self._parameters:
            e.reset()

    def _reset_samples(self):

        # Recurse through sub-models
        for m in self._models:
            m._reset_samples()

        # Reset samples
        for e in self._samples:
            e.reset()

    def _reset_equations(self):

        # Recurse through sub-models
        for m in self._models:
            m._reset_equations()

        # Reset samples
        for e in self._equations:
            e.reset()


    def _reset_flows(self):

        # Recurse through sub-models
        for m in self._models:
            m._reset_flows()

        # Reset samples
        for e in self._flows:
            e.reset(self.dt)

    def _reset_time(self):

        # Recurse through sub-models
        for m in self._models:
            m._reset_time()

        self.t.reset()
        self.date.reset()

    def _reset_output(self):
        self._output = None

    def _reset_output_mc(self):
        self._output_mc = None

    # Reset all model elements to initial conditions
    def _reset(self):

        # Reset time
        self._reset_time()
        self._reset_output()

        # Reset all elements
        self._reset_pools()
        self._reset_parameters()
        self._reset_samples()
        self._reset_equations()
        self._reset_flows()


    # Save all output
    def _save_output(self):
        self._output = {}
        for key in self.out:
            e = getattr(self, key)
            #key = e.name

            if isinstance(e, BuildingBlock):
                #self._output[key] = getattr(self, key).get_hist()
                self._output[key] = e.get_hist()
            elif isinstance(e, Model):
                self._output[key] = e._save_output()

        return self._output

    # Do a run
    def _run(self, end=None, dt=None, tunit=None, start_time=None,
             start_date=None, init=None):

        # First apply initial conditions from init dict
        if init is not None:
            self.set_init(init)

        # Override for any of the following run parameters
        if end is not None:
            self._push_init('end', end)

        if dt is not None:
            self._push_init('dt', dt)

        if tunit is not None:
            self._push_init('tunit', tunit)

        if start_time is not None:
            self._push_init('t', start_time)

        if start_date is not None:
            self._push_init('date', start_date)

        # Number of sim steps
        n = int(self.end()/self.dt())

        # Reset after applying initial conditions
        self._reset()

        # For each time step update everything
        for i in range(n):

            # Update model elements
            self._update()

        # Save output
        self._save_output()

    # Create container for mc output based on output from first replication
    def _init_output_mc(self, output):
        output_mc = {}
        for k, v in output.items():
            if not isinstance(v, dict):
                output_mc[k] = np.array([v])
            else:
                output_mc[k] = self._init_output_mc(v)

        return output_mc

    # Append output from subsequent replications to the mc output
    def _append_output_mc(self, output_mc, output):
        for k, v in output.items():
            if not isinstance(v, dict):
                output_mc[k] = np.append(output_mc[k], np.array([v]), axis=0)
            else:
                self._append_output_mc(output_mc[k], v)

    # Save output from MC runs
    def _save_output_mc(self):
        if self._output_mc is None:
            self._output_mc = self._init_output_mc(self._output)
        else:
            self._append_output_mc(self._output_mc, self._output)

    # Monte carlo runs
    def _run_mc(self, reps=None, end=None, dt=None, tunit=None,
                start_time=None, start_date=None, init=None):
        # First apply initial conditions from init dict
        if init is not None:
            self.set_init(init)

        # Override for any of the following run parameters
        if reps is not None:
            self._push_init('reps', reps)

        if end is not None:
            self._push_init('end', end)

        if dt is not None:
            self._push_init('dt', dt)

        if tunit is not None:
            self._push_init('tunit', tunit)

        if start_time is not None:
            self._push_init('t', start_time)

        if start_date is not None:
            self._push_init('date', start_date)

        # Reset mc output
        self._reset_output_mc()

        # Run all reps and save mc output
        for n in range(int(self.reps())):
            self._run()
            self._save_output_mc()
