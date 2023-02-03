import datetime

import pandas as pd


# Function to read init from excel file
def read_excel_init(file, sheet=None):
    # If file is a string (first call), read the file
    if type(file) is str:
        file = pd.read_excel(file, None)

    # If sheet is None (first call), get the first sheet, else get the
    # specified sheet
    if sheet is None:
        df = list(file.values())[0]
    else:
        df = file[sheet]

    init = {}
    for c in df.columns:
        # Get raw column as list, removing nans
        v = [x for x in df[c] if not pd.isna(x)]

        # If it's the output tracking list
        if c == 'out':
            init[c] = v

        # If a single value
        elif len(v) == 1:
            # Try loading a sheet with that name
            try:
                init[c] = read_excel_init(file, v[0])
            except KeyError:
                init[c] = v[0]

        # If a column of values, save the list
        else:
            init[c] = v

    return init



# Function to read init from excel file
def read_excel_init2(file, sheet=None):
    # If file is a string (first call), read the file
    if type(file) is str:
        file = pd.read_excel(file, None)

        
        
    # If sheet is None (first call), get the first sheet, else get the
    # specified sheet
    if sheet is None:
        df_run = file['run']
        df = file['model']
    else:
        df_run = None
        df = file[sheet]

    init = {}
    
    # if first pass, get run params
    if df_run is not None:
        for c in df_run.columns:
            # Get raw column as list, removing nans
            init[c] = df[c][0]
    
    # get model params
    for c in df.columns:
        # Get raw column as list, removing nans
        v = [x for x in df[c] if not pd.isna(x)]

        # If it's the output tracking list
        if c == 'out':
            init[c] = v

        # If a single value
        elif len(v) == 1:
            # Try loading a sheet with that name
            try:
                init[c] = read_excel_init(file, v[0])
            except KeyError:
                init[c] = v[0]

        # If a column of values, save the list
        else:
            init[c] = v

    return init


# Class for running models and saving results
class RunManager:

    def __init__(self):
        self.runs = {}

    def clear_runs(self):
        self.runs = {}

    def __getitem__(self, key):
        return self.runs[key]

    # Run a model using init (initial conditions)
    def _run(self, model, init=None, duration=None, label=None, dt=None,
             start_date=None, start_time=None, reps=None):

        # If init is a string, assume it is an excel file and try to read it
        if type(init) == str:
            init = read_excel_init2(init)

        # Run info
        model_type = str(type(model)).split('.')[1][:-2]
        init_hash = hash(str(init))
        # Running in cloud so may not match local time
        timestamp = str(datetime.datetime.now())

        # Create default label is label is None
        if label is None:
            label = 'Run model <%s> init hash <%s> at <%s>' % (model_type,
                                                               init_hash,
                                                               timestamp)

        # Create run entry
        run_data = {}
        run_data['model'] = model_type
        run_data['init_hash'] = init_hash
        run_data['timestamp'] = timestamp

        # Get reps from param or from init
        if reps is None:
            try:
                reps = init['reps']
            except KeyError:
                reps = 100

        # Ensure reps is an int
        reps = int(reps)

        # Initialize and run the model
        if reps == 1:
            model._run(duration, dt, start_time, start_date, init)
            run_data['output'] = model._output
        else:
            model._run_mc(reps, duration, dt, start_time, start_date, init)
            run_data['output_mc'] = model._output_mc

        run_data['reps'] = reps
        run_data['x_times'] = model.t.value_hist
        run_data['x_dates'] = model.date.value_hist

        self.runs[label] = run_data

    # Execute a run on model once using init (initial conditions)
    def run(self, model, init=None, duration=None, label=None, dt=None,
            start_date=None, start_time=None):
        self._run(model, init, duration, label, dt,
                  start_date, start_time, reps=1)

    # Execute a monte-carlo run on model using init (initial conditions)
    def run_mc(self, model, init=None, duration=None, label=None, dt=None,
               start_date=None, start_time=None, reps=None):
        self._run(model, init, duration, label, dt,
                  start_date, start_time, reps)
