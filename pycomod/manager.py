import datetime
import pandas as pd

#function to read init from excel file
def read_excel_init(file, sheet=None):
    #if file is a string (first call), read the file
    if type(file) is str:
        file = pd.read_excel(file, None)

    #if sheet is None (first call), get the first sheet, else get the specified sheet
    if sheet is None:
        df = list(file.values())[0]
    else:
        df = file[sheet]

    init = {}
    for c in df.columns:
        #get raw column as list, removing nans
        v = [x for x in df[c] if not pd.isna(x)]

        #if it's the output tracking list
        if c == '_out':
            init[c] = v

        #if a single value
        elif len(v) == 1:
            #try loading a sheet with that name
            try:
                init[c] = read_excel_init(file, v[0])
            except KeyError:
                init[c] = v[0]

        #if a column of values, save the list
        else:
            init[c] = v

    return init


#class for running models and saving results
class run_manager:
    def __init__(self):
        self.runs = {}

    def clear_runs(self):
        self.runs = {}

    def __getitem__(self, key):
        return self.runs[key]

    #run a model using init (initial conditions)
    def _run(self, model, init=None, duration=None, label=None, dt=None, start_date=None, start_time=None, reps=None):

        #if init is a string, assume it is an excel file and try to read it
        if type(init) == str:
            init = read_excel_init(init)

        #run info
        model_type = str(type(model)).split('.')[1][:-2]
        init_hash = hash(str(init))
        timestamp = str(datetime.datetime.now()) #running in cloud so may not match local time

        #create default label is label is None
        if label is None:
            label = 'Run model <%s> init hash <%s> at <%s>' % (model_type, init_hash, timestamp)

        #create run entry
        run_data = {}
        run_data['model'] = model_type
        run_data['init_hash'] = init_hash
        run_data['timestamp'] = timestamp

        #get reps from param or from init
        if reps is None:
            try:
                reps = init['_reps']
            except KeyError:
                reps = 100

        #ensure reps is an int
        reps = int(reps)

        #initialize and run the model
        if reps == 1:
            model._run(duration, dt, start_time, start_date, init)
            run_data['output'] = model._output
        else:
            model._run_mc(reps, duration, dt, start_time, start_date, init)
            run_data['output_mc'] = model._output_mc

        run_data['reps'] = reps
        run_data['x_times'] = model._t.value_hist
        run_data['x_dates'] = model._date.value_hist

        self.runs[label] = run_data


    #execute a run on model once using init (initial conditions)
    def run(self, model, init=None, duration=None, label=None, dt=None, start_date=None, start_time=None):
        self._run(model, init, duration, label, dt, start_date, start_time, reps=1)


    #execute a monte-carlo run on model using init (initial conditions)
    def run_mc(self, model, init=None, duration=None, label=None, dt=None, start_date=None, start_time=None, reps=None):
        self._run(model, init, duration, label, dt, start_date, start_time, reps)
