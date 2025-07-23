import numpy as np
import matplotlib.pyplot as mplot
import matplotlib.dates as mdates

months = mdates.MonthLocator()  # For month intervals on plots
months_fmt = mdates.DateFormatter('%b')


class Plotter:

    @classmethod
    def show(cls):
        # Display plots
        mplot.show()

    def __init__(self, figsize=(14, 6), fontsize=12, title=None,
                 xlabel=None, ylabel=None, ylimit=None, xdates = False):
        # Mpl settings
        mplot.rc('font', size=fontsize)
        mplot.rc('figure', figsize=figsize)

        # Create figure
        self.fig, self.ax = mplot.subplots()

        # Grid lines
        self.ax.grid(True, ls=':')

        # Monthly axis tick marks
        #self.ax.xaxis.set_major_locator(months)
        #self.ax.xaxis.set_major_formatter(months_fmt)

        # Title
        if title is not None:
            self.ax.set_title(title)

        # Axis labels
        if xlabel is not None:
            self.ax.set(xlabel=xlabel)
        if ylabel is not None:
            self.ax.set(ylabel=ylabel)

        # Y axis limits
        if ylimit is not None:
            self.ax.set_ylim(*ylimit)
            
        self.xdates = xdates
        





    def plot(self, element, **kwargs):
        # First setup plot if not done already
        if self.fig is None:
            self.setup()
            
 
        try:
            color = kwargs['color']
        except KeyError:
            color = 'steelblue'

        try:
            label = kwargs['label']
        except KeyError:
            label = '.'.join(args)

        try:
            cumsum = kwargs['cumsum']
        except KeyError:
            cumsum = False
            
        try:
            interval = kwargs['interval']
        except KeyError:
            interval = 75
            
        try:
            step = kwargs['step']
        except KeyError:
            step = False
            
        try:
            alpha = kwargs['alpha']
        except KeyError:
            alpha = 1
            
        try:
            linestyle = kwargs['linestyle']
        except KeyError:
            linestyle = '-'
            
            
            
        # if plotting an element from a regular run
        if 'values' in element:
            d = element['values']
            
            if self.xdates:
                x = element['dates']
            else:
                x = element['times']
            
            # If cumulative
            if cumsum:
                d = np.cumsum(d)

            if step:
                self.ax.step(x, d, color=color, label=label, alpha=alpha, linestyle=linestyle, where='post')
            else:
                self.ax.plot(x, d, color=color, label=label, alpha=alpha, linestyle=linestyle)
        
        
        # else plotting an element from a MC run
        else:
            d = element['mc_values']     
            
            if self.xdates:
                x = element['mc_dates']
            else:
                x = element['mc_times']      
                
            # If cum sum, cumulative sum along time axis
            if cumsum:
                d = np.cumsum(d, axis=1)

            p_low = (100 - interval)/2
            p_med = 50
            p_high = 100 - p_low

            pL = np.percentile(d, p_low, axis=0)
            pM = np.percentile(d, p_med, axis=0)
            pH = np.percentile(d, p_high, axis=0)

            self.ax.plot(x, pM, color=color, label=label)
            self.ax.fill_between(x, pL, pH, alpha=0.33, color=color, linewidth=0)

        self.ax.legend()

        # if self.xdates:

            # self.ax.set_xticks([0, 100, 200])
            # self.ax.set_xticklabels(['yo', 'mo', 'do'])
