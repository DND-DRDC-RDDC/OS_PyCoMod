import numpy as np
import matplotlib.pyplot as mplot
import matplotlib.dates as mdates

months = mdates.MonthLocator()  #for month intervals on plots
months_fmt = mdates.DateFormatter('%b')

class plotter:
    @classmethod
    def show(cls):
        # display plots
        mplot.show()

    def __init__(self, figsize=(14,6), fontsize=12, title=None, xlabel=None, ylabel=None, ylimit=None):
        # mpl settings
        mplot.rc('font', size=fontsize)
        mplot.rc('figure', figsize=figsize)

        # create figure
        self.fig, self.ax = mplot.subplots()

        # grid lines
        self.ax.grid(True, ls=':')

        # monthly axis tick marks
        self.ax.xaxis.set_major_locator(months)
        self.ax.xaxis.set_major_formatter(months_fmt)

        # title
        if title is not None:
            self.ax.set_title(title)

        # axis labels
        if xlabel is not None:
            self.ax.set(xlabel=xlabel)
        if ylabel is not None:
            self.ax.set(ylabel=ylabel)

        # y axis limits
        if ylimit is not None:
            self.ax.set_ylim(*ylimit)

    def plot(self, run, elements, **kwargs):
        # first setup plot if not done already
        if self.fig is None:
            self.setup()

        x = run['x_dates']

        # init timeseries data for plotting
        d = np.zeros(len(x))

        # parse elements
        elements = elements.replace(' ', '').split('+') #remove whitespace and split on +

        # for each supplied element
        for s in elements:

            # split breadcrumbs
            s = s.split('.')

            # get the data
            data = run['output']
            for e in s:
                data = data[e]

            # if data is 2d (meaning it includes cohorts), sum across cohorts
            if data.ndim == 2:
                data = data.sum(axis=1)

            # append to data
            d = d + data


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


        # if cumulative
        if cumsum:
            d = np.cumsum(d)


        self.ax.plot(x, d, color=color, label=label)

        self.ax.legend()

    def plot_mc(self, run, elements, **kwargs):
        # first setup plot if not done already
        if self.fig is None:
            self.setup()

        r = run['reps']  #CHECK THAT THIS WORKS!!
        x = run['x_dates']

        # init timeseries data for plotting
        d = np.zeros((r,len(x)))

        # parse elements
        elements = elements.replace(' ', '').split('+') #remove whitespace and split on +

        # for each supplied element
        for s in elements:

            # split breadcrumbs
            s = s.split('.')

            # get the data
            data = run['output_mc']
            for e in s:
                data = data[e]

            # if data is 2d (meaning it includes cohorts), sum across cohorts
            if data.ndim == 3:
                data = data.sum(axis=2)

            # append to data
            d = d + data

        try:
            color = kwargs['color']
        except KeyError:
            color = 'steelblue'

        try:
            interval = kwargs['interval']
        except KeyError:
            interval = 75

        try:
            label = kwargs['label']
        except KeyError:
            label = '.'.join(args)

        try:
            cumsum = kwargs['cumsum']
        except KeyError:
            cumsum = False

        # if cum sum, cumulative sum along time axis
        if cumsum:
            d = np.cumsum(d, axis=1)

        p_low = (100 - interval)/2
        p_med = 50
        p_high = 100 - p_low

        pL = np.percentile(d, p_low, axis=0)
        pM = np.percentile(d, p_med, axis=0)
        pH = np.percentile(d, p_high, axis=0)

        self.ax.plot(x, pM, color=color, label=label)
        self.ax.fill_between(x, pL, pH, alpha=0.33, color=color, linewidth = 0)

        self.ax.legend()

