from tensegrity.plotter import MatplotlibPlotter
from tensegrity.snelson import Snelson

s = Snelson()

plotter = MatplotlibPlotter(s)
fig, ax = plotter.plot()

fig.show()
