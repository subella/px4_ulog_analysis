### Ulog analysis notebooks

A series of notebooks to plot various information from PX4 autopilot logs (ulog format).

The following three notebooks are probably of use to most people until things are cleaned up and refactored:

* `plot_adaptive.py`: Show the achieved vs. commanded trajectory of the drone and the adaptive terms
* `plot_comparison.py`: Show trajectory tracking error from two different runs
* `plot_aggregate.py`: Aggregate multiple trials and plot tracking error


To use jupytext, see [here](https://github.com/mwouts/jupytext/blob/master/docs/install.md). It hopefully works out of the box on most standard distros, but ymmv (mine did on arch).
