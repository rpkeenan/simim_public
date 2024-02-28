import matplotlib.pyplot as plt
from matplotlib import rc_context
from matplotlib import colormaps

pltsty = {
    'font.size' : 18,

    # Axis appearance
    'xtick.direction' : 'in',
    'ytick.direction' : 'in',
    'xtick.top' : True,
    'ytick.right' : True,

    # Legends
    'legend.frameon' : False,

    # Lines
    'lines.dashed_pattern' : [5,3],
}

# Matplotlib style sheet and wrapper
def pltdeco(func):
    def wrapper(*args, **kwargs):
        with rc_context(pltsty):
            return func(*args, **kwargs)
    return wrapper

cmap = colormaps['viridis']
cmap_r = colormaps['viridis_r']
