# Verbosity - 1=Default, 0=Don't print unless override
# Flux - 0=Default (no), 1=Flush after each print
_simim_verbosity = 1
_simim_verbosity_flush = False
_simim_verbosity_prefix = ''

def set_simim_verbosity(level=None, flush=None, prefix=None):
    """Set the level of verbosity for SimIM programs
    
    Parameters
    ----------
    level: {0,1,2}
        How verbose to be - 0=say nothing, 1=say normal monitoring statements,
        2=say debug statements. Defaults setting is 1 
    flush: bool
        Whether to flush print statement output immediately (added for running
        with slurm). Default setting is False
    prifx: string
        Appended before every printed output, default value is ''
    """
    global _simim_verbosity
    global _simim_verbosity_flush
    global _simim_verbosity_prefix

    # Check errors
    if level is not None:
        if level not in [0,1,2,True,False]:
            raise ValueError("Verbosity level must be 0 (no verbose), 1 (verbose), or 2 (debug)")
    if flush is not None:
        if not isinstance(flush, bool):
            raise ValueError("flush must be True/False")

    # Set level
    if level is not None:
        _simim_verbosity = level

    # Set flush
    if flush is not None:
        _simim_verbosity_flush = flush

    # Set prefix
    if prefix is not None:
        _simim_verbosity_prefix = prefix


def simim_verbose(*args,level=1,**kwargs):
    """Print stuff with settable verbosity level
    
    Parameters
    ----------
    *args:
        Passed to print function if verbose is on
    **kwargs:
        Passed to print function if verbose is on
    level: {-1,1,2}
        Verbosity level for print statement - level=-1: force print,
        even when verbose is set to 0, level=1: print when verbose 
        level is 1, level=2: print when verbose level is 2 (debug)
    """
    global _simim_verbosity
    global _simim_verbosity_flush
    global _simim_verbosity_prefix

    if level not in [-1,1,2]:
        raise ValueError("Verbosity level not recognized")

    if level <= _simim_verbosity:
        print(_simim_verbosity_prefix,*args,flush=_simim_verbosity_flush,**kwargs)

def simim_input(arg,**kwargs):
    """Wrap input calls to match verbose prefix (cannot change verbosity level)
    
    Parameters
    ----------
    arg: str
        Passed to input function
    **kwargs:
        Passed to input function
    """
    global _simim_verbosity_prefix

    return input(_simim_verbosity_prefix+arg,**kwargs)