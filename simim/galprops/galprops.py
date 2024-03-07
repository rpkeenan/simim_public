import warnings

class MultiProp():
    def __init__(self,prop_names,prop_function,kwargs,
                 give_args_in_h_units=False,function_return_in_h_units=False,
                 units=['None'],h_dependence=[0],
                 wrap=True):
        """Class that describes how to add properties to a lightcone
        or snapshot

        Parameters
        ----------
        prop_names : list
            A list containing the names of the properties in order
        prop_function : function
            A function that given a set of arguments (specified by kwargs)
            will return a list containing the values for each property as
            separate elements. The order of the list must match the order
            in prop_names.
        kwargs : list
            The names of the keyword arguments for prop_function. These will
            correspond to properties already associated with the light cone/
            snapshot. A special case is "cosmo" which will retreive the
            cosmology metadata
        give_args_in_h_units : bool
            If False, all properties used as inputs to the prop_function call
            will be converted to units free of little h (ie Msun instead of
            Msun/h). Default if False.
        function_return_in_little_h : bool
            If False, all properties returned by the function will be converted
            into little h units using the h_dependence parameter
        units : list
            A list containing the units of the properties. It should either
            be of length one (if all properties have the same units) or should
            match the length of prop_names and have the same order.
        h_dependence : list
            A list containing the h dependence of the properties. It should
            either be of length one (if all properties have the same dependence)
            or should match the length of prop_names and have the same order.
            The dependence should be specified in powers of h, ie a quantity
            with units Mpc/h will have h_dependence = -1, a quantity with no
            h in the units will have h_dependence = 0, etc.
        wrap : bool
            If a function returns a single set of property values instead of a
            list of values for many properties (ie if only one property is
            returned), this should be set to True - or just use the prop class
        """

        self.names = prop_names
        self.n_props = len(prop_names)
        if self.n_props == 1 and wrap:
            self.wrap = True
        else:
            self.wrap = False

        self.prop_function = prop_function
        self.kwargs = kwargs
        if len(units)==1:
            self.units = {key:units[0] for key in self.names}
        elif len(units)==len(self.names):
            self.units = {self.names[i]:units[i] for i in range(len(self.names))}
        else:
            raise ValueError("units shape not compatible with prop_names shape")

        if len(h_dependence)==1:
            self.h_dependence = {key:h_dependence[0] for key in self.names}
        elif len(h_dependence)==len(self.names):
            self.units = {self.names[i]:h_dependence[i] for i in range(len(self.names))}
        else:
            raise ValueError("h_dependence shape not compatible with prop_names shape")

        self.give_args_in_h_units = give_args_in_h_units
        self.function_return_in_h_units = function_return_in_h_units

    def evaluate_all(self,target,use_all_inds=False,kw_remap={},kw_arguments={}):
        """Apply the properties to a target object.

        Parameters
        ----------
        target : lightcone.handler.handler or siminterface.simhandler.SnapHandler class
            The lightcone or snapshot to evaluate the property for
        use_all_inds : bool
            If True values will be assigned for all halos, otherwise only
            active halos will be evaluated, and others will be assigned nan.
        kw_remap : dict, optional
            A dictinary remaping kwargs of the property generating function to
            different properties of the lightcone. By default if the function
            has kwarg 'x' it will be evaluated on lightcone property 'x', but
            passing the dictionary {'x':'y'} will result in lightcone the
            function being evaluated on lightcone property 'y'.
        kw_arguments : dict, optional
            A list of additional keyword arguments to be passed to the property
            function call

        Returns
        -------
        vals : list
            A list containing the values of each property assigned.
        """

        arguments = {}
        for kwarg in self.kwargs:
            if kwarg in kw_remap.keys():
                kw_use = kw_remap[kwarg]
            else:
                kw_use = kwarg
    
            # Check that target has required fields (ie kwargs)
            if not target.has_property(kw_use) and not kw_use in target.extra_props.keys():
                raise ValueError("Property {} not found".format(kwarg))

            # Check if the kwarg is in target properties space or will need to be
            # loaded
            if target.has_property(kw_use):
                arguments[kwarg] = target.return_property(kw_use,use_all_inds,in_h_units=self.give_args_in_h_units)
            else:
                arguments[kwarg] = target.extra_props[kw_use]

        # Evaluate function
        vals = self.prop_function(**arguments,**kw_arguments)

        if self.wrap:
            vals = [vals]
        if len(vals) != len(self.names):
            raise ValueError("prop_function does not return a number of properties matching prop_names")

        # Now, if the function doesn't return in little-h units we need to
        # convert into them
        if not self.function_return_in_h_units:
            for i in range(len(vals)):
                if self.h_dependence[self.names[i]] != 0:
                    vals[i] = vals[i] / target.h**self.h_dependence[self.names[i]]

        return vals

    def evaluate(self,target,use_all_inds=False,kw_remap={},kw_arguments={}):
        """See evaluate_all - identical functionality. This
        is just a wrapper to match the prop subclass"""
        return self.evaluate_all(target,use_all_inds,kw_remap)

# Function should return an array of length n, where n is the number
# of objects it is evaluated over, or should be of shape n x m
# where m is the number of individual properties determined simultaneously
class Prop(MultiProp):
    def __init__(self,prop_name,prop_function,kwargs,
                 give_args_in_h_units=False,function_return_in_h_units=False,
                 units='None',h_dependence=0):
        """Class that describes how to add a single property to a lightcone
        or snapshot.

        Parameters
        ----------
        prop_name : str
            The name of the property
        prop_function : function
            A function that given a set of arguments (specified by kwargs)
            will return the values of the property for each halo.
        kwargs : list
            The names of the keyword arguments for prop_function. These will
            correspond to properties already associated with the light cone/
            snapshot. A special case is "cosmo" which will retreive the
            cosmology metadata
        give_args_in_h_units : bool
            If False, all properties used as inputs to the prop_function call
            will be converted to units free of little h (ie Msun instead of
            Msun/h). Default if False.
        function_return_in_little_h : bool
            If False, all properties returned by the function will be converted
            into little h units using the h_dependence parameter
        units : str
            The units of the property.
        h_dependence : float
            The h dependence of the property.
            The dependence should be specified in powers of h, ie a quantity
            with units Mpc/h will have h_dependence = -1, a quantity with no
            h in the units will have h_dependence = 0, etc.
        """

        super().__init__(prop_names=[prop_name],
                         prop_function=prop_function,
                         kwargs=kwargs,
                         give_args_in_h_units=give_args_in_h_units,
                         function_return_in_h_units=function_return_in_h_units,
                         units=[units],h_dependence=[h_dependence],
                         wrap=True)

        self.name = prop_name
        self.unit = units
        self.h_dep = h_dependence

    def evaluate(self,target,use_all_inds=False,kw_remap={}):
        """Apply the property to a target object.

        Parameters
        ----------
        target : lightcone.handler.handler or siminterface.simhandler.SnapHandler class
            The lightcone or snapshot to evaluate the property for
        use_all_inds : bool
            If True values will be assigned for all halos, otherwise only
            active halos will be evaluated, and others will be assigned nan.
        kw_remap : dict, optional
            A dictinary remaping kwargs of the property generating function to
            different properties of the lightcone. By default if the function
            has kwarg 'x' it will be evaluated on lightcone property 'x', but
            passing the dictionary {'x':'y'} will result in lightcone the
            function being evaluated on lightcone property 'y'.

        Returns
        -------
        vals : float or array
            The values of the property assigned to each halo.
        """

        vals = super().evaluate_all(target=target,
                                    use_all_inds=use_all_inds,
                                    kw_remap=kw_remap)
        return vals[0]


# Aliases for backwards compatibility
def prop(*args, **kwargs):
    warnings.warn("prop is depricated, use Prop instead")
    return Prop(*args, **kwargs)

def multiprop(*args, **kwargs):
    warnings.warn("multiprop is depricated, use MultiProp instead")
    return MultiProp(*args, **kwargs)