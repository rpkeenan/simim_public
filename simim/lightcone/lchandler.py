import os

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u

from matplotlib import animation

from simim.map import Gridder
from simim._paths import _SimIMPaths
from simim._handlers import Handler
from simim._pltsetup import *

from fnmatch import fnmatch

class LCHandler(Handler):
    """Class to handle I/O and basic analysis for light cone hdf5 files
    
    This class handles basic operations for accessing and analyzing lightcone
    data. 

    The general philosophy of Handlers is to not load actual properties of 
    a halo into memory until they are requested, and to remove them from 
    memory when they are no longer in use (or at least make it convenient
    to do so).
    """

    def __init__(self,sim,name,number,in_h_units=False):
        """Initialize Handler for a specified lightcone

        This class a generic interface for interacting with data in SimIM's
        standardized lightcone format. 

        Light cones must first be built using simim.lightcone.LCMaker.
        Lightcones are saved in sets in the SimIM data directory, each set of
        lightcones can contain one or many individual lightcones, created in a
        uniform way from a parent simulation. Accessing one of these lightcones
        in a Handler requires specifying the name of the parent simulation, the
        name of the lightcone set, and the number of the desired lightcone
        (indexed starting at 0).

        Parameters
        ----------
        sim : str
            The name of the simulation from which the lightcone was generated. 
        name : str
            The name of the lightcone set. 
        number : int
            The numeric index of the lightcone within the set
        in_h_units : bool
            If True, values will be returned, plotted, etc. in units including little h.
            If False, little h dependence will be removed. This can be overridden in 
            most method calls.
        """

        # Handle file path stuff
        paths = _SimIMPaths()
        if not sim in paths.lcs:
            raise ValueError("Lightcones for simulation {} not found. Try generating them or updating the path".format(sim))
        path = paths.lcs[sim]

        if not os.path.exists(os.path.join(path,name)):
            raise ValueError("Lightcones named '{}' not found. Try generating them or updating the path".format(name))
        path = os.path.join(path, name)

        if not os.path.exists(os.path.join(path,'lc_{:04d}.hdf5'.format(number))):
            raise ValueError("Lightcone number '{}' not found.".format(number))
        path = os.path.join(path,'lc_{:04d}.hdf5'.format(number))

        super().__init__(path=path,objectname='Lightcone',groupname='Lightcone Full',in_h_units=in_h_units)

        # Get the metadata
        self.metadata = {}
        with h5py.File(self.path,'r') as file:
            for key in file.attrs.keys():
                self.metadata[key] = file.attrs[key]

        # Make the cosmology
        self.cosmo = FlatLambdaCDM(H0=100*self.metadata['cosmo_h']*u.km/u.s/u.Mpc,
                                   Om0=self.metadata['cosmo_omega_matter'],
                                   Ob0=self.metadata['cosmo_omega_baryon'],
                                   Tcmb0=2.7255*u.K)
        self.open_angle = self.metadata['open angle']
        self.shape = self.metadata['shape']
        self.aspect_ratio = self.metadata['aspect ratio']
        self.minimum_redshift = self.metadata['minimum redshift']
        self.maximum_redshift = self.metadata['maximum redshift']
        self.extra_props['cosmo'] = self.cosmo
        self.extra_props['open_angle'] = self.open_angle
        self.extra_props['shape'] = self.shape
        self.extra_props['aspect_ratio'] = self.aspect_ratio
        self.extra_props['minimum redshift'] = self.minimum_redshift
        self.extra_props['maximum redshift'] = self.maximum_redshift


    def volume(self,redshift_min=None,redshift_max=None,shape=None,open_angle=None,aspect_ratio=None,in_h_units=None):
        """Compute the comoving volume of the light cone

        Parameters
        ----------
        redshift_min, redshift_max : float, optional
            The minimum/maximum redshift to consider - the minimum/maximum
            redshift of the lightcone by default
        open_angle : float, optional
            The opening angle of the lightcone - by default matches the
            lightcone
        aspect_ratio : float, optional
            The aspect ratio of the box sides (for square lightcones) -
            by default matches the lightcone
        shape : 'box', 'circle', optional
            The shape of the box - by default matches the lightcone
        in_h_units : bool (default is determined by self.default_in_h_units)
            If True, values will be returned in units including little h.
            If False, little h dependence will be removed. Defaults to whatever
            is set globally for the Handler instance.

        Returns
        -------
        volume : float
            The volume of the lightcone in comoving Mpc^3
        """

        if in_h_units is None:
            in_h_units = self.default_in_h_units

        # Set defaults
        if redshift_min == None:
            redshift_min=self.metadata['minimum redshift']
        if redshift_max == None:
            redshift_max=self.metadata['maximum redshift']
        if shape == None:
            shape=self.metadata['shape']
        if open_angle == None:
            open_angle=self.metadata['open angle']
        if aspect_ratio == None:
            aspect_ratio=self.metadata['aspect ratio']

        # Check everything
        if redshift_min < self.metadata['minimum redshift']:
            raise ValueError("redshift_min lower than minimum of lightcone")
        if redshift_max > self.metadata['maximum redshift']:
            raise ValueError("redshift_max higher than maximum of lightcone")

        if shape not in ['box','circle']:
            raise ValueError("Shape not recognized")

        if shape == self.metadata['shape'] or self.metadata['shape'] == 'box':
            if open_angle > self.metadata['open angle']:
                raise ValueError("openangle larger than lightcone")
            if open_angle*aspect_ratio > self.metadata['open angle']*self.metadata['aspect ratio']:
                raise ValueError("secondary angle (openangle * aspect) larger than lightcone")

        if shape == 'box' and self.metadata['shape'] == 'circle':
            radius2 = open_angle**2 + (open_angle*aspect_ratio)**2
            if radius2 > self.metadata['open angle']:
                raise ValueError("specified box does not fit in circular lightcone")

        # Figure out the volume
        if shape == 'box':
            area = open_angle * open_angle * aspect_ratio
        elif shape == 'circle':
            area = np.pi * (open_angle/2)**2

        v2 = self.cosmo.comoving_volume(redshift_max).value
        v1 = self.cosmo.comoving_volume(redshift_min).value
        volume = (v2-v1) * area/(4*np.pi)

        if in_h_units:
            volume *= self.h**3

        return volume

    def eval_stat_evo(self, redshift_bins, stat_function, kwargs, kw_remap={}, other_kws={}, zmin_kw=False, zmax_kw=False, volume_kw=False, give_args_in_h_units=None):
        """Compute the evolution of a statistic over a specified set of
        redshift bins
        """

        if give_args_in_h_units is None:
            give_args_in_h_units = self.default_in_h_units

        redshift_bins = np.sort(redshift_bins)
        
        inds_active_save = np.copy(self.inds_active)

        vals = []
        for bin in range(len(redshift_bins)-1):
            self.set_property_range('redshift',
                                    pmin=redshift_bins[bin],
                                    pmax=redshift_bins[bin+1],
                                    reset=True)

            if zmin_kw:
                other_kws['zmin'] = redshift_bins[bin]
            if zmax_kw:
                other_kws['zmax'] = redshift_bins[bin+1]
            if volume_kw:
                other_kws['volume'] = self.volume(redshift_min = redshift_bins[bin],
                                                    redshift_max = redshift_bins[bin+1],
                                                    in_h_units=give_args_in_h_units)

            vals.append(self.eval_stat(stat_function,
                                       kwargs=kwargs,
                                       kw_remap=kw_remap,
                                       other_kws=other_kws,
                                       use_all_inds=False,
                                       give_args_in_h_units=give_args_in_h_units))

            other_kws.pop('zmin',None)
            other_kws.pop('zmax',None)

        # Reset active inds
        self.inds_active = np.copy(inds_active_save)

        return redshift_bins, vals

    def grid(self, *property_names, 
             restfreq=None,
             in_h_units=None,use_all_inds=False,
             res=None,ralim=None,declim=None,zlim=None,
             norm=None):
        """Place selected properties into a 3d grid
        
        Uses the properties of the array to construct a position
        (ra,dec,redshift)-value (property_names) grid. Only required argument is
        a valid property name or names. Additional arguments can specify the
        limits and resolution of the grid. Passing a line rest frequency to
        restfreq will cause the grid to be constructed in terms of the
        corresponding observed frequencies instead of redshift.

        Parameters
        ----------
        property_names : str
            The name or names of properties in the Handler instance
        restfreq : float
            A rest frequency to use for converting the third axis from redshift
            to frequency. The returned axis will be constructed as
            restfreq/(1+z)
        in_h_units : bool (default is determined by self.default_in_h_units)
            If True, positions and property values fed to the gridder will be in
            units including little h. If False, little h dependence will be
            removed. Defaults to whatever is set globally for the Handler instance.
        use_all_inds : bool, default=False
            If True function all halos will be gridded, otherwise only active
            halos will be included.
        res : float, optional
            The resolution for the grid in Mpc (if in_h_units==False) or Mpc/h
            (if in_h_units==True). If no value is specified, it will default to
            1/100th of the box edge length
        ralim, decylim, zlim : tuples, optional
            Tuples containing minimum and maximum values of the grid along the
            x, y, and z axes, in units of Mpc (if in_h_units==False) or Mpc/h
            (if in_h_units==True). If no values are specified the defaults are
            (0, box edge length).
        norm : None, float
            Apply a normalization to the gridded values. If a float is given
            each cell will multiplied by the float

        Returns
        -------
        grid : simim.map.grid instance
            The gridded properties
        """

        if in_h_units is None:
            in_h_units = self.default_in_h_units
        
        x = self.return_property('ra',in_h_units=in_h_units,use_all_inds=use_all_inds)
        y = self.return_property('dec',in_h_units=in_h_units,use_all_inds=use_all_inds)
        z = self.return_property('redshift',in_h_units=in_h_units,use_all_inds=use_all_inds)
        if restfreq is not None:
            z = restfreq/(1+z)
        props = np.array([self.return_property(p,in_h_units=in_h_units,use_all_inds=use_all_inds) for p in property_names]).T
        
        if ralim is None:
            ralim = (-self.open_angle/2,self.open_angle/2)
        if declim is None:
            declim = (-self.open_angle*self.aspect_ratio/2,self.open_angle*self.aspect_ratio/2)
        if zlim is None:
            if restfreq is not None:
                zlim = (restfreq/(1+self.maximum_redshift), restfreq/(1+self.minimum_redshift))
            else:
                zlim = (self.minimum_redshift, self.maximum_redshift)

        c = []
        l = []

        for lim in ralim,declim,zlim:
            if len(lim) != 2:
                raise ValueError("ralim, declim, and zlim must each be None or have length = 2")
            else:
                c.append((lim[0]+lim[1])/2)
                l.append(np.max(lim) - np.min(lim))

        if res is None:
            res = [x/100 for x in l]

        grid = Gridder(np.array([x, y, z]).T, props,
                       center_point=c, side_length=l,
                       pixel_size=res)

        if norm is None:
            norm = 1        
        grid.grid *= norm

        return grid



    @pltdeco
    def animate(self, save=None, use_all_inds=False, colorpropname='mass',colorscale='log', sizepropname='mass',sizescale='log',in_h_units=None):
        """Make an animation of the light cone

        Parameters
        ----------
        use_all_inds : bool, optional
            If True values will be assigned for all halos, otherwise only
            active halos will be evaluated, and others will be assigned nan.
            Default is False.
        save : str, optional
            If specified, the plot will be saved to the given location
        colorpropname : str
            Name of the property used to determine marker color for each
            galaxy, default is 'mass'
        colorpropscale : 'log' or 'linear'
            Determines how the colorbar will be applied. Default is 'log'
        in_h_units : bool (default is determined by self.default_in_h_units)
            If True, values will be plotted in units including little h.
            If False, little h dependence will be removed. Defaults to whatever
            is set globally for the Handler instance.

        Returns
        -------
        None
        """

        if in_h_units is None:
            in_h_units = self.default_in_h_units
            
        x = self.return_property('pos_x',use_all_inds=use_all_inds,in_h_units=in_h_units)
        y = self.return_property('pos_y',use_all_inds=use_all_inds,in_h_units=in_h_units)
        z = self.return_property('pos_z',use_all_inds=use_all_inds,in_h_units=in_h_units)

        ra = self.return_property('ra',use_all_inds=use_all_inds,in_h_units=in_h_units)
        dec = self.return_property('dec',use_all_inds=use_all_inds,in_h_units=in_h_units)
        redshift = self.return_property('redshift',use_all_inds=use_all_inds,in_h_units=in_h_units)

        # Get limits
        phys_lim_xy = self.cosmo.comoving_transverse_distance(self.metadata['maximum redshift']).value * self.h * self.metadata['open angle']
        phys_lim_zmax = self.cosmo.comoving_distance(self.metadata['maximum redshift']).value * self.h
        phys_lim_zmin = self.cosmo.comoving_distance(self.metadata['minimum redshift']).value * self.h
        obs_lim_xy = self.metadata['open angle'] * 180/np.pi*60
        obs_lim_zmax = self.metadata['maximum redshift']
        obs_lim_zmin = self.metadata['minimum redshift']

        # Values along a circle, outline type
        outline = self.metadata['shape']
        edgepaternx = np.array([-1,1,1,-1,-1])
        edgepaterny = np.array([-1,-1,1,1,-1])

        theta = np.linspace(0,2*np.pi,1000)
        theta = np.concatenate((theta,[2*np.pi]))

        # Set up color map
        colorprop = self.return_property(colorpropname,use_all_inds=use_all_inds,in_h_units=in_h_units)
        if colorscale == 'log':
            colorprop = np.log10(colorprop)
        elif colorscale != 'linear':
            raise ValueError("colorscale not recognized")
        colormin = np.amin(colorprop)
        colorrange = np.ptp(colorprop)
        colors = cmap((colorprop-colormin)/colorrange)

        # Set up sizes
        sizeprop = self.return_property(sizepropname,use_all_inds=use_all_inds,in_h_units=in_h_units)
        if sizescale == 'log':
            sizeprop = np.log10(sizeprop)
        elif sizescale != 'linear':
            raise ValueError("sizescale not recognized")
        sizemin = np.amin(sizeprop)
        sizerange = np.ptp(sizeprop)
        sizes = ((sizeprop-sizemin)/sizerange + .2) * 5

        # set up plots
        figure = plt.figure(figsize=(8,8))
        figure.subplots_adjust(left=.05,right=.9,bottom=.15,top=.95)
        title = plt.suptitle('')
        # figure.legend(handles=[p0,p1,p2,p3,p4,p5],loc=8,bbox_to_anchor=(.5,0),ncol=3,markerscale=.5,fontsize='small')

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([colormin,(colormin+colorrange)])

        ax_cb = figure.add_axes([.91,.15,.03,.8])
        if colorscale == 'log':
            figure.colorbar(sm,ax=ax_cb,label='log '+colorpropname)
        else:
            figure.colorbar(sm,ax=ax_cb,label=colorpropname)

        # cone plot
        plot_cone = plt.subplot(212)
        plot_cone.set_title('Light Cone')

        cone = plot_cone.scatter(z, x, s=1, c=colors)
        box, = plot_cone.plot([], [], color='#ff7f0e')

        # physical cross section plot
        plot_xy = plt.subplot(222)
        plot_xy.set_title('Physical Cross-section (Mpc)')
        plot_xy.axis('equal')
        plot_xy.set(xlim = (-.55*phys_lim_xy,.55*phys_lim_xy), ylim = (-.55*phys_lim_xy,.55*phys_lim_xy))

        xy_scatter = plot_xy.scatter(x, y, s=np.zeros(len(x)), c=colors)
        xy_outline, = plot_xy.plot([], [], color='#ff7f0e')

        ### angular cross section plot
        plot_radec = plt.subplot(221)
        plot_radec.set_title('Angular Cross-section (arcmin)')
        plot_radec.axis('equal')
        plot_radec.set(xlim = (-.55*obs_lim_xy,.55*obs_lim_xy), ylim = (-.55*obs_lim_xy,.55*obs_lim_xy))

        radec_scatter = plot_radec.scatter(ra*180/np.pi*60, dec*180/np.pi*60, s=np.zeros(len(x)), c=colors)
        if outline == 'circle':
            radec_outline, = plot_radec.plot(obs_lim_xy/2*np.cos(theta), obs_lim_xy/2*np.sin(theta), color='#ff7f0e')
        else:
            radec_outline, = plot_radec.plot(obs_lim_xy/2*edgepaternx, obs_lim_xy/2*edgepaterny, color='#ff7f0e')

        # Initialization for animation_init
        def animation_init():
            title.set_text('')

            box.set_data([], [])

            # cone.set_sizes(np.zeros(len(x)))

            xy_outline.set_data([], [])

            radec_scatter.set_sizes(np.zeros(len(x)))

        step = 50
        nsteps = np.ceil((phys_lim_zmax-phys_lim_zmin)/step).astype('int')
        step = np.ceil((phys_lim_zmax-phys_lim_zmin)/nsteps).astype('int')

        def animate(i):
            #title
            redshift = z_at_value(self.cosmo.comoving_distance, (phys_lim_zmin+step*(i+.5))*u.Mpc)
            title.set_text('redshift = {:.2f}'.format(redshift))

            #animate p1
            box.set_data([phys_lim_zmin+step*i,phys_lim_zmin+step*i,phys_lim_zmin+step*(i+1),phys_lim_zmin+step*(i+1),phys_lim_zmin+step*i],
                         [-phys_lim_xy/2,phys_lim_xy/2,phys_lim_xy/2,-phys_lim_xy/2,-phys_lim_xy/2])

            #select halos to show in p2 and p3
            inds = np.where((z > phys_lim_zmin + step*i) & (z < phys_lim_zmin + step*(i+1)))
            new_sizes = np.zeros(len(z))
            new_sizes[inds] = sizes[inds]

            #animate p2
            xy_scatter.set_sizes(new_sizes)

            #animate p3
            radec_scatter.set_sizes(new_sizes)

            r = self.cosmo.comoving_transverse_distance(redshift) * self.metadata['open angle']/2
            if outline == 'circle':
                xy_outline.set_data(r*np.cos(theta),r*np.sin(theta))
            else:
                xy_outline.set_data(r*edgepaternx, r*edgepaterny)

        anim = animation.FuncAnimation(figure, animate, init_func=animation_init, frames=nsteps, interval=100, blit=False)
        if save != None:
            anim.save(save+'.mp4')

        plt.show()

class LCIterator():
    """Class to perform analysis on many light cone hdf5 files iteratively
    
    The general philosophy of Handlers is to not load actual properties of 
    a halo into memory until they are requested, and to remove them from 
    memory when they are no longer in use (or at least make it convenient
    to do so). Here we create the handlers for each lightcone and write 
    wrappers that iterate over all lcs and call the relevant LCHandler 
    method.
    """

    def __init__(self,sim,name,numbers=None,in_h_units=False,require_consistent=True):
        """Initialize Handler for all lightcone

        This class a generic interface for interacting with data in SimIM's
        standardized lightcone format, with the ability to iterate over 
        multiple light cones.

        Light cones must first be built using simim.lightcone.LCMaker.
        Lightcones are saved in sets in the SimIM data directory, each set of
        lightcones can contain one or many individual lightcones, created in a
        uniform way from a parent simulation. Accessing a set of lightcones
        requires specifying the name of the parent simulation, the
        name of the lightcone set, and optionally, the numbers of the desired 
        lightcones (default is to assume all lightcones are desired)

        Parameters
        ----------
        sim : str
            The name of the simulation from which the lightcone was generated. 
        name : str
            The name of the lightcone set. 
        numbers : list, optional
            List containing numeric indices of the lightcones within the set to include
        in_h_units : bool
            If True, values will be returned, plotted, etc. in units including little h.
            If False, little h dependence will be removed. This can be overridden in 
            most method calls.
        require_consistent : bool
            If True (default), will check that all light cones have the same basic parameters
        """

        # Handle file path stuff
        paths = _SimIMPaths()
        if not sim in paths.lcs:
            raise ValueError("Lightcones for simulation {} not found. Try generating them or updating the path".format(sim))
        path = paths.lcs[sim]

        if not os.path.exists(os.path.join(path,name)):
            raise ValueError("Lightcones named '{}' not found. Try generating them or updating the path".format(name))
        path = os.path.join(path, name)

        # Figure out what lightcones to load
        if numbers is None:
            files = os.listdir(path)
            numbers = [int(f.split('_')[1].split('.')[0]) for f in files if fnmatch(f,'lc_*.hdf5')]
            print("No lightcone numbers specified, initializing all {} light cones in {}/{}".format(len(numbers),sim,name))
        
        # Need these for iteration
        self.n_lcs = len(numbers)
        self.lc_numbers = np.array(numbers,ndmin=1)

        # Load all of the handlers as a dictionary
        self.lc_handlers = {}
        for i_lc in self.lc_numbers:
            self.lc_handlers[str(i_lc)] = LCHandler(sim, name, i_lc, in_h_units)

        # Consistency checking
        self.require_consistent = require_consistent
        self.is_consistent = True

        # Check for consistency of metadata
        ref_lc = self.lc_handlers[str(self.lc_numbers[0])]
        for attr in ['cosmo','open_angle','aspect_ratio','minimum_redshift','maximum_redshift']:
            setattr(self,attr,getattr(ref_lc,attr))
            for i_lc in self.lc_numbers[1:]:
                if getattr(self.lc_handlers[str(i_lc)],attr) != getattr(self,attr):
                    if self.require_consistent:
                        raise ValueError("Attribute {} does not match for light cones {} and {}".format(attr, self.lc_numbers[0],i_lc))
                    else:
                        self.is_consistent = False

        # Programatically create wrappers around Handler methods
        # setting the list manually to avoid things that won't work well
        # with wrappers (e.g. animation, plotting)

        attrs = [('eval_stat',"""Evaluate stat_function over the objects in each lightcone"""),
                 ('eval_stat_evo',"""Compute the evolution of a statistic over a specified set of redshift bins"""), 
                 ('delete_property',"""Remove a property from the saved file on the disk"""),
                 ('write_property',"""Write a property from object memory onto the saved file on the disk"""),
                 ('delete_property',"""Remove a property from the saved file on the disk"""),
                 ('make_property',"""Use a galprops.prop instance to evaluate a new property"""),
                 ('return_property',"""Load a property from lightcone file and return"""),
                 ('set_in_h_units',"""Globally set whether units are interpreted to be in little h units"""),
                 ('set_property_range',"""Set a range in a given property to be the active indices"""),
                 ]

        for attr,doc in attrs:
            def f(*args, attr=attr, **kwargs):
                result = {}
                for k, h in self.lc_handlers.items():
                    result[k] = getattr(h,attr)(*args, **kwargs)
                return result
            f.__doc__ = doc
            setattr(self,attr,f)


    def volume(self,redshift_min=None,redshift_max=None,shape=None,open_angle=None,aspect_ratio=None,in_h_units=None,number=None):
        """Compute the comoving volume of a single light cone

        Parameters
        ----------
        redshift_min, redshift_max : float, optional
            The minimum/maximum redshift to consider - the minimum/maximum
            redshift of the lightcone by default
        open_angle : float, optional
            The opening angle of the lightcone - by default matches the
            lightcone
        aspect_ratio : float, optional
            The aspect ratio of the box sides (for square lightcones) -
            by default matches the lightcone
        shape : 'box', 'circle', optional
            The shape of the box - by default matches the lightcone
        in_h_units : bool (default is determined by self.default_in_h_units)
            If True, values will be returned in units including little h.
            If False, little h dependence will be removed. Defaults to whatever
            is set globally for the Handler instance.
        number : int, optional
            Number of the light cone to execute function for, only relevant
            if light cones do not have consistent parameters

        Returns
        -------
        volume : float
            The volume of the lightcone in comoving Mpc^3
        """

        if not self.is_consistent and number is None:
            raise ValueError("Light cones do not have consistent parameters - specify 'number' to get volume based on the cosmology and geometry of a particular light cone")
        elif number is None:
            number = self.lc_numbers[0]
        elif number not in self.lc_numbers:
            raise ValueError("lightcone {} not in the list {}".format(number,self.lc_numbers))

        return self.lc_handlers[str(number)].volume(redshift_min=redshift_min,redshift_max=redshift_max,shape=shape,open_angle=open_angle,aspect_ratio=aspect_ratio,in_h_units=in_h_units)
    
    # def eval_stat_evo(self, redshift_bins, stat_function, kwargs, kw_remap={}, other_kws={}, zmin_kw=False, zmax_kw=False, volume_kw=False, give_args_in_h_units=None):
    #     """Compute the evolution of a statistic over a specified set of
    #     redshift bins"""

    #     result = {}
    #     for k, h in self.lc_handlers.items():
    #         result[k] = h.eval_stat_evo(redshift_bins=redshift_bins, stat_function=stat_function, kwargs=kwargs, kw_remap=kw_remap, other_kws=other_kws, zmin_kw=zmin_kw, zmax_kw=zmax_kw, volume_kw=volume_kw, give_args_in_h_units=give_args_in_h_units)
    #     return result