from typing import Callable
from numpy.typing import ArrayLike
import warnings
import os

import numpy as np
from scipy.signal import fftconvolve

from simim._pltsetup import *
from matplotlib import animation

# To do
# 1. Axis transformations and apropriate transformations of data
# 5. Average instead of sum within a cell (e.g. for gridding timestreams)

###############################################################################
##### SECTION 0: Helper Functions #############################################
###############################################################################
def _axis_edges(ax):
    """Given bin centers (ax) along an axis return the bin edges"""
    diff = np.diff(ax)
    newax = ax[:-1] + diff
    newax = np.concatenate(([ax[0]-diff[0]/2], newax, [ax[-1]+diff[-1]/2]))
    return newax
    
    # delta = np.mean(np.abs(np.diff(ax)))
    # ax = ax - delta/2
    # ax = np.concatenate((ax,[ax[-1]+delta]))
    # return ax

def _unpad(x, pad_width):
    """Remove pad_width elements from each end of array x"""

    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return x[tuple(slices)]

###############################################################################
##### SECTION 1: Under the hood functionality #################################
###############################################################################
class Grid():
    """Parent class for grids
    
    This class contains most of the functionality, for making and analyzing
    grids (maps, 3d cubes, spectral cubes, etc). Child classes defined elsewhere
    should be used to actually initialize and construct the grid.
    """

    def __init__(self,n_properties,center_point,side_length,pixel_size,axunits=None,gridunits=None):
        """Create a Grid instance

        This method creates an instance for handling gridded data in arbitrary
        dimensions. The dimensions and axes of the grid are specified here. The
        data grid itself is not created in memory until the ``init_grid`` method
        is called. 

        The constructed grid has N+1 dimensions, where N is the number of
        "spatial" dimensions of the data, and the final index can be used to
        access grids for multipl properties.

        Parameters
        ----------
        n_properties : int >= 1
            The number of properties which will be stored in the grid. Note this
            number can be increased later
        center_point : tuple of floats
            The center point in each spatial dimension of the grid.
        side_length : float or tuple of floats
            The length along each spatial dimension of the grid. If a single
            value is given, all dimensions will be assumed to have the same
            length. Otherwise the length of the tuple should match the length of
            the the center_point tuple.
        pixel_size : float or tuple of floats
            The length along each spatial dimension of a single grid cell. If a
            single value is given, all dimensions will be assumed to have the
            same length. Otherwise the length of the tuple should match the
            length of the the center_point tuple. If side_length is not not an
            an integer multiple of pixel_size the side length will be shortened
            to give an integer number of pixels.
        axunits : str or tuple of str (optional)
            A string or tuple of strings specifying the units of the spatial
            axes.
        gridunits : str or tuple of str (optional)
            A string or tuple of strings specifying the units of the grid values.
            A single value can be specified, or one value can be provided for 
            each property, in which case the length of the tuple should equal
            n_properties.

        Class atributes
        ---------------
        self.n_objects : int
            The number of objects with positions specified
        self.n_dimensions : int
            The number of dimensions in which the grid positions are set
        self.n_properties : int
            The number of properties specified in values (if values isn't
            specified this will be 1, for the number counts returned by default)
        self.pixel_size : self.n_dimensions x float array
            The pixel size in each dimension
        self.center_point : n_dimensions x float array
            The grid center point
        self.side_length : n_dimensions x float array
            The length of the box sides in each dimension
        self.n_pixels : n_dimensions x int array
            The number of pixels along each dimension
        self.axunits : n_dimensions x str array
            The units of the axes
        self.gridunits : n_properties x str array
            The units of the grid
        self.axes : list of n_dimensions arrays
            The physical values along each dimension of the grid
        self.fourier_axes : list of n_dimensions arrays
            The physical values along each dimension of the fourier transform of
            the grid
        self.fourier_space : list of n_dimensions bools
            If self.fourier_space[i] is True, the grid is in fourier space along
            the ith dimension
        self.grid : n_dimensions + 1 dimensional array
            The gridded values with the first n_dimensions axes corresponding to
            the positions, and the final axis indexing the different properties
            that have been gridded.
        """

        # Handle n_properties
        self.n_properties = n_properties
        self.n_objects = 0

        # Handle center_point inputs:
        self.center_point = np.array(center_point,ndmin=1,copy=True)
        self.n_dimensions = len(self.center_point)

        # Handle side_length inputs:
        self.side_length = np.array(side_length,ndmin=1,copy=True)
        if len(self.side_length) == 1:
            self.side_length = np.ones(self.n_dimensions)*self.side_length[0]
        if len(self.side_length) != self.n_dimensions:
            raise ValueError("side_length don't match data dimensionality")

        # Handle pixel_size inputs
        self.pixel_size = np.array(pixel_size,ndmin=1,copy=True)
        if len(self.pixel_size) != self.n_dimensions:
            if len(self.pixel_size) == 1:
                self.pixel_size = np.ones(self.n_dimensions)*self.pixel_size[0]
            else:
                raise ValueError("pixel_sizes don't match data dimensionality")

        # Handle units inputs:
        if axunits is None:
            self.axunits = np.array([None for i in range(self.n_dimensions)])
            self.fourier_axunits = np.array([None for i in range(self.n_dimensions)])
        else:
            self.axunits = np.array(axunits,ndmin=1,copy=True)
            if len(self.axunits) == 1:
                self.axunits = np.array([self.axunits[0] for i in range(self.n_dimensions)])
            if len(self.axunits) != self.n_dimensions:
                raise ValueError("axunits don't match data dimensionality")
            self.fourier_axunits = np.array([i+'^-1' if i is not None else None for i in self.axunits],ndmin=1)

        if gridunits is None:
            self.gridunits = np.array([None for i in range(self.n_properties)])
        else:
            self.gridunits = np.array(gridunits,ndmin=1,copy=True)
            if len(self.gridunits) == 1:
                self.gridunits = np.array([self.gridunits[0] for i in range(self.n_properties)])
            if len(self.gridunits) != self.n_properties:
                raise ValueError("gridunits don't match number of properties")

        # Set up the grid - make sure the side length is compatible with
        # number of pixels
        n_pixels_decimal = self.side_length/self.pixel_size
        n_pixels_ceil = np.ceil(np.round(n_pixels_decimal,1))
        if np.any(n_pixels_decimal != n_pixels_ceil):
            self.side_length = self.pixel_size * n_pixels_ceil
            if (not side_length is None) and (not center_point is None):
                warnings.warn("Side lengths increased to accomodate integer number of pixels")
        self.n_pixels = n_pixels_ceil.astype('int')

        self.axes = []
        self.axes_centers = []
        self.fourier_axes = []
        self.fourier_axes_centers = []
        for i in range(self.n_dimensions):
            axis = np.arange(self.n_pixels[i]+1,dtype='float')
            axis *= self.pixel_size[i]
            axis -= self.side_length[i]/2
            axis += self.center_point[i]
            self.axes_centers.append(axis[:-1]+self.pixel_size[i]/2)
            self.axes.append(axis)

            self.fourier_axes_centers.append(np.fft.fftshift(np.fft.fftfreq(self.n_pixels[i],self.pixel_size[i])))
            self.fourier_axes.append(_axis_edges(self.fourier_axes_centers[i]))

        self.fourier_space = np.zeros(self.n_dimensions,dtype=bool)

        self.grid_active = False
        self.is_power_spectrum = False


    def _check_property_input(self,properties):
        """Make sure a list of properties are all in the grid
        
        Checks that all values in list properties are < self.n_properties"""

        if properties is None:
            properties = np.arange(self.n_properties)
        
        properties = np.array(properties,ndmin=1)
        if not np.all(np.isin(properties,np.arange(self.n_properties))):
            raise ValueError("Some property indices do not correspond to property indices of the grid")
        
        return properties


    def init_grid(self):
        """Create the data grid
        
        This function is separated out from __init__ so that memory isn't used up
        when it isn't needed, and so that grids can be created in different ways
        by child classes.
        
        Parameters
        ----------
        None
        """
        self.grid_active = True
        self.grid = np.zeros(np.concatenate((self.n_pixels,[self.n_properties])))
        self.n_objects = 0


    def pad(self,ax,pad,val=0):
        """Pad/unpad the edges of a grid along specified axes
        
        This method adds a specified number of additional cells to BOTH the
        beginning and end of a grid. The axes to pad can be specified with the
        ax parameter. By default zero padding is performed, but a different
        value can be specified in val.

        If a value less than zero is specified in pad, the edges of the grid
        will be cropped.

        This method will apply the padding in whatever space a given axis is 
        currently represented in (Fourier or physical space).

        Parameters
        ----------
        ax : int or tuple of ints
            The index or indices of the axis to pad, each value should
            correspond to a spatial axis of the grid (so for a 3d grid 0, 1, or
            2 are valid).
        pad : int or tuple of ints
            The number of cells to add to EACH side of the axis(es) specified by
            the ax parameter. If a tuple is given, it should have the same shape
            as ax. If a negative value is specified, the axes will be cropped by
            this number of cells.
        val : float, optional
            The value to store in the added cells. This defaults to 0
        """

        # Pads symmetrically - specify one value in pad for each axis in ax.
        # Negative values of pad will unpad the grid.
        ax = np.array(ax,ndmin=1).astype(int)
        if np.any(ax>=self.n_dimensions) or np.any(ax<0):
            raise ValueError('specified axes not valid')

        pad = np.array(pad,ndmin=1).astype(int)
        if len(pad) == 1 and len(ax)>1:
            pad = np.ones(len(ax),dtype=int)*pad[0]
        elif len(pad) != len(ax):
            raise ValueError("length of ax ({}) and pad ({}) must match".format(len(ax),len(pad)))
        
        # Check we're not removing more pixels than there are:
        for i in range(len(ax)):
            if pad[i] < 0 and self.n_pixels[ax[i]] <= -2*pad[i]:
                raise ValueError("Cannot remove more pixels than present in dimension {}".format(ax[i]))

        pad_full = []
        unpad_full = []

        new_axes = [a for a in self.axes]
        new_axes_centers = [a for a in self.axes_centers]
        new_fourier_axes = [a for a in self.fourier_axes]
        new_fourier_axes_centers = [a for a in self.fourier_axes_centers]
        new_n_pixels = np.copy(self.n_pixels)
        new_side_length = np.copy(self.side_length)
        new_pixel_size = np.copy(self.pixel_size)

        for i in range(self.n_dimensions):
            if i not in ax:
                pad_full.append((0,0))
                unpad_full.append((0,0))

            else:
                i_pad = ax.tolist().index(i)

                if pad[i_pad] == 0:
                    pad_full.append((0,0))
                    unpad_full.append((0,0))

                else:
                    if pad[i_pad] > 0:
                        pad_full.append((pad[i_pad],pad[i_pad]))
                        unpad_full.append((0,0))
                    elif pad[i_pad] < 0:
                        pad_full.append((0,0))
                        unpad_full.append((-pad[i_pad],-pad[i_pad]))

                    new_n_pixels[i] = self.n_pixels[i] + 2*pad[ax[i]]
                    if self.fourier_space[i]:
                        new_side_length[i] = self.side_length[i]
                        new_pixel_size[i] = self.side_length[i] / new_n_pixels[i]
                    else:
                        new_side_length[i] = new_n_pixels[i] * self.pixel_size[i]
                        new_pixel_size[i] = self.pixel_size[i]

                    new_axes[i] = np.arange(new_n_pixels[i]+1,dtype='float') * new_pixel_size[i] - new_side_length[i]/2 + self.center_point[i]
                    new_axes_centers[i] = new_axes[i][:-1] + new_pixel_size[i]/2

                    new_fourier_axes_centers[i] = np.fft.fftshift(np.fft.fftfreq(new_n_pixels[i],new_pixel_size[i]))
                    new_fourier_axes[i] = _axis_edges(new_fourier_axes_centers[i])

        # Won't pad properties array:
        pad_full.append((0,0))
        unpad_full.append((0,0))

        # Pad / unpad the relevant axes and update the grid parameters
        if self.grid_active:
            self.grid = np.pad(self.grid,pad_full,constant_values=val)
            self.grid = _unpad(self.grid,unpad_full)

        self.axes = new_axes
        self.axes_centers = new_axes_centers
        self.fourier_axes = new_fourier_axes
        self.fourier_axes_centers = new_fourier_axes_centers
        self.n_pixels = new_n_pixels
        self.side_length = new_side_length
        self.pixel_size = new_pixel_size

        if self.grid_active:
            if self.grid.shape[:-1] != tuple(self.n_pixels):
                raise ValueError("Something went wrong - grid doesn't have expected size")


    # Should extend this method to work on fourier space
    def crop(self,ax,min=None,max=None):
        """Crop a single grid axis to a specified range of values

        Note that this will only work on axes which are represented in physical
        (not Fourier) space
        
        Parameters
        ----------
        ax : int
            The index of the axis to pad. This should correspond to a spatial
            axis of the grid (so for a 3d grid 0, 1, or 2 are valid inputs).
        min : float (optional)
            Minimum value along the axis that will be kept. If none is specified
            the lower end of the grid will not be cropped. Only cells where the
            cener lies above the value of min are kept.
        max : float (optional)
            Maximum value along the axis that will be kept. If none is specified
            the upper end of the grid will not be cropped. Only cells where the
            center lies below the value of max are kept.
        """
    
        # Still need to sort out how to set up new axes
        ax = int(ax)
        if ax < 0 or ax > self.n_dimensions-1:
            raise ValueError("Specified axis does not exist")

        # Check if this is in Fourier space
        if self.fourier_space[ax]:
            raise ValueError("cropping not supported for Fourier space (see the pad method)")
        
        # Set min and max if not specified
        if min is None:
            min = np.min(self.axes[ax])
        if max is None:
            max = np.max(self.axes[ax])
        if max <= min:
            raise ValueError("max must be greater than min")
        
        if np.all(self.axes_centers[ax]>max) or np.all(self.axes_centers[ax]<min):
            raise ValueError("No cells within specified limits")

        # Crop the grid
        if self.grid_active:
            self.grid = np.take(self.grid,np.nonzero((self.axes_centers[ax]>=min) & (self.axes_centers[ax]<=max))[0],axis=ax)
            
        self.axes_centers[ax] = self.axes_centers[ax][(self.axes_centers[ax]>=min) & (self.axes_centers[ax]<=max)]
        self.axes[ax] = np.concatenate((self.axes_centers[ax] - self.pixel_size[ax]/2,[np.max(self.axes_centers[ax]) + self.pixel_size[ax]/2]))
        self.n_pixels[ax] = len(self.axes_centers[ax])
        self.side_length[ax] = np.ptp(self.axes[ax])
        self.center_point[ax] = np.min(self.axes[ax]) + self.side_length[ax]/2
        self.fourier_axes_centers[ax] = np.fft.fftshift(np.fft.fftfreq(self.n_pixels[ax],self.pixel_size[ax]))
        self.fourier_axes[ax] = _axis_edges(self.fourier_axes_centers[ax])


    # Should expand this method to support linear transformations
    def scale_axis(self,ax,factor,axunits='same',scalecenter=True):
        """Rescale an axis or axes of the grid by a constant factor

        Note that this transformation is always applied to the real-
        space axes, the Fourier axes will be rescaled by 1/factor.
        
        Parameters
        ----------
        ax : int or tuple of ints
            The index or indices of the axis to transform, each value should
            correspond to a spatial axis of the grid (so for a 3d grid 0, 1, or
            2 are valid).
        factor : float or tuple of floats
            The factor by which to multiply each axis specified in ax.
        scalecenter : bool
            Specifies whether the center of the grid should also be scaled by the
            scale factor or if the center point should be held fixed and only the
            extent of the grid relative to this point should be rescaled.
        axunits : 'same' or str
            If 'same' (default) the axis units will be left unchanged. Otherwise
            the units of the transformed axis will be changed to the specified 
            string.        
        """

        ax = np.array(ax,ndmin=1).astype(int)
        if np.any(ax>=self.n_dimensions) or np.any(ax<0):
            raise ValueError('specified axes not valid')
        
        factor = np.array(factor,ndmin=1)
        if len(factor) == 1:
            factor = np.ones(len(ax))*factor[0]
        elif len(factor) != len(ax):
            raise ValueError('length of factor must be 1 or match the length of ax')
            
        for i in ax:
            if scalecenter:
                self.center_point[i] *= factor[i]
            self.side_length[i] *= factor[i]
            self.pixel_size[i] *= factor[i]

            # Handle units inputs:
            if axunits is None:
                self.axunits[i] = None
                self.fourier_axunits[i] = None
            elif axunits == 'same':
                self.axunits[i] = self.axunits[i]
                self.fourier_axunits[i] = self.fourier_axunits[i]
            elif len(np.array(axunits,ndmin=1,copy=True))==1:
                self.axunits[i] = axunits
                self.fourier_axunits[i] = axunits+'^-1'
            elif len(np.array(axunits,ndmin=1,copy=True))==len(ax):
                self.axunits[i] = axunits[i]
                self.fourier_axunits[i] = axunits[i]+'^-1'
            else:
                raise ValueError('length of axunits must be 1 or match the length of ax')

            # Set up the grid - make sure the side length is compatible with
            # number of pixels
            if scalecenter:
                self.axes[i] = self.axes[i]*factor[i]
                self.axes_centers[i] = self.axes_centers[i]*factor[i]
            else:
                self.axes[i] = (self.axes[i]-self.center[i])*factor[i] + self.center[i]
                self.axes_centers[i] = (self.axes_centers[i]-self.center[i])*factor[i] + self.center[i]

            self.fourier_axes[i] = self.fourier_axes[i]/factor[i]
            self.fourier_axes_centers[i] = self.fourier_axes_centers[i]/factor[i]


    def copy_axes(self,n_properties=1):
        """Create a new grid instance with the same axes as this one
        
        Parameters
        ----------
        n_properties : int
            The number of properties to include in the new grid

        """
        grid_copy = Grid(n_properties=n_properties,
                          center_point=np.copy(self.center_point),
                          side_length=np.copy(self.side_length),
                          pixel_size=np.copy(self.pixel_size),
                          axunits=np.copy(self.axunits),
                          gridunits=np.copy(self.gridunits))
        
        return grid_copy


    def copy(self,properties=None):
        """Make a copy of this Grid as a new Grid instance
        
        Parameters
        ----------
        properties : int or tuple of ints
            Indices of the grid properties to keep in the copy. By 
            default all properties are kept.
        """

        properties = self._check_property_input(properties)
        n_properties = len(properties)

        grid_copy = self.copy_axes(n_properties)
        
        if self.grid_active:
            grid_copy.grid = np.copy(self.grid[...,tuple(properties)])
            grid_copy.grid_active = True
            grid_copy.n_objects = self.n_objects

        grid_copy.fourier_space = np.copy(self.fourier_space)
        grid_copy.is_power_spectrum = self.is_power_spectrum

        return grid_copy


    def save(self,path,compress=False,overwrite=False):
        """Save the grid instance in a .npz file that can be
        reloaded with the LoadGrid class
        
        Parameters
        ----------
        path : str
            filename to which the data is to be saved
        compress : bool (default=False)
            If True the grid will be saved in a compressed format
        overwrite : bool (default=False)
            If path already exists and overwrite is set to False the
            file will not be overwritten and an error will be raised.
        """

        if compress:
            savefunc = np.savez_compressed
        else:
            savefunc = np.savez

        if os.path.exists(path) and not overwrite:
            raise ValueError("The file you are trying to create already exists")

        save_data = {'n_properties':self.n_properties,
                     'center_point':self.center_point,
                     'side_length':self.side_length,
                     'pixel_size':self.pixel_size,
                     'axunits':self.axunits,
                     'gridunits':self.gridunits,
                     'grid_active':self.grid_active,
                     'n_objects':self.n_objects,
                     'fourier_space':self.fourier_space,
                     'is_power_spectrum':self.is_power_spectrum
                     }
        if self.grid_active:
            save_data['grid'] = self.grid

        savefunc(path,**save_data)


    def add_new_prop(self,value=None):
        """Add a new property to the grid"""

        if self.grid_active:
            if value is None:
                value = np.zeros(np.concatenate((self.n_pixels,[1])))
            elif value.ndim == self.grid.ndim:
                if np.any(value.shape[:-1] != self.n_pixels):
                    raise ValueError("Value array must match shape of grid")
            elif value.ndim == self.n_dimensions:
                if np.any(value.shape != self.n_pixels):
                    raise ValueError("Value array must match shape of grid")
                value = value[...,np.newaxis]
            else:
                raise ValueError("Value array must match shape of grid")

            self.grid = np.concatenate((self.grid,value),axis=-1)
            self.n_properties += value.shape[-1]

        # Different behavior if grid isn't activated yet
        if not self.grid_active:
            if value is not None:
                raise ValueError("data array for this Grid has not been initialized, cannot add new values in property array -- use add_new_prop(value=None) to add an empty new property without initializing grid")
            else:
                self.n_properties += 1



    def add_from_cat(self,positions,values=None,new_props=False,properties=None):
        """Add values to the grid

        Parameters
        ----------
        positions : array
            The positions of each object should be an n_objects x n_dimensions
            array
        values : array (optional)
            Values to grid, should be an n_objects x n_properties array. If
            values is not specified and the Grid's n_properties attribute is 
            equal to 1, counts in cell for each postions value will be added 
            to the grid.
        new_props : bool (optional)
            If True, this method will create new entries along the poperty
            dimension for each set of values given rather than adding them
            on top of existing values. Default is False.
        properties : int or list of ints (optional)
            If specified, the values will only be added to the specified property
            indices. If None, values will be added to all property indices.
            Ignored if new_props = True
        """
        
        if not self.grid_active:
            raise ValueError("data array for this Grid has not been initialized")
        
        if np.any(self.fourier_space):
            raise ValueError("Some axes are in fourier space, cannot add new properties in map space.")
        
        # If the positions array is empty don't need to do much
        if len(positions) == 0:
            if positions.ndim == 1:
                positions = positions.reshape((positions.shape[0],1))

                if new_props:
                    new_n_properties = values.shape[1]
                    new_grid = np.zeros(np.concatenate((self.n_pixels,[new_n_properties])))
                    self.add_new_prop(new_grid)

        # Otherwise make sure the positions array is in the right shape and matches the shape of the grid
        else:
            positions = np.array(positions,ndmin=1,copy=True)
            if positions.shape[0] > 0:
                if positions.shape[1] !=self.n_dimensions:
                    raise ValueError('positions (dim={}) does not have the right number of dimensions for the grid (dim={})'.format(positions.shape[1],self.n_dimensions))

            if values is None:
                values = np.ones(len(positions))
            values = np.array(values,ndmin=1,copy=True)
            if values.ndim == 1:
                values = values.reshape((values.shape[0],1))

            if values.shape[0] != positions.shape[0]:
                raise ValueError("position and values array do not have equal length.")

            # Put data into coordinate units
            positions -= self.center_point.reshape(1,self.n_dimensions)
            positions += self.side_length.reshape(1,self.n_dimensions)/2
            positions /= self.pixel_size.reshape(1,self.n_dimensions)
            positions = np.floor(positions).astype('int')

            # Get rid of anything that doesn't fit
            values = values[(~np.any(positions<0,axis=1)) & (~np.any(positions>=self.n_pixels,axis=1))]
            positions = positions[(~np.any(positions<0,axis=1)) & (~np.any(positions>=self.n_pixels,axis=1))]
            positions = tuple(positions.T[i] for i in range(positions.shape[1]))

            if new_props:
                new_n_properties = values.shape[1]
                new_grid = np.zeros(np.concatenate((self.n_pixels,[new_n_properties])))
                np.add.at(new_grid,positions,values)
                self.add_new_prop(new_grid)

            else:
                properties = self._check_property_input(properties)
                if values.shape[1] != len(properties) and values.shape[1] != 1:
                    raise ValueError("Values array does not contain the correct number of properties.")

                for ip in properties:
                    if values.shape[1] == 1:
                        v = values.flatten()
                    else:
                        v = values[:,ip]
                    np.add.at(self.grid,positions+(ip,),v)

            # Count the number of objects
            self.n_objects += len(values)


    def add_from_pos_plus_array(self, positions: ArrayLike, values: ArrayLike, ax: int = -1, new_prop: bool = False, properties: int = None):
        """Add values to a grid from a list of positions (dimension N-1), and
        the values along the final dimension
        
        Takes an array of positions with 1 dimension less than the dimensionality 
        of the grid plus an array that will be added at that position along the final 
        dimension.

        Parameters
        ----------
        positions : array
            The positions of each object should be an n_objects x (n_dimensions-1)
            array
        values : array
            Values to grid, should be an n_objects x grid_size_in_final_dimension array.
        ax : int (optional)
            The axis along which the values are to be added, by default the final position
            axis is assumed
        new_prop : bool (optional)
            If True, this method will create a new entry along the poperty
            dimension of the grid rather than adding values on top of any existing values
            in the grid. Default is False.
        properties : int or list of ints (optional)
            If specified, the values will only be added to the specified property
            indices. If None, values will be added to all property indices.
            Ignored if new_props = True
        """

        if not self.grid_active:
            raise ValueError("data array for this Grid has not been initialized")
        
        if np.any(self.fourier_space):
            raise ValueError("Some axes are in fourier space, cannot add new properties in map space.")

        # Check that the grid has the specified axis
        if ax < 0: 
            ax = self.n_dimensions+ax
        if ax>=self.n_dimensions or ax<0:
            raise ValueError('specified axes not valid')

        # If the positions array is empty don't need to do much
        if len(positions) == 0:
            if positions.ndim == 1:
                if new_prop:
                    new_grid = np.zeros(np.concatenate((self.n_pixels,[1])))
                    self.add_new_prop(new_grid)
            
            return

        # Otherwise make sure the positions and values arrays are in the right shapes and match the shape of the grid
        positions = np.array(positions,ndmin=2)
        values = np.array(values,ndmin=2)

        if positions.shape[0] > 0:
            if positions.shape[1] != self.n_dimensions-1:
                raise ValueError('positions (dim={}) should have one fewer dimensions than the grid (dim={})'.format(positions.shape[1],self.n_dimensions))

        if values.shape[0] != positions.shape[0]:
            raise ValueError("values (len={}) and positions (len={}) should have the same first dimension length".format(len(values), len(positions)))

        if values.shape[1] != self.n_pixels[ax]:
            raise ValueError("The second dimension size of values (d={}) should match the size of grid dimension {} (d={})".format(values.shape[0],ax,self.n_pixels[ax]))

        # Put data into coordinate units
        cp = np.delete(self.center_point,ax).reshape(1, self.n_dimensions-1)
        sl = np.delete(self.side_length,ax).reshape(1, self.n_dimensions-1)
        px = np.delete(self.pixel_size,ax).reshape(1, self.n_dimensions-1)
        nx = np.delete(self.n_pixels,ax)

        positions = (positions - cp + sl/2) / px
        positions = np.floor(positions).astype('int')

        # Get rid of anything that doesn't fit
        values = values[(~np.any(positions<0,axis=1)) & (~np.any(positions>=nx,axis=1))]
        positions = positions[(~np.any(positions<0,axis=1)) & (~np.any(positions>=nx,axis=1))]

        # Convert postions into tuples of coordinate lists:
        positions = tuple(positions.T[i] for i in range(positions.shape[1]))

        # Move array-axis to predictable location (end of grid)
        if not new_prop:
            if properties is None:
                self.grid = np.moveaxis(self.grid,ax,-2)
                np.add.at(self.grid, positions, np.expand_dims(values,2))
                self.grid = np.moveaxis(self.grid,-2,ax)
            else:
                properties = self._check_property_input(properties)
                for ip in properties:
                    self.grid = np.moveaxis(self.grid,ax,-2)
                    np.add.at(self.grid, positions+(slice(None),ip,), values)
                    self.grid = np.moveaxis(self.grid,-2,ax)


        else:
            new_grid = np.zeros(np.concatenate((self.n_pixels,[1])))
            new_grid = np.moveaxis(new_grid,ax,-2)
            np.add.at(new_grid,positions,np.expand_dims(values,2))
            new_grid = np.moveaxis(new_grid,-2,ax)
            
            self.add_new_prop(new_grid)

        # Count the number of objects
        self.n_objects += len(values)


    def add_from_spec_func(self, positions: ArrayLike, spec_function: Callable, spec_function_arguments: ArrayLike = None, spec_ax: int = -1,
                           is_cumulative: bool = False, eval_as_loop=False, 
                           careful_with_memory: bool = True,
                           new_prop: bool = False, properties: int = None):
        """Add spectra to a grid
        
        Takes an array of positions with 1 dimension less than the dimensionality 
        of the grid plus a function that is evaluated over the final dimension of
        the grid. The function is evaulated at axis value along the final axis.

        spec_function should take as its first argument an array of frequencies 
        (values of the Grid instances axis matching the spec_ax parameter), and as its
        second argument an array of shape n_objects x n_parameters containing any 
        additional function arguments for the spec_function. The parameters for each
        object are then passed as spec_function_arguments. By default, it is assumed
        that the function will be evaluated simultaneously for all objects, and
        return an array of shape (n_objects x spec_ax_length).
        
        Example spec_function:
        >>> # Gaussian spectral line w/ 100 km/s sigma, assumes axis is in GHz
        >>> # and line rest frequency is 115 GHz
        >>> def spec_func(axis, params):
        >>> ... lum = params[:,0]
        >>> ... redshift = params[:,1]
        >>> ... nu0 = 115/(1+redshift) # Redshift line center
        >>> ... nu_sig = 300 / 3e5 * nu0 # FWHM in GHz
        >>> ... # Reshape arrays so everything broadcasts together into the right shape:
        >>> ... lum = lum.reshape(-1,1)
        >>> ... nu_sig = nu_sig.reshape(-1,1)
        >>> ... nu0 = nu0.reshape(-1,1)
        >>> ... ax = axis.reshape(1,-1)
        >>> ... spec = lum / np.sqrt(2*np.pi*nu_sig**2) * np.exp(-0.5 * (ax-nu0)**2/nu_sig**2)
        >>> ... return spec
        The corresponding spec_function_arguments would be
        >>> spec_function_arguments = np.array([[lum1, redshift1],...,[lumn,redshiftn]])
        
        The optimal way to characterize a spectrum for this type of gridding is
        in terms of its cumulative value along spectral axis, which can then be
        evaluated at each bin edge and differenced to return an integrated 
        spectrum that will not depend on the steps along the grid or (for example)
        lose flux from narrow peaks that aren't well sampled by the Grid axis.
        If spec_function is written as a cumulative function, then set ``is_cumulative``
        to True, and the spectrum will be constructed by evaluating spec_function
        at the edges of the axis cells and then differencing. This is not the default
        as it is not common to write analytic formulae for the integral of a spectrum.

        Parameters
        ----------
        positions : array
            The positions of each object should be an n_objects x (n_dimensions-1)
            array
        spec_function : Callable
            Function that will be evaluated to compute the spectra of each object
        spec_function_args : array
            Array of shape n_objects x n_function_parameters that will be passed 
            to spec_function when evaluating spectra of each object
        spec_ax : int (optional)
            The axis along which the values are to be added, by default the final position
            axis is assumed
        eval_as_loop : bool (default = False)
            If True, the spec_function will be evaluated in a loop - one call per 
            object in the positions and spec_function_args arrays.
        is_cumulative : bool (default = False)
            If True, the spec_function returns the integral of the spectrum from 0 to
            a given point along the axis, and the final spectrum is calculated by 
            differencing evaluated values at the edge of each bin in the Grid.
        careful_with_memory : bool (default = True)
            If True, the input arrays of positions and spec_function_args will be 
            broken up into smaller chunks and added to the grid one chunk at a time.
            This is helpful for preventing memory overflows when processing large
            catalogs. The chunk size is determined so that the number of array 
            elements used to hold spectra is never larger than the number of array
            elements in the grid itself.
        new_prop : bool (optional)
            If True, this method will create a new entry along the poperty
            dimension of the grid rather than adding values on top of any existing values
            in the grid. Default is False.
        properties : int or list of ints (optional)
            If specified, the values will only be added to the specified property
            indices. If None, values will be added to all property indices.
            Ignored if new_props = True
        """

        if not hasattr(spec_function, '__call__'):
            raise ValueError("spec_function must have a __call__ method")

        if not self.grid_active:
            raise ValueError("data array for this Grid has not been initialized")
        
        if np.any(self.fourier_space):
            raise ValueError("Some axes are in fourier space, cannot add new properties in map space.")

        # Check that the grid has the specified spec_axis
        if spec_ax < 0: 
            spec_ax = self.n_dimensions+spec_ax
        if spec_ax>=self.n_dimensions or spec_ax<0:
            raise ValueError('specified spec_ax not valid')

        # If the positions array is empty don't need to do much
        if len(positions) == 0:
            if positions.ndim == 1:
                if new_prop:
                    new_grid = np.zeros(np.concatenate((self.n_pixels,[1])))
                    self.new_prop(new_grid)
            
            return

        # Otherwise make sure the positions and spec_function_arguments arrays are in the right shapes and match the shape of the grid
        positions = np.array(positions,ndmin=2)
        if positions.shape[0] > 0:
            if positions.shape[1] != self.n_dimensions-1:
                raise ValueError('positions (dim={}) should have one fewer dimensions than the grid (dim={})'.format(positions.shape[1],self.n_dimensions))

        if spec_function_arguments is None:
            spec_function_arguments = np.zeros((len(positions),0))
        else:
            spec_function_arguments = np.array(spec_function_arguments,ndmin=1)
        
        if spec_function_arguments.shape[0] != positions.shape[0]:
            raise ValueError('spec_function_arguments must have the same length as positions')
        if spec_function_arguments.ndim == 1:
            spec_function_arguments = spec_function_arguments[...,np.newaxis]

        # Determine if a loop is necessary
        if careful_with_memory:
            chunk_size = np.prod(np.delete(self.n_pixels,spec_ax)) # maximum number of spectral channels computed equals size of the grid
        else:
            chunk_size = len(positions)

        # Pick axis to use
        if is_cumulative:
            spec_axis_eval = self.axes[spec_ax]
        else:
            spec_axis_eval = self.axes_centers[spec_ax]

        # Loop is here to allow a version which minimizes memory usage.
        # default parameters will cause the loop to only happen once and
        # simultaneously evaluate all spectra
        for i in range(np.ceil(len(positions)/chunk_size).astype(int)):
            
            pi = positions[i*chunk_size:(i+1)*chunk_size]
            argi = spec_function_arguments[i*chunk_size:(i+1)*chunk_size]


            if eval_as_loop:
                values = []
                for ig in range(len(pi)):
                    values.append(spec_function(spec_axis_eval, argi[ig].reshape(1,-1)).flatten())
            else:
                values = spec_function(spec_axis_eval, argi)
            
            values = np.array(values,ndmin=2)

            if is_cumulative:
                values = np.diff(values, axis=1)

            if values.shape[1] != self.n_pixels[spec_ax]:
                raise ValueError("The length of the returned spectrum (l={}) should match the size of grid dimension {}".format(values.shape[1],spec_ax))
            if values.shape[0] != len(pi):
                raise ValueError("The number of returned spectra should match the number of positions")
        
            self.add_from_pos_plus_array(pi, values, new_prop=new_prop, properties=properties, ax=spec_ax)
            if new_prop: # After first loop need to change behavior
                new_prop = False
                properties = self.n_properties - 1


    def sum_properties(self,properties=None,in_place=True):
        """Sum values of different properties
        
        Parameters
        ----------
        properties : int or tuple of ints
            Indices of the grid properties to include in the sum. By 
            default all properties are used.
        in_place : bool (default = True)
            If True, the sum will be stored in a new property index of
            the current Grid instance. If False, a new grid containing
            the summed property is returned

        Returns
        -------
        Grid instance
            Either the original grid, with the new property added in the
            last property index (in_place=True, default) or a new grid
            containing only the summed property (in_place=False)
        """

        if not self.grid_active:
            raise ValueError("data array for this Grid has not been initialized")
        
        if properties is None:
            properties = np.arange(self.n_properties)
        
        properties = np.array(properties,ndmin=1)
        if not np.all(np.isin(properties,np.arange(self.n_properties))):
            raise ValueError("Some property indices do not correspond to property indices of the grid")
        
        sum_grid = np.expand_dims(np.sum(self.grid[...,tuple(properties)],axis=-1),-1)
        if in_place:
            self.add_new_prop(sum_grid)

            return self
            
        else:
            new_grid = self.copy_axes(n_properties=1)
            new_grid.init_grid()
            new_grid.grid = sum_grid

            return new_grid


    def sample(self,positions,properties=None):
        """Draw samples from the Grid at a list of positions
        
        This function takes a list of positions and returns the value of the
        grid cell closest to the specified position. Useful for simulating
        timestreams.

        Parameters
        ----------
        positions : array
            An array containg the positions at which to sample the grid. It
            should have a shape of (n_samples, n_dimensions) where n_dimensions
            is the number of dimensions of the grid. Repeats of the same
            position are allowed.
        properties : int or tuple of ints
            Indices of the grid properties to return. By default all properties
            are returned.

        Returns
        -------
        samples
            An array containing the property values at every requrested position.
            This array will have the shape (n_samples, n_properites) and is two
            dimensional even if only one property was requested.
        """

        if not self.grid_active:
            raise ValueError("data array for this Grid has not been initialized")
        
        # Select the properties to sample
        if properties is None:
            properties = np.arange(self.n_properties,dtype=int)
        else:
            properties = np.array(properties,ndmin=1,dtype=int)
            if np.any(properties > self.n_properties-1) or np.any(properties < 0):
                raise ValueError("Property index does not exist")
        properties = tuple(properties)

        # Make sure the positions array is in the right shape and matches the dimensions of the grid
        positions = np.array(positions,ndmin=1,copy=True)
        if positions.shape[0] > 0:
            if positions.shape[1] != self.n_dimensions:
                raise ValueError('positions (dim={}) does not have the right number of dimensions for the grid (dim={})'.format(positions.shape[1],self.n_dimensions))

        # Put positions into coordinate units
        positions -= self.center_point.reshape(1,self.n_dimensions)
        positions += self.side_length.reshape(1,self.n_dimensions)/2
        positions /= self.pixel_size.reshape(1,self.n_dimensions)
        positions = np.floor(positions).astype('int')

        # Find points outside the grid area, store them, and for now set their pixels to 0
        invalid_positions = np.nonzero(np.any(positions<0,axis=1) | np.any(positions/self.n_pixels.reshape(1,self.n_dimensions)>=1,axis=1))
        positions[invalid_positions] = 0

        positions_tuple = tuple([tuple(line) for line in positions.T])
        samples = self.grid[positions_tuple]
        samples[invalid_positions] = np.nan
        samples = samples[:,properties]

        # This is to make sure all samples have a sample number and a
        # property dimension.
        if samples.ndim == 1:
            samples = np.expand_dims(samples,-1)
        
        return samples


    def convolve(self,source,ax=None,in_place=True,pad=None):
        """Convolve this Grid with another Grid instance.
        
        Carries out convolution of this Grid's data array with the data array of
        another Grid instance specified by source. The source grid should
        typically be psf-like (see e.g. the PSF and Spectral PSF classes) but
        the method functionality is completely general.

        The axes along which to perform the convolution can be specified.

        Parameters
        ----------
        source : Grid instance
            A Grid instance containing the pattern to convolve with the grid.
        ax : int or tuple of ints
            The index or indices of the axis to convolve, each value should
            correspond to a spatial axis of the grid (so for a 3d grid 0, 1, or
            2 are valid).
        in_place : bool (default = True)
            If True, the convolved grid will overwrite the original data grid of
            this Grid instance. If False, a new Grid containing the convolved
            grid property is returned
        pad : int (optional)
            Specify a number of cells to zero pad onto the each side of the grid
            during the convolution.

        Returns
        -------
        convolvedgrid
            A Grid instance containing the convolved grid. Either this Grid 
            (in_place=True, default), or a new Grid instance (in_place=False)
        """

        # Check inputs
        if not self.grid_active:
            raise ValueError("No grid is initialized.")
        if np.any(self.fourier_space):
            raise ValueError("grid has some axes in Fourier space, convolution requires all axes be in map space")
        if not source.grid_active:
            raise ValueError("No source grid is initialized.")
        if self.is_power_spectrum:
            raise ValueError("This grid is a power spectrum, not a map")

        # If the grid we convolve with has higher dimension than self, raise an error
        if source.grid.ndim > self.grid.ndim:
            raise ValueError("source grid has too many dimensions")
        # Otherwise, if the grid we convolve with has lower dimensions that self,
        # we will pad its dimensions
        elif source.grid.ndim != self.grid.ndim:
            nsourcepad = self.grid.ndim - source.grid.ndim
            shape = np.concatenate((source.grid.shape[:-1],
                                    [1 for i in range(nsourcepad)],
                                    [source.grid.shape[-1]])) # The final axis is always the 'property' axis
        else:
            nsourcepad = 0
            shape = source.grid.shape

        if ax is None:
            ax = np.arange(self.n_dimensions-nsourcepad).astype('int')
            # ax = np.arange(self.n_dimensions-1).astype('int')
        else:
            ax = np.array(ax,ndmin=1,dtype='int')
            if np.any(ax>=self.n_dimensions) or np.any(ax<0):
                raise ValueError('specified axes not valid')

        # Pad the array
        if pad is not None:
            if not isinstance(pad,int):
                if len(pad) != len(ax):
                    raise ValueError("pad must have one entry for each axis specified in ax")
        
        if isinstance(pad,int):
            pad = np.array([pad for i in range(len(ax))]).astype(int)
        elif pad is None:
            pad = np.array([0 for i in range(len(ax))]).astype(int)

        # Do the dang thing
        if in_place:
            self.pad(ax,pad)
            self.grid = fftconvolve(self.grid,source.grid.reshape(shape),axes=ax,mode='same')
            self.pad(ax,-pad)

            return self

        else:
            convgrid = self.copy()

            convgrid.pad(ax,pad)
            convgrid.grid = fftconvolve(convgrid.grid,source.grid.reshape(shape),axes=ax,mode='same')
            convgrid.pad(ax,-pad)

            return convgrid


    # Should extend this method to work in Fourier space
    # This is not optimized for multiple axes simultaneously -
    # uses a fairly clunky loop.
    def down_sample(self,ax,factor,mode='average'):
        """Degrade the resolution of the grid along a specified axis
        by binning groups of pixels.

        This is only supported in real-space currently
        
        Parameters
        ----------
        ax : int or tuple of ints
            The index or indices of the axis to donwsample, each value should
            correspond to a spatial axis of the grid (so for a 3d grid 0, 1, or
            2 are valid).
        factor : int or tuple of ints
            Number of pixels to bin together along the axes specified.
        mode : 'average' or 'sum' (default='average')
            Specify whether bin values should be computed by summing or
            averaging groups of adjaced cells. Default is average
        """

        ax = np.array(ax,ndmin=1).astype(int)
        if np.any(ax>=self.n_dimensions) or np.any(ax<0):
            raise ValueError('specified axes not valid')
        
        factor = np.array(factor,ndmin=1).astype(int)
        if len(factor) == 1:
            factor = np.ones(len(ax))*factor[0]
        elif len(factor) != len(ax):
            raise ValueError('length of factor must be 1 or match the length of ax')
            
        for i in ax:
            if self.fourier_space[i]:
                raise ValueError("Downsampling not supported for Fourier space (see the pad method)")

        # Set the function to do the collapsing of cells
        if mode == 'average':
            use_function = np.mean
        elif mode == 'sum':
            use_function = np.sum
        else:
            raise ValueError("mode not recognized - must be 'average' or 'sum'.")

        for axi in ax:
            shape = np.concatenate((self.n_pixels,[self.n_properties]))
            shape[axi] = shape[axi]//factor[axi]

            # Downsample
            new_ax_centers = np.zeros(self.axes_centers[axi].shape)
            new_grid = np.zeros(shape)
            for i in range(shape[axi]):
                slice_ax = slice(i*factor[axi],(i+1)*factor[axi])
                slices = [slice(0,None) if j!=axi else slice_ax for j in range(self.n_dimensions+1)]
                slices_new = [slice(0,None) if j!=axi else i for j in range(self.n_dimensions+1)]

                if self.grid_active:
                    new_grid[slices_new] = use_function(self.grid[slices],axis=axi)
                new_ax_centers[i] = np.mean(self.axes_centers[axi][slice_ax])

            # Set up new grid
            if self.grid_active:
                self.grid = new_grid
            self.axes_centers[axi] = new_ax_centers
            self.axes[axi] = np.concatenate((new_ax_centers-(factor[axi]*self.pixel_size[axi])/2, [new_ax_centers[-1]+(factor[axi]*self.pixel_size[axi])/2]))
            self.pixel_size[axi] = self.pixel_size[axi]*factor[axi]
            self.n_pixels[axi] = self.n_pixels[axi]//factor[axi]
            self.side_length[axi] = np.ptp(self.axes[axi])
            self.center_point[axi] = np.min(self.axes[axi]) + self.side_length[axi]/2
            self.fourier_axes_centers[axi] = np.fft.fftshift(np.fft.fftfreq(self.n_pixels[axi],self.pixel_size[axi]))
            self.fourier_axes[axi] = _axis_edges(self.fourier_axes_centers[axi])


    def collapse_dimension(self,ax,in_place=True,weights=None,mode='average'):
        """Collapse one or more dimensions of a grid
        
        Parameters
        ----------
        ax : int or tuple of ints
            The index or indices of the axis to collapse, each value should
            correspond to a spatial axis of the grid (so for a 3d grid 0, 1, or
            2 are valid).
        in_place : bool (default = True)
            If True, the collapsed grid will overwrite the original data grid of
            this Grid instance. If False, a new Grid containing the collapsed
            grid is returned
        weights : array or list of arrays (optional)
            The weights to apply to each cell along the axis being collapsed. If
            multipl axes are specified in ax, a list should be specified here.
            If no values are given, uniform weights are assumed.
        mode : 'sum' or 'average'
            Determines whether grids are averaged or summed along the axis being
            collapsed.

        Returns
        -------
        collapsededgrid
            A Grid instance containing the collapsed grid. Either this Grid 
            (in_place=True, default), or a new Grid instance (in_place=False)
        """

        if not self.grid_active:
            raise ValueError("No grid is initialized.")

        # Check that the grid has the specified axes
        ax = np.array(ax,ndmin=1,dtype=int)
        if np.any(ax>=self.n_dimensions) or np.any(ax<0):
            raise ValueError('specified axes not valid')
        ax_keep = np.setdiff1d(np.arange(self.n_dimensions),ax)

        # Deal with weights
        # Weights can either be None, a 1d array that will be multiplied along
        # the axis being collapsed, a self.n_dimensions array that will be multiplied
        # against each position in the grid before collapsing, or a self.n_dimensions + 1
        # array that will be multiplied across the whole grid (allowing weights to varry
        # for different properties)
        if weights is None:
            weights = [np.ones(len(self.axes_centers[i])) for i in ax]
        elif len(ax) == 1:
            weights = [weights]

        if len(weights) != len(ax):
            raise ValueError("Must provide a weights array for each axis to collapse")
        else:
            for i in range(len(ax)):
                weights[i] = np.array(weights[i])
                if weights[i].ndim == 1:
                    if len(weights[i]) != len(self.axes_centers[ax[i]]):
                        raise ValueError("Weights for axis {} must match size of grid. {} // {}".format(ax[i],len(weights[i]),len(self.axes_centers[ax[i]])))
                    else:
                        shape = [1 for i in range(self.n_dimensions)]+[1]
                        shape[ax[i]] = len(weights[i])
                        weights[i] = weights[i].reshape(tuple(shape))
                elif weights[i].ndim == self.n_dimensions:
                    if weights[i].shape != self.grid.shape[:-1]:
                        raise ValueError("Weights must match shape of grid")
                    weights[i] = np.expand_dims(weights[i],-1)
                elif weights[i].ndim == self.n_dimensions+1:
                    if weights[i].shape != self.grid.shape:
                        raise ValueError("Weights must match shape of grid")
                    weights[i] = np.expand_dims(weights[i],-1)
                else:
                    raise ValueError("Weights for axis {} must match size of grid. {} // {}".format(ax[i],len(weights[i]),len(self.axes_centers[ax[i]])))
        
        if mode=='sum':
            norm = [1 for i in ax]
        elif mode=='average':
            norm = []
            for i in range(len(ax)):
                if weights[i].ndim == 1:
                    norm.append(np.sum(weights[i]))
                else:
                    norm.append(np.sum(weights[i],axis=ax[i]))
        else:
            raise ValueError("mode not recognized - options are 'sum','average'")

        # Do the dang thing
        if in_place:
            for i in range(len(ax)):
                self.grid = np.sum(self.grid*weights[i],axis=ax[i]) / norm[i]

            # Update grid
            self.n_pixels = self.n_pixels[ax_keep]
            self.axes = [self.axes[i] for i in range(self.n_dimensions) if i not in ax]
            self.axes_centers = [self.axes_centers[i] for i in range(self.n_dimensions) if i not in ax]
            self.fourier_axes = [self.fourier_axes[i] for i in range(self.n_dimensions) if i not in ax]
            self.fourier_axes_centers = [self.fourier_axes_centers[i] for i in range(self.n_dimensions) if i not in ax]
            
            self.axunits = self.axunits[ax_keep]
            self.fourier_axunits = self.fourier_axunits[ax_keep]
            self.fourier_space = self.fourier_space[ax_keep]

            self.n_dimensions = self.n_dimensions - len(ax)
            self.center_point = self.center_point[ax_keep]
            self.side_length = self.side_length[ax_keep]
            self.pixel_size = self.pixel_size[ax_keep]

            return self

        else:
            newgrid = self.copy()

            for i in range(len(ax)):
                newgrid.grid = np.sum(newgrid.grid*weights[i],axis=ax[i]) / norm[i]

            # Update grid
            newgrid.n_pixels = self.n_pixels[ax_keep]
            newgrid.axes = [self.axes[i] for i in range(self.n_dimensions) if i not in ax]
            newgrid.axes_centers = [self.axes_centers[i] for i in range(self.n_dimensions) if i not in ax]
            newgrid.fourier_axes = [self.fourier_axes[i] for i in range(self.n_dimensions) if i not in ax]
            newgrid.fourier_axes_centers = [self.fourier_axes_centers[i] for i in range(self.n_dimensions) if i not in ax]

            newgrid.axunits = self.axunits[ax_keep]
            newgrid.fourier_axunits = self.fourier_axunits[ax_keep]
            newgrid.fourier_space = self.fourier_space[ax_keep]

            newgrid.n_dimensions = self.n_dimensions - len(ax)
            newgrid.center_point = self.center_point[ax_keep]
            newgrid.side_length = self.side_length[ax_keep]
            newgrid.pixel_size = self.pixel_size[ax_keep]

            return newgrid


    def fourier_transform(self,ax=None,normalize=True):
        """Fourier transform the grid

        This method will fourier transform the grid along specified axes.
        If the grid is already in Fourier space along a given axis, the
        inverse fourier transform will be computed instead. These changes
        are applied directly to the grid.

        Parameters
        ----------
        ax : int or list of ints (optional)
            Indices of the axes to be transformed, if no axes are given
            the operation will be applied to all axes
        normalize : bool (optional)
            Determines whether the a normalization should be applied to the
            transform (the normalization is pixel_size[i] for forward
            transforms along axis i, and 1/pixel_size[i] for inverse transforms)
        """

        if not self.grid_active:
            raise ValueError("No grid is initialized.")
        if self.is_power_spectrum:
            raise ValueError("This grid is a power spectrum, not a map")

        if ax is None:
            ax = np.arange(self.n_dimensions).astype('int')
        else:
            ax = np.array(ax,ndmin=1,dtype='int')
            if np.any(ax>=self.n_dimensions) or np.any(ax<0):
                raise ValueError('specified axes not valid')

        # Determine whether this is a forward or inverse FT along
        # each axis
        ax_forward = []
        norm_forward = 1
        ax_backward = []
        norm_backward = 1
        for i in ax:
            if not self.fourier_space[i]:
                ax_forward.append(i)
                norm_forward *= self.pixel_size[i]
            else:
                ax_backward.append(i)
                norm_backward /= self.pixel_size[i]

        # Do the forward FFTs
        self.grid = np.fft.fftshift(np.fft.fftn(self.grid,axes=ax_forward),axes=ax_forward)
        if normalize:
            self.grid *= norm_forward
        for i in ax_forward:
            self.fourier_space[i] = True

        # Do the inverse FFTs
        self.grid = np.fft.ifftn(np.fft.ifftshift(self.grid,axes=ax_backward),axes=ax_backward)
        if normalize:
            self.grid *= norm_backward
        for i in ax_backward:
            self.fourier_space[i] = False


    # This doesn't currently carry over/transform units of the the grids
    def power_spectrum(self, cross_grid=None, ax=None, in_place=False, normalize=True):
        """Compute the power spectrum of a grid in map space

        The default behavior is to return a new grid instance containing the
        power spectrum, however if in_place is set to True, the original grid
        will be replaced with the power spectrum. Note that this procedure
        cannot be reversed.

        A cross-power spectrum can be computed by specifying a cross_grid -
        another grid instance to be taken as the second term in the cross-power.

        This function checks whether a grid is already in power spectrum form,
        and will not run if it is.

        Parameters
        ----------
        cross_grid : Grid instance, optional
            A second grid, which will be used to compute a cross-power spectrum
        ax : int or list of ints, optional
            Indices of the axes to be put into fourier space to compute the
            power spectrum. If no axes are given the operation will be applied
            to all axes. It does not matter what space the grid is initially
            in (coordinate of Fourier) - the method ensures that all axes
            specified in ax are transformed to Fourier space before computing
            a power spectrum. Any axes that must be transformed will be
            transformed back after the power spectrum is computed (unless 
            in_place is set to true, in which case the power spectrum grid
            is left in Fourier space).
        in_place : bool, optional
            If set to False (default) a grid instance containing the power
            spectrum will be returned. If set to True the existing grid will
            be overwritten by the power spectrum.
        normalize : bool, optional
            Determines whether the a normalization should be applied to the
            fourier transform. See fourier_transform method for details
        """

        if not self.grid_active:
            raise ValueError("No grid is initialized.")
        if self.is_power_spectrum:
            raise ValueError("This grid is already a power spectrum")

        # Checks on the cross_grid
        if cross_grid is self:
            cross_grid = None
        if not cross_grid is None:
            if not isinstance(cross_grid,Grid):
                raise ValueError("cross_grid is not a valid grid instance")
            if not cross_grid.grid_active:
                raise ValueError("No cross_grid is initialized.")
            if cross_grid.is_power_spectrum:
                raise ValueError("This cross_grid is already a power spectrum")
            if cross_grid.n_dimensions != self.n_dimensions:
                raise ValueError("grid n_dimensions ({}) does not match cross_grid ({})".format(self.n_dimensions,cross_grid.n_dimensions))
            if cross_grid.n_properties != 1 and self.n_properties != 1:
                if cross_grid.n_properties != self.n_properties:
                    raise ValueError("grid n_properties ({}) is not compatible cross_grid ({})".format(self.n_properties,cross_grid.n_properties))

        # Determine which axes are in Fourier space already and which
        # need to be transformed
        if ax is None:
            ax = np.arange(self.n_dimensions).astype('int')
        else:
            ax = np.array(ax,ndmin=1,dtype='int')
            if np.any(ax>=self.n_dimensions) or np.any(ax<0):
                raise ValueError('specified axes not valid')

        transform_list = []
        for i in ax:
            if not self.fourier_space[i]:
                transform_list.append(i)
        for i in np.setdiff1d(np.arange(self.n_dimensions).astype('int'),ax):
            if self.fourier_space[i]:
                transform_list.append(i)

        # Make appropriate fourier_transform call
        self.fourier_transform(transform_list,normalize=normalize)

        # Same for cross_grid
        if not cross_grid is None:
            cross_transform_list = []
            for i in ax:
                if not cross_grid.fourier_space[i]:
                    cross_transform_list.append(i)
            for i in np.setdiff1d(np.arange(self.n_dimensions).astype('int'),ax):
                if cross_grid.fourier_space[i]:
                    cross_transform_list.append(i)

            # Make appropriate fourier_transform call
            cross_grid.fourier_transform(cross_transform_list,normalize=normalize)

        # Figure out the units to assign
        unit_end = ''
        units,n = np.unique(self.axunits[ax][self.axunits[ax] != None],return_counts=True)

        for i in range(len(units)):
            if n[i] > 1:
                unit_end += (' '+units[i]+'^'+str(n[i]))
            else:
                unit_end += (' '+units[i])

        # If in_place, then just overwrite the grid with the power spectrum
        if in_place:
            if not cross_grid is None:
                self.grid = (self.grid * cross_grid.grid.conj()).real
                # for i in range(len(self.gridunits)):
                #     if not self.gridunits[i] is None:
                #         self.gridunits[i] += cross_grid.gridunits[i] + unit_end
            else:
                self.grid = (self.grid * self.grid.conj()).real
                # for i in range(len(self.gridunits)):
                #     if not self.gridunits[i] is None:
                #         self.gridunits[i] += '^2' + unit_end

            self.is_power_spectrum = True
            # for i in range(len(self.gridunits)):
            #     if not self.gridunits[i] is None:
            #         self.gridunits += unit_end

            # Invert fourier transforms
            if not cross_grid is None:
                cross_grid.fourier_transform(cross_transform_list,normalize=normalize)

            return self

        # Else we need to create a new grid instance to hold the power
        # spectrum and return it, plus invert the fourier transforms
        # to the original grid
        else:
            powspec = Grid(self.n_properties,self.center_point,self.side_length,self.pixel_size,self.axunits,self.gridunits)

            # TO DO!! Tidy up the units
            # for i in range(len(powspec.gridunits)):
            #     if not powspec.gridunits is None:
            #         powspec.gridunits += unit_end

            if not cross_grid is None:
                powspec.grid = (self.grid * cross_grid.grid.conj()).real
                # for i in range(len(self.gridunits)):
                #     if not self.gridunits[i] is None:
                #         self.gridunits[i] += cross_grid.gridunits[i] + unit_end
            else:
                powspec.grid = (self.grid * self.grid.conj()).real
                # for i in range(len(self.gridunits)):
                #     if not self.gridunits[i] is None:
                #         self.gridunits[i] += '^2' + unit_end

            powspec.grid_active = True
            powspec.is_power_spectrum = True
            powspec.n_objects = self.n_objects
            powspec.fourier_space = np.copy(self.fourier_space)
            powspec.fourier_space[:] = True

            # Invert fourier transforms
            if not cross_grid is None:
                cross_grid.fourier_transform(cross_transform_list,normalize=normalize)
            self.fourier_transform(transform_list,normalize=normalize)

            return powspec


    def spherical_average(self,ax=None,center=None,shells=None,binmode='linear',weights=None,return_std=False,biased_std=True,return_n=False):
        """Compute averages in spherical shells along the grid axes
        
        This function only works if all axes are in real space or all axes are
        in Fourier space. Mixtures of the two are not supported.

        Parameters
        ----------
        ax : int or tuple of ints
            The axes or axes in which the data should be averaged. By default
            all axes will be averaged resulting in a 1d array.
        center : tuple of floats (optional)
            The location of the center of the sphere used for averaging. By
            default the origin of the grid is used.
        shells : int or array (optional)
            If an int is given, the average will be carried out in this number
            of shells. If an array is given, it will be treated as the shell
            edges. If no value is given, the shells will be automatically
            computed.
        binmode : 'linear' or 'log'
            Whether spherical shells should be linearly or logarithmically
            spaced. This is ignored if the shells are specified in the shells
            keyword.
        weights : array (optional)
            An array of the same number of dimensions as the grid (including or
            excluding the properties dimension) specifying the weights to give
            each grid cell when computing the average. If not specified all
            cells receive uniform weight.
        return_std : bool (default=False)
            If set to True the standard deviation will be returned in addion to
            the average
        biased_std : bool (default=True)
            If True the biased standard deviation is computed when return_std
            is set to True
        return_n : bool (default=False)
            If set to True the number of cells used to compute the average in each
            shell is also returned

        Returns
        -------
        axes
            The axes describing the average - if averaged over all dimensions this 
            will simply be the shells
        spherical_average
            The average values in each shell
        std (optional)
            The standard deviation in each shell
        n (optional)
            The number of grid cells that fell in each shell
        """

        if not self.grid_active:
            raise ValueError("No grid is initialized.")

        # If axis isn't specified use all of them
        if ax is None:
            ax = np.arange(self.n_dimensions).astype(int)
        # Otherwise check that the grid has the specified axes
        else:
            ax = np.array(ax,ndmin=1,dtype=int)
            if np.any(ax>=self.n_dimensions) or np.any(ax<0):
                raise ValueError('specified axes not valid')

        # Check that all axes are in either real or fourier space
        if not np.all(self.fourier_space[ax]==True) and not np.all(self.fourier_space[ax]==False):
            raise ValueError("Averaged axes must all be in either real or fourier space")
        
        # Select correct axes to define 
        if np.all(self.fourier_space[ax]==True):
            axvals = self.fourier_axes_centers
            axfourier = True
        else:
            axvals = self.axes_centers
            axfourier = False

        # If center isn't specified, use the center of the grid (spatial coordinates), or zero (fourier coordinates)
        if center is None:
            if axfourier:
                center = np.zeros(self.n_dimensions)
            else:
                center = np.copy(self.center_point)
        # Otherwise make sure it has the same dimensions as the grid
        else:
            center = np.array(center,ndmin=1)
            if len(center) != self.n_dimensions:
                raise ValueError("center dimensions don't match grid: center Nd={}, grid Nd={}".format(len(center),self.n_dimensions))
        
        # If weights isn't specified, use uniform weights
        if weights is None:
            weights = np.ones(1).reshape([1 for i in range(self.grid.ndim)])
        # Make sure weights has a shape compatible with grid
        else:
            weights = np.array(weights)
            if weights.ndim < self.grid.ndim:
                s = weights.shape
                for i in range(self.grid.ndim-weights.ndim):
                    s += (1,)
                weights = weights.reshape(s)
            for sw,sg in zip(weights.shape,self.grid.shape):
                if sw != sg and sw != 1:
                    raise ValueError("weights shape not compatible with grid: weights cast to {}, grid shape {}".format(weights.shape,self.grid.shape))

        # Compute the radius of cells using only the specified axes
        rad = np.zeros(self.grid.shape)
        for i_ax in ax:
            s = np.ones(self.n_dimensions+1,dtype=int)
            s[i_ax] = self.n_pixels[i_ax]
            rad += ((axvals[i_ax]-center[i_ax]).reshape(s))**2
        rad = np.sqrt(rad)
        rad = rad.flatten()

        # Set up shells
        if shells is None or np.issubdtype(type(shells), np.integer):

            bin_min = np.min(rad)
            bin_max = np.max(rad)
            bin_max = bin_max + (bin_max-bin_min)/1e10 # Make max slightly larger to ensure highest data point fits

            # case 1: no specifications given use min and max of axes
            if shells is None:
                bin_n = 10
            
            # Case 2: number of shells specified
            elif np.issubdtype(type(shells), np.integer):
                if shells<1:
                    raise ValueError("Positive number of shells required")
                bin_n = shells
            
            # Create shells in linear or log space
            if binmode == 'linear':
                shells = np.linspace(bin_min,bin_max,bin_n+1)
            elif binmode == 'log':
                if bin_min <=0:
                    bin_min = np.min(rad[rad>0])
                shells = np.logspace(np.log10(bin_min),np.log10(bin_max),bin_n+1)
            else:
                raise ValueError("bin mode must be in ['log','linear'].")
        
        # case 3: shells are specified by user
        else:
            shells = np.array(shells)
            if np.any(np.diff(shells)<=0):
                raise ValueError("shells must be strictly increasing")

        # Determine shape and axes of averaged data
        keep_axes = np.ones(self.n_dimensions,dtype=bool)
        keep_axes[ax] = False
        shape = [len(shells)-1] + [self.n_pixels[i_ax] for i_ax in range(self.n_dimensions) if keep_axes[i_ax]] + [self.n_properties]
        binned_shape = tuple(shape)
        binned_axes = [shells] + [axvals[i_ax] for i_ax in range(self.n_dimensions) if keep_axes[i_ax]] + [np.arange(self.n_properties)]

        # Determine the correct bin for each cell - digitize data into shells
        # and then get indices of dimensions that aren't being flattened
        bins_for_cells = np.digitize(rad,shells) - 1
        good_inds = np.nonzero((bins_for_cells>-1) & (bins_for_cells<len(shells)-2))
        coords_for_cells = np.indices(self.grid.shape)
        coords_for_cells = [bins_for_cells] + \
                           [coords_for_cells[i_ax].flatten() for i_ax in range(self.n_dimensions) if keep_axes[i_ax]] + \
                           [coords_for_cells[-1].flatten()]
        coords_for_cells = np.array(coords_for_cells).T[good_inds] # This is now an n_cells x n_dimensions (of binned data) array

        # Bin the data
        if axfourier and not self.is_power_spectrum:
            dtype_data = complex
        else:
            dtype_data = float


        # add.at needed to add items to a given index more than once in a single operation.
        # tuple(np.split(coords_for_cells,coords_for_cells.shape(1),axis=1)) creates 
        # a tuple ([cell0_x,cell1_x,...],[cell0_y,cell1_y,...],...) specifying the grid 
        # coordinates where each cell should be added
        binned_averages = np.zeros(binned_shape,dtype=dtype_data)
        np.add.at(binned_averages,tuple(np.split(coords_for_cells,coords_for_cells.shape[1],axis=1)),(self.grid*weights).flatten()[good_inds].reshape(-1,1))

        binned_weights = np.zeros(binned_shape,dtype=int)
        np.add.at(binned_weights,tuple(np.split(coords_for_cells,coords_for_cells.shape[1],axis=1)),(np.ones(self.grid.shape)*weights).flatten()[good_inds].reshape(-1,1))

        binned_averages /= binned_weights

        # Count samples if requested
        if return_n:
            binned_n = np.zeros(binned_shape,dtype=int)
            np.add.at(binned_n,tuple(np.split(coords_for_cells,coords_for_cells.shape[1],axis=1)),np.ones(self.grid.size)[good_inds].reshape(-1,1))

        # Compute standard deviation if requested
        if return_std:

            good_inds_grid = np.nonzero((bins_for_cells.reshape(self.grid.shape)>-1) & (bins_for_cells.reshape(self.grid.shape)<len(shells)-2))
            means = binned_averages[tuple(np.split(coords_for_cells,coords_for_cells.shape[1],axis=1))].flatten()
            std_vals = np.copy(self.grid)
            std_vals[good_inds_grid] = std_vals[good_inds_grid] - means
            std_vals = std_vals**2 * weights

            binned_std = np.zeros(binned_shape,dtype=dtype_data)
            np.add.at(binned_std,tuple(np.split(coords_for_cells,coords_for_cells.shape[1],axis=1)),std_vals.flatten()[good_inds].reshape(-1,1))
            
            if biased_std:
                binned_std = np.sqrt(binned_std/binned_weights)
            else:
                binned_weights2 = np.zeros(binned_shape,dtype=int)
                np.add.at(binned_weights2,tuple(np.split(coords_for_cells,coords_for_cells.shape[1],axis=1)),(np.ones(self.grid.shape)*weights**2).flatten()[good_inds].reshape(-1,1))

                binned_std = np.sqrt(binned_std/(binned_weights-binned_weights2/binned_weights))

        if return_std and return_n:
            return binned_axes, binned_averages, binned_n, binned_std
        elif return_std:
            return binned_axes, binned_averages, binned_std
        elif return_n:
            return binned_axes, binned_averages, binned_n
        return binned_axes, binned_averages


    @pltdeco
    def visualize(self, ax=[0,1], slice_indices='sum', property=0, figsize=(5,5), axkws={}, plotkws={}):
        """Quick grid visualizations

        Plots a 2D representation of the grid
        
        Parameters
        ----------
        ax : list of two axes
            List of len 2 specifying the 2 axes to plot along, other axes will
            either be collapsed via a sum, or shown at a single slice.
        slice_indices : list or 'sum'
            List containing indices at which to extract values of the other
            spatial axes of the grid. If 'sum' is given, the other axes will be
            collapsed by summing over all indices.
        property : int (default=0)
            The index of the property to plot. Shows the property in the 0 index
            by 0.
        figsize : tuple
            Size of the figure
        axkws : dict, optional
            A dictionary of keyword args and values that will be fed to ax.set()
            when creating the plot axes
        plotkws : dict, optional
            A dictionary of keyword args and values that will be fed to
            plt.pcolor() when creating the plot data        
        """

        if not self.grid_active:
            raise ValueError("data array for this Grid has not been initialized")
        
        if self.n_dimensions < 2:
            raise ValueError("visualize only works for 2D or larger maps")

        fig, plot = plt.subplots(figsize=figsize)
        plot.set(xlabel='Axis {} [{}]'.format(ax[0],self.axunits[ax[0]]))
        plot.set(ylabel='Axis {} [{}]'.format(ax[1],self.axunits[ax[1]]))
        plot.set(**axkws)
        
        if len(ax) != 2:
            raise ValueError("ax must specify exactly two dimensions to plot")

        remaining_ax = [i for i in range(self.n_dimensions) if i not in ax]
        if slice_indices == 'sum':
            image = np.sum(self.grid,axis=tuple(remaining_ax))[:,:,property]
        elif len(slice_indices) != len(remaining_ax):
            raise ValueError("Slice index must be specified for all ax not plotted")
        else:
            slices = []
            slice_idx = 0
            for i in range(self.n_dimensions):
                if i in ax:
                    slices.append(slice(0,None))
                else:
                    slices.append(slice_indices[slice_idx])
                    slice_idx += 1
            image = self.grid[tuple(slices)][:,:,property]

        cb=plot.pcolor(self.axes[ax[0]],self.axes[ax[1]],image.real.T,**plotkws)
        fig.colorbar(cb)
        plt.show()


    @pltdeco
    def animate(self, save=None, prop_ind=0, slide_dim=2, face_dims=[0,1], i0=0, still=False, logscale=False, minrescale=1, maxrescale=1):
        """Animated stepthrough of the grid for visualizing 3d data"""

        if not self.grid_active:
            raise ValueError("data array for this Grid has not been initialized")
        
        if self.n_dimensions < 3:
            raise ValueError("animate only works for 3D or larger maps")

        # Check the grid is in coordinate space
        if np.any(self.fourier_space):
            warnings.warn("Some dimensions in fourier space. All values will be cast to real.")

        # handle property selection
        if prop_ind >= self.n_properties:
            raise ValueError("property index too large")

        # handle dimension selection
        if slide_dim >= self.n_dimensions:
            raise ValueError("slide_dim index too large")
        face_dims = np.array(face_dims,ndmin=1)
        if np.any(face_dims >= self.n_dimensions):
            raise ValueError("face_dims index too large")
        if len(face_dims) != 2:
            raise ValueError("Specify 2 dimensions for face_dims")

        flatten_dims = np.setdiff1d(np.arange(self.n_dimensions),np.concatenate(([slide_dim],face_dims)))

        if self.fourier_space[slide_dim]:
            ax0 = self.fourier_axes_centers[slide_dim]
        else:
            ax0 = self.axes_centers[slide_dim] - self.pixel_size[slide_dim]/2
        slide_ax = np.copy(ax0)
        ax0 = _axis_edges(np.copy(ax0))

        if self.fourier_space[face_dims[0]]:
            ax1 = self.fourier_axes_centers[face_dims[0]]
        else:
            ax1 = self.axes_centers[face_dims[0]]
        ax1 = _axis_edges(np.copy(ax1))

        if self.fourier_space[face_dims[1]]:
            ax2 = self.fourier_axes_centers[face_dims[1]]
        else:
            ax2 = self.axes_centers[face_dims[1]]
        ax2 = _axis_edges(np.copy(ax2))

        # Determine property range allowed
        if logscale:
            vmax = np.log10(np.amax(self.grid[...,prop_ind]))
            vmin = vmax-10
        else:
            vmin = minrescale*np.amin(self.grid[...,prop_ind])
            vmax = maxrescale*np.amax(self.grid[...,prop_ind])

        # Set up plots
        figure = plt.figure(figsize=(10,5))
        title = plt.suptitle('')

        plot_edge = plt.subplot(121)
        plot_edge.set_title('Summed Along Axis {}'.format(face_dims[0]))
        # plot_edge.set(xlabel='Axis {}'.format(slide_dim),ylabel='Axis {}'.format(face_dims[1]))

        map_edge = np.sum(self.grid[...,prop_ind].real,axis=tuple(np.concatenate((flatten_dims,[face_dims[0]]))))
        if slide_dim < face_dims[1]:
            map_edge = map_edge.T
        if logscale:
            map_edge = np.log10(map_edge)
            map_edge[~np.isfinite(map_edge)] = vmin
        edge = plot_edge.pcolormesh(ax0, ax2,
                                    map_edge,
                                    cmap='inferno',
                                    vmin=np.amin(map_edge), vmax=np.amax(map_edge))
        box, = plot_edge.plot([],[],color='red',lw=.75)
        box.set_data([ax0[i0],ax0[i0],ax0[i0+1],ax0[i0+1],ax0[i0]],
                     [ax2[0],ax2[-1],ax2[-1],ax2[0],ax2[0]])

        plot_face = plt.subplot(122)
        plot_face.set_title('Axis {} = {:.2f}'.format(slide_dim,slide_ax[0]))
        # plot_face.set(xlabel='Axis {}'.format(face_dims[0]),ylabel='Axis {}'.format(face_dims[1]))

        map_face = np.take(np.sum(self.grid[...,prop_ind].real,axis=tuple(flatten_dims)),indices=i0,axis=slide_dim)
        if face_dims[0] < face_dims[1]:
            map_face = map_face.T
        if logscale:
            map_face = np.log10(map_face)
            map_face[~np.isfinite(map_face)] = vmin
        face = plot_face.pcolormesh(ax1, ax2,
                                    map_face,
                                    cmap='inferno',
                                    vmin=vmin, vmax=vmax)

        nsteps = len(slide_ax)
        def animate(i):
            plot_face.set_title('Axis {} = {:.2f}'.format(slide_dim,slide_ax[i]))
            box.set_data([ax0[i],ax0[i],ax0[i+1],ax0[i+1],ax0[i]],
                         [ax2[0],ax2[-1],ax2[-1],ax2[0],ax2[0]])
            map_face = np.take(np.sum(self.grid[...,prop_ind].real,axis=tuple(flatten_dims)),indices=i,axis=slide_dim)
            if face_dims[0] < face_dims[1]:
                map_face = map_face.T
            if logscale:
                map_face = np.log10(map_face)
                map_face[~np.isfinite(map_face)] = vmin
            face.set_array(map_face.ravel())

        if not still:
            anim = animation.FuncAnimation(figure,animate,frames=nsteps,interval=500,blit=False)
            if save != None:
                anim.save(save+'.mp4')
        else:
            if save != None:
                plt.savefig(save)

        plt.show()
        return anim



###############################################################################
##### SECTION 2: Warppers #####################################################
###############################################################################

class Gridder(Grid):
    def __init__(self,
                 positions,
                 values = None,
                 center_point = None,
                 side_length = None,
                 pixel_size = 1,
                 axunits = None,
                 gridunits = None,
                 setndims = None):
        """Put properties into a grid

        Only required argument is positions - an array of object positions. If
        no other arguments are specified, code will return a grid of counts in
        cells and will to construct a grid based on the extremal values of the
        position list and a pixel edge length of 1 - this will work reasonably
        for a densely populated grid, but may get poor results if the positions
        don't sample the full volume you're trying to represent.

        To specify a grid provide the center_point, side_length, and pixel_size.

        To grid a property instead of number counts, specify the values
        parameter with an array of values equal in length to positions.

        Parameters
        ----------
        positions : array
            The positions of each object should be an n_objects x n_dimensions
            array
        values : array (optional)
            Values to grid, an arbitrary number of properties can be specified
            and gridded independently. Should be an n_objects x n_properties
        center_point : array (optional)
            The center of the grid, must match the number of dimensions
            specified in positions. If left unspecified, it will be set to the
            point half way between the largest and smallest positions value in
            each dimension
        side_length : float or array (optional)
            The length of the box edges. Should either be a single value or an
            array with the same number of elements as the dimensions specified
            in positions. If left unspecified, it will be set to slightly larger
            than the stretch between the largest and smallest position in each
            dimension. If this is not an integer multiple of the pixel_size, it
            will be increased to the next integer multiple.
        pixel_size : float or array (optional)
            The length of the pixel edges. Should either be a single value or an
            array with the same number of elements as the dimensions specified
            in positions. Default is 1 in each dimension.
        axunits : str or array (optional)
            The units of each axis. Should either be a single value or an array
            with the same number of elements as the dimensions specified in
            positions
        gridunits : str or array (optional)
            The units of the grid. Should either be a single value or an array
            with the same number of elements as the number properties
        setndims : int (optional)
            The number of dimensions of the object positions - the program tries
            to determine this by default, but may be useful if the positions
            array is empty.
        """

        # Handle position inputs:
        positions = np.array(positions,ndmin=1,copy=True)
        if positions.ndim == 1:
            positions = positions.reshape((positions.shape[0],1))

        n_objects = positions.shape[0]

        if setndims is None:
            if n_objects == 0:
                raise ValueError('positions has length zero, and no dimensions are specified')
            n_dimensions = positions.shape[1]
        else:
            if not isinstance(setndims,int):
                raise ValueError('setndims must have type int')
            if n_objects == 0:
                n_dimensions = setndims
            elif setndims != positions.shape[1]:
                raise ValueError('setndims does not match the shape of the positions array')
            else:
                n_dimensions = positions.shape[1]

        # Handle value inputs:
        # If values is set to none, we'll count objects in cells
        if values is None:
            values = np.ones((n_objects,1),dtype='int')
        else:
            values = np.array(values,ndmin=1,copy=True)
            if values.ndim == 1:
                values = values.reshape((values.shape[0],1))
        n_properties = values.shape[1]

        if values.shape[0] != n_objects:
            raise ValueError("position and values array do not have equal length.")

        # Handle center_point inputs:
        if center_point is None or side_length is None:
            if n_objects == 0:
                raise ValueError("No objects provided, cannot fit a grid")
            mins = np.amin(positions,axis=0)
            ptps = np.ptp(positions,axis=0)

        if center_point is None:
            center_point = mins + ptps/2
            center_point[ptps==0] = mins[ptps==0] # In case dimension is flat
        else:
            center_point = np.array(center_point,ndmin=1,copy=True)
            if len(center_point) != n_dimensions:
                raise ValueError("center_point don't match data dimensionality")

        # Handle side_length inputs:
        if side_length is None:
            side_length = ptps * 1.0000001 # Expand slightly so both extrema get gridded
            side_length[ptps==0] = 1 # In case dimension is flat
        else:
            side_length = np.array(side_length,ndmin=1,copy=True)
            if len(side_length) == 1:
                side_length = np.ones(n_dimensions)*side_length[0]
            if len(side_length) != n_dimensions:
                raise ValueError("side_length don't match data dimensionality")

        super().__init__(n_properties,center_point,side_length,pixel_size,axunits,gridunits)
        super().init_grid()
        super().add_from_cat(positions,values)


class LoadGrid(Grid):
    def __init__(self,path):
        """Generate a Grid instance by loading data saved by the Grid.save method
        
        Parameters
        ----------
        path : str
            filename from which the data is to be loaded
        """

        input = np.load(path,allow_pickle=True)

        super().__init__(n_properties=input['n_properties'],
                         center_point=input['center_point'],
                         side_length=input['side_length'],
                         pixel_size=input['pixel_size'],
                         axunits=input['axunits'],
                         gridunits=input['gridunits'])

        if input['grid_active']:
            self.grid_active = True
            self.grid = input['grid']
            self.n_objects = input['n_objects']
        self.fourier_space = input['fourier_space']
        self.is_power_spectrum = input['is_power_spectrum']


def gridder_function(positions, values = None, center_point = None, side_length = None, pixel_size = 1, setndims = None):
    """Wrapper for Gridder to simply return the grid and axes - see gridder docs

    Returns
    -------
    grid : n_dimensions + 1 dimensional array
        The gridded values with the first n_dimensions axes corresponding
        to the positions, and the final axis indexing the different
        properties that have been gridded.
    axes : list of n_dimensions arrays
        The physical values along each dimension of the grid
    """

    gridder_instance = Gridder(positions, values, center_point, side_length, pixel_size, setndims=setndims)
    return gridder_instance.grid, gridder_instance.axes



###############################################################################
##### SECTION 3: Beams, Masks, PSFs, etc ######################################
###############################################################################

class PSF(Grid):
    def __init__(self,fwhm,pixel_size,side_length=None,axunits=None,norm='area'):
        """Create a grid containing a Nd Gaussian PSF

        The number of dimensions is inferred from the length 
        of the fwhm input.
        
        Parameters
        ----------
        fwhm : float or tuple of floats
            The full width at half half maximum of the PSF
            in each dimension.
        pixel_size : float or array
            The length of the pixel edges. Should either be a single value or an
            array with the same number of elements as fwhm.
        side_length : float or tuple of floats (optional)
            The length along each spatial dimension of the grid. If a single
            value is given, all dimensions will be assumed to have the same
            length. Otherwise the length of the tuple should match the length of
            the fwhm. If no value is specified, a side length of 6 times the 
            FWHM is used.
        axunits : str or tuple of str (optional)
            A string or tuple of strings specifying the units of the spatial
            axes.
        norm : 'area' or 'peak'
            Determines whether the PSF is normalized to have a peak value of 
            1 or a total area/volume of 1.
        """

        n_properties = 1

        # Check inputs
        if norm not in ['area','peak']:
            raise ValueError("norm option not recognized")

        self.fwhm = np.array(fwhm,ndmin=1,copy=True)

        n_dimensions = self.fwhm.shape[0]
        center_point = np.zeros(n_dimensions)

        if not side_length is None:
            side_length = np.array(side_length,ndmin=1)
            if len(side_length) == 1:
                side_length = np.ones(n_dimensions) * side_length
            elif len(side_length) != n_dimensions:
                raise ValueError("side_lengths does not have same dimensionality as fwhm")
        else:
            side_length = 6*self.fwhm

        pixel_size = np.array(pixel_size,ndmin=1,copy=True)
        if len(pixel_size) == 1:
            pixel_size = np.ones(n_dimensions) * pixel_size
        elif len(pixel_size) != n_dimensions:
            raise ValueError("pixel_size does not have same dimensionality as fwhm")

        super().__init__(n_properties,center_point,side_length,pixel_size,axunits,gridunits=None)

        # Make the grid
        sig = self.fwhm / (2*np.sqrt(2*np.log(2)))
        psfnd = np.ones(self.n_pixels)
        for i in range(len(self.axes)):
            psf1d = np.exp(-self.axes_centers[i]**2 / (2*sig[i]**2))
            shape = np.ones(self.n_dimensions,dtype='int')
            shape[i] = len(psf1d)
            psfnd *= psf1d.reshape(shape)
        shape = np.concatenate((psfnd.shape,[1]))
        self.grid = psfnd.reshape(shape)

        if norm == 'area':
            self.grid /= np.sum(self.grid)

        self.grid_active = True
        self.n_objects = 0


class SpectralPSF(Grid):
    def __init__(self,spec_axis,spec0,fwhm0,pixel_size,side_length=None,axunits=None,norm='area',exp=1):
        """Create a grid containing a (N-1)d Gaussian PSF that varies 
        in width along one additional axis

        For creating Gaussian beams that varry as a function of wavelength/
        frequency.
        
        Parameters
        ----------
        spec_axis : array
            The central wavelengths/frequencies of each channel for which the 
            PSF should be computed. This should be uniformly spaced.
        spec0 : float
            The wavelength/frequency at which the FWHM (fwhm0) will be specified
        fwhm0 : float or tuple of floats
            The beam FWHM at spec0. A different value may be specified for each 
            dimension.
        pixel_size : float or array
            The length of the pixel edges in the spatial dimensions. Should either 
            be a single value or an array with the same number of elements as fwhm0.
        side_length : float or tuple of floats (optional)
            The length along each spatial dimension of the grid. If no value is 
            specified, a side length of 6 times the FWHM is used.
        axunits : str or tuple of str (optional)
            A string or tuple of strings specifying the units of the spatial
            axes. The final value should be the units for the spectral axis.
        norm : 'area' or 'peak'
            Determines whether the PSF is normalized to have a peak value of 
            1 or a total area/volume of 1. Each spectral slice is normalized
            independently.
        exp : float (default=1)
            The power of spec0/spec (either nu0/nu or lambda0/lambda) by which 
            the FWHM varries. Default is 1 - appropriate for a spectral axis in
            frequency units for a typical Gaussian beam.
        """
        n_properties = 1

        # Check inputs
        if norm not in ['area','peak']:
            raise ValueError("norm option not recognized")

        self.fwhm0 = np.array(fwhm0,ndmin=1,copy=True)
        spec_axis = np.array(spec_axis,ndmin=1,copy=True)

        n_dimensions = self.fwhm0.shape[0] + 1
        center_point = np.zeros(self.fwhm0.shape[0])
        center_point = np.concatenate((center_point,[np.ptp(spec_axis)/2+np.amin(spec_axis)]))

        if not side_length is None:
            side_length = np.array(side_length,ndmin=1)
            if len(side_length) == 1:
                side_length = np.ones(n_dimensions-1) * side_length
            elif len(side_length) != n_dimensions-1:
                raise ValueError("side_lengths does not have same dimensionality as fwhm0")
        else:
            side_length = 3*self.fwhm0 * np.amax(spec0/spec_axis)
        side_length = np.concatenate((side_length,[np.ptp(spec_axis)+np.abs(spec_axis[1]-spec_axis[0])]))

        pixel_size = np.array(pixel_size,ndmin=1,copy=True)
        if len(pixel_size) == 1:
            pixel_size = np.ones(n_dimensions-1) * pixel_size
        elif len(pixel_size) != n_dimensions-1:
            raise ValueError("pixel_size does not have same dimensionality as fwhm0")
        pixel_size = np.concatenate((pixel_size,[np.abs(spec_axis[1]-spec_axis[0])]))

        super().__init__(n_properties,center_point,side_length,pixel_size,axunits,gridunits=None)
        self.axes[-1] = spec_axis

        # Make the grid
        sig = self.fwhm0 / (2*np.sqrt(2*np.log(2)))
        psfnd = np.ones(self.n_pixels)
        for i in range(len(self.axes)-1):
            xx,ss = np.meshgrid(self.axes_centers[i],spec_axis,indexing='ij')
            psf2d = np.exp(-xx**2 / (2*(sig[i]*(spec0/ss)**exp)**2))
            shape = np.ones(self.n_dimensions,dtype='int')
            shape[i] = len(self.axes_centers[i])
            shape[-1] = len(spec_axis)
            psfnd *= psf2d.reshape(shape)
        shape = np.concatenate((psfnd.shape,[1]))
        self.grid = psfnd.reshape(shape)

        if norm == 'area':
            axes = tuple(np.arange(self.n_dimensions-1,dtype='int'))
            sums = np.sum(self.grid,axis=axes)
            shape = np.concatenate((np.ones(self.n_dimensions-1,dtype='int'),[len(sums),1]))
            self.grid /= sums.reshape(shape)

        self.grid_active = True
        self.n_objects = 0



###############################################################################
##### SECTION 3: More Ways to Create Grids ####################################
###############################################################################

class GridFromAxes(Grid):
    def __init__(self,*axes,n_properties=1,axunits=None,gridunits=None):
        """Initialize a Grid from a set of axes
        
        Create a grid by specifying the axes. Each axis given should
        be an array specifying the edges of the desired grid cells along
        that dimension.

        Parameters
        ----------
        *axes : array
            One or more arrays of evenly spaced numbers describing the 
            edgeds of each grid cell in one dimension.
        n_properties : int >= 1 (optional)
            The number of properties which will be stored in the grid. Note this
            number can be increased later.
        axunits : str or tuple of str (optional)
            A string or tuple of strings specifying the units of the spatial
            axes.
        gridunits : str or tuple of str (optional)
            A string or tuple of strings specifying the units of the grid values.
            A single value can be specified, or one value can be provided for 
            each property, in which case the length of the tuple should equal
            n_properties.
        """

        center_point = []
        side_length = []
        pixel_size = []
        for i,ax in enumerate(axes):
            if len(ax) > 1:
                steps = ax[1:]-ax[:-1]
                if not np.all(np.isclose(steps, steps[0])):
                    raise ValueError("grid spacing is not uniform")
            
            center_point.append((ax[-1]+ax[0])/2)
            side_length.append(ax[-1]-ax[0])
            pixel_size.append((ax[-1]-ax[0])/(len(ax)-1))
        
        center_point = np.array(center_point)
        side_length = np.array(side_length)
        pixel_size = np.array(pixel_size)

        super().__init__(n_properties,center_point,side_length,pixel_size,axunits,gridunits)

        n_pixels_decimal = self.side_length/self.pixel_size
        n_pixels_ceil = np.ceil(n_pixels_decimal)
        if np.any(n_pixels_decimal != n_pixels_ceil):
            self.side_length = self.pixel_size * n_pixels_ceil
            if (not side_length is None) and (not center_point is None):
                warnings.warn("Side lengths increased to accomodate integer number of pixels")
        self.n_pixels = n_pixels_ceil.astype('int')

class GridFromAxesAndFunction(GridFromAxes):
    def __init__(self,function,*axes,function_kwargs={},n_properties=1,axunits=None,gridunits=None):
        """Initialize a Grid from a set of axes and populate it using a function
        the axes
        
        Create a grid by specifying the axes. Each axis given should be an array
        specifying the edges of the desired grid cells along that dimension. A
        function which takes the axis centers (determined as the midpoints along
        the specified axis edges) as arguments and returns an array is then used
        to populate the grid.

        Parameters
        ----------
        function : function
            Function that will be used to populate the grid. This should take as
            arguments an arbitrary number of arrays (or a number of arrays equal
            to the number of axes specified) and return an array of shape (n1,
            n2, ... ni).
        *axes : array
            One or more arrays of evenly spaced numbers describing the edgeds of
            each grid cell in one dimension.
        function_kwargs : dict (optional)
            Dictionary containing additional keyword arguments to pass when
            calling function.
        n_properties : int >= 1 (optional)
            The number of properties which will be stored in the grid. Note this
            number can be increased later.
        axunits : str or tuple of str (optional)
            A string or tuple of strings specifying the units of the spatial
            axes.
        gridunits : str or tuple of str (optional)
            A string or tuple of strings specifying the units of the grid
            values. A single value can be specified, or one value can be
            provided for each property, in which case the length of the tuple
            should equal n_properties.
        """
        super().__init__(*axes,n_properties=n_properties,axunits=axunits,gridunits=gridunits)
        self.init_grid()
        shape = self.grid.shape
        self.grid = np.expand_dims(function(*self.axes_centers,**function_kwargs),axis=-1)
        if self.grid.shape != shape:
            raise ValueError("function does not produce a grid of the correct shape")

