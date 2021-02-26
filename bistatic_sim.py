#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:53:46 2020

@author: semmerson
"""
import os, pyart, pickle, lzma
import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
import matplotlib.animation as mani
import matplotlib.colors as colors
import datetime as dt
from netCDF4 import Dataset
from scipy.fftpack import fft2, fftshift
from scipy import interpolate, ndimage, signal, spatial
from skimage.restoration import unwrap_phase
from scipy.io import loadmat
from numba import njit
from pyart.map._load_nn_field_data import _load_nn_field_data
from pyart.map.grid_mapper import NNLocator
#from line_profiler import LineProfiler
#from numba.types import float64, int64, complex128
cdict = {'red':   [[0.0,  0.0, 0.0],
                   [0.25,  0.782, 0.782],
                   [0.333,  0.0, 0.0],
                   [0.499,  0.0, 0.0],
                   [0.5,  0.938, 0.938],
                   [0.583,  0.824, 0.824],
                   [0.666,  0.745, 0.745],
                   [0.749,  0.431, 0.431],
                   [0.75,  0.588, 0.588],
                   [0.875,  1.0, 1.0],
                   [1.0,  0.0, 0.0]],
         'green': [[0.0,  0.0, 0.0],
                   [0.25, 0.782, 0.782],
                   [0.333,  1.0, 1.0],
                   [0.499,  0.105, 0.105],
                   [0.5,  0.938, 0.938],
                   [0.583,  0.725, 0.725],
                   [0.666,  0.196, 0.196],
                   [0.749,  0.0, 0.0],
                   [0.75,  0.0, 0.0],
                   [0.875,  1.0, 1.0],
                   [1.0,  0.098, 0.098]],
         'blue':  [[0.0,  0.0, 0.0],
                   [0.25,  0.782, 0.782],
                   [0.333,  1.0, 1.0],
                   [0.499,  0.898, 0.898],
                   [0.5,  0.0, 0.0],
                   [0.583,  0.0, 0.0],
                   [0.666,  0.0, 0.0],
                   [0.749,  0.0, 0.0],
                   [0.75,  0.784, 0.784],
                   [0.875,  1.0, 1.0],
                   [1.0,  0.098, 0.098]]}
z_cmap = colors.LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
cdict = {'red':   [[0.0,  1.0, 1.0],
                   [0.15, 1.0, 1.0],
                   [0.2,  0.98, 0.98],
                   [0.249,  0.447, 0.447],
                   [0.25,  0.412, 0.412],
                   [0.274,  0.125, 0.125],
                   [0.275,  0.098, 0.098],
                   [0.324,  0.184, 0.184],
                   [0.325,  0.216, 0.216],
                   [0.374,  0.675, 0.675],
                   [0.375,  0.706, 0.706],
                   [0.399,  0.129, 0.129],
                   [0.4,  0.039, 0.039],
                   [0.474,  0.059, 0.059],
                   [0.475,  0.282, 0.282],
                   [0.499,  0.416, 0.416],
                   [0.5,  0.51, 0.51],
                   [0.524,  0.478, 0.478],
                   [0.525,  0.412, 0.412],
                   [0.599,  0.949, 0.949],
                   [0.6,  0.976, 0.976],
                   [0.637,  1.0, 1.0],
                   [0.638,  1.0, 1.0],
                   [0.649,  1.0, 1.0],
                   [0.65,  1.0, 1.0],
                   [0.699,  1.0, 1.0], 
                   [0.7,  0.996, 0.996], 
                   [0.8,  0.38, 0.38],
                   [0.85,  0.235, 0.235],
                   [1.0,  0.176, 0.176]],
         'green': [[0.0,  0.863, 0.863],
                   [0.15,  0.078, 0.078],
                   [0.2,  0.016, 0.016],
                   [0.249,  0.012, 0.012],
                   [0.25,  0.008, 0.008],
                   [0.274,  0.004, 0.004],
                   [0.275,  0.004, 0.004],
                   [0.324,  0.843, 0.843],
                   [0.325,  0.886, 0.886],
                   [0.374,  0.937, 0.937],
                   [0.375,  0.941, 0.941],
                   [0.399,  0.992, 0.992],
                   [0.4,  0.973, 0.973],
                   [0.474,  0.388, 0.388],
                   [0.475,  0.439, 0.439],
                   [0.499,  0.49, 0.49],
                   [0.5,  0.416, 0.416],
                   [0.524,  0.188, 0.188],
                   [0.525,  0.0, 0.0],
                   [0.599,  0.004, 0.004],
                   [0.6,  0.227, 0.227],
                   [0.637,  0.557, 0.557],
                   [0.638,  0.616, 0.616],
                   [0.649,  0.867, 0.867],
                   [0.65,  0.902, 0.902],
                   [0.699,  0.592, 0.592], 
                   [0.7,  0.537, 0.537], 
                   [0.8,  0.024, 0.024],
                   [0.85,  0.0, 0.0],
                   [1.0,  0.0, 0.0]],
         'blue':  [[0.0,  0.863, 0.863],
                   [0.15,  0.706, 0.706],
                   [0.2,  0.51, 0.51],
                   [0.249,  0.553, 0.553],
                   [0.25,  0.557, 0.557],
                   [0.274,  0.553, 0.553],
                   [0.275,  0.557, 0.557],
                   [0.324,  0.883, 0.883],
                   [0.325,  0.898, 0.898],
                   [0.374,  0.949, 0.949],
                   [0.375,  0.953, 0.953],
                   [0.399,  0.196, 0.196],
                   [0.4,  0.137, 0.137],
                   [0.474,  0.078, 0.078],
                   [0.475,  0.278, 0.278],
                   [0.499,  0.412, 0.412],
                   [0.5,  0.471, 0.471],
                   [0.524,  0.224, 0.224],
                   [0.525,  0.0, 0.0],
                   [0.599,  0.024, 0.024],
                   [0.6,  0.329, 0.329],
                   [0.637,  0.831, 0.831],
                   [0.638,  0.616, 0.616],
                   [0.649,  0.69, 0.69],
                   [0.65,  0.663, 0.663],
                   [0.699,  0.337, 0.337], 
                   [0.7,  0.314, 0.314], 
                   [0.8,  0.008, 0.008],
                   [0.85,  0.0, 0.0],
                   [1.0,  0.0, 0.0]]}

v_cmap = colors.LinearSegmentedColormap('testCmap', segmentdata=cdict, N=512)        

def map_to_grid(x,y,z,data, grid_shape, grid_limits, grid_origin=None,
                grid_origin_alt=None, grid_projection=None,
                fields=None, gatefilters=False,
                map_roi=True, weighting_function='Barnes2', toa=17000.0,
                copy_field_data=True, algorithm='kd_tree', leafsize=10.,
                roi_func='dist_beam', constant_roi=None,
                z_factor=0.05, xy_factor=0.02, min_radius=500.0,
                h_factor=1.0, nb=1.5, bsp=1.0, **kwargs):
    """
    Map one or more radars to a Cartesian grid.
    Generate a Cartesian grid of points for the requested fields from the
    collected points from one or more radars. The field value for a grid
    point is found by interpolating from the collected points within a given
    radius of influence and weighting these nearby points according to their
    distance from the grid points. Collected points are filtered
    according to a number of criteria so that undesired points are not
    included in the interpolation.
    Parameters
    ----------
    x : 2D or 3D array
        Array of x coordinates
    y : 2D or 3D array
        Array of y coordinates
    z : 2D or 3D array
        Array of z coordinates
    data : 2D or 3D array
        Moment data to be interpolated
    grid_shape : 3-tuple of floats
        Number of points in the grid (z, y, x).
    grid_limits : 3-tuple of 2-tuples
        Minimum and maximum grid location (inclusive) in meters for the
        z, y, x coordinates.
    grid_origin : (float, float) or None
        Latitude and longitude of grid origin. None sets the origin
        to the native coordinate origin.
    grid_origin_alt: float or None
        Altitude of grid origin, in meters. None sets the origin
        to the native coordinate origin.
    grid_projection : dict
        Projection parameters defining the map projection used to transform the
        locations of the radar gates in geographic coordinate to Cartesian
        coodinates. None will use the default dictionary which uses a native
        azimutal equidistance projection. See :py:func:`pyart.core.Grid` for
        additional details on this parameter. The geographic coordinates of
        the radar gates are calculated using the projection defined for each
        radar. No transformation is used if a grid_origin and grid_origin_alt
        are None and a single radar is specified.
    fields : list or None
        List of fields within the radar objects which will be mapped to
        the cartesian grid. None, the default, will map the fields which are
        present in all the radar objects.
    gatefilters : GateFilter, tuple of GateFilter objects, optional
        Specify what gates from each radar will be included in the
        interpolation onto the grid. Only gates specified in each gatefilters
        will be included in the mapping to the grid. A single GateFilter can
        be used if a single Radar is being mapped. A value of False for a
        specific element or the entire parameter will apply no filtering of
        gates for a specific radar or all radars (the default).
        Similarily a value of None will create a GateFilter from the
        radar moments using any additional arguments by passing them to
        :py:func:`moment_based_gate_filter`.
    roi_func : str or function
        Radius of influence function. A functions which takes an
        z, y, x grid location, in meters, and returns a radius (in meters)
        within which all collected points will be included in the weighting
        for that grid points. Examples can be found in the
        :py:func:`example_roi_func_constant`,
        :py:func:`example_roi_func_dist`, and
        :py:func:`example_roi_func_dist_beam`.
        Alternatively the following strings can use to specify a built in
        radius of influence function:
            * constant: constant radius of influence.
            * dist: radius grows with the distance from each radar.
            * dist_beam: radius grows with the distance from each radar
              and parameter are based of virtual beam sizes.
        The parameters which control these functions are listed in the
        `Other Parameters` section below.
    map_roi : bool
        True to include a radius of influence field in the returned
        dictionary under the 'ROI' key. This is the value of roi_func at all
        grid points.
    weighting_function : 'Barnes' or 'Barnes2' or 'Cressman' or 'Nearest'
        Functions used to weight nearby collected points when interpolating a
        grid point.
    toa : float
        Top of atmosphere in meters. Collected points above this height are
        not included in the interpolation.
    Other Parameters
    ----------------
    constant_roi : float
        Radius of influence parameter for the built in 'constant' function.
        This parameter is the constant radius in meter for all grid points.
        This parameter is used when `roi_func` is `constant` or constant_roi
        is not None. If constant_roi is not None, the constant roi_func is
        used automatically.
    z_factor, xy_factor, min_radius : float
        Radius of influence parameters for the built in 'dist' function.
        The parameter correspond to the radius size increase, in meters,
        per meter increase in the z-dimension from the nearest radar,
        the same foreach meteter in the xy-distance from the nearest radar,
        and the minimum radius of influence in meters. These parameters are
        only used when `roi_func` is 'dist'.
    h_factor, nb, bsp, min_radius : float
        Radius of influence parameters for the built in 'dist_beam' function.
        The parameter correspond to the height scaling, virtual beam width,
        virtual beam spacing, and minimum radius of influence. These
        parameters are only used when `roi_func` is 'dist_mean'.
    copy_field_data : bool
        True to copy the data within the radar fields for faster gridding,
        the dtype for all fields in the grid will be float64. False will not
        copy the data which preserves the dtype of the fields in the grid,
        may use less memory but results in significantly slower gridding
        times. When False gates which are masked in a particular field but
        are not masked in the `refl_field` field will still be included in
        the interpolation. This can be prevented by setting this parameter
        to True or by gridding each field individually setting the
        `refl_field` parameter and the `fields` parameter to the field in
        question. It is recommended to set this parameter to True.
    algorithm : 'kd_tree'.
        Algorithms to use for finding the nearest neighbors. 'kd_tree' is the
        only valid option.
    leafsize : int
        Leaf size passed to the neighbor lookup tree. This can affect the
        speed of the construction and query, as well as the memory required
        to store the tree. The optimal value depends on the nature of the
        problem. This value should only effect the speed of the gridding,
        not the results.
    Returns
    -------
    grids : dict
        Dictionary of mapped fields. The keys of the dictionary are given by
        parameter fields. Each elements is a `grid_size` float64 array
        containing the interpolated grid for that field.
    """

    # check the parameters
    if weighting_function.upper() not in [
            'CRESSMAN', 'BARNES2', 'BARNES', 'NEAREST']:
        raise ValueError('unknown weighting_function')
    if algorithm not in ['kd_tree']:
        raise ValueError('unknown algorithm: %s' % algorithm)
    badval = -9999.0

    # determine the number of gates (collected points)
    total_gates = data.size
    data = np.ma.masked_invalid(data)
    # create arrays to hold the gate locations and indicators if the gate
    # should be included in the interpolation.
    gate_locations = np.ma.empty((total_gates, 3), dtype=np.float64)
    include_gate = np.ones((total_gates), dtype=np.bool)

    offsets = []    # offsets from the grid origin, in meters, for each radar

    # create a field lookup tables
    field_data = np.ma.empty((total_gates), dtype=np.float64)

    # calculate cartesian locations of gates
    xg_loc = x
    yg_loc = y
    zg_loc = z

    # add gate locations to gate_locations array
    gate_locations[:,0] = zg_loc.flatten()
    gate_locations[:,1] = yg_loc.flatten()
    gate_locations[:,2] = xg_loc.flatten()
    del xg_loc, yg_loc

    # determine which gates should be included in the interpolation
    gflags = zg_loc < toa      # include only those below toa
    include_gate = gflags.flatten()

    del gflags, zg_loc

    # copy/store references to field data for lookup
    field_data = data.ravel()

    # build field data lookup tables
    filtered_field_data = field_data[include_gate]

    # populate the nearest neighbor locator with the filtered gate locations
    nnlocator = NNLocator(gate_locations[include_gate], algorithm=algorithm,
                          leafsize=leafsize)

    # unpack the grid parameters
    nz, ny, nx = grid_shape
    zr, yr, xr = grid_limits
    z_start, z_stop = zr
    y_start, y_stop = yr
    x_start, x_stop = xr

    if nz == 1:
        z_step = 0.
    else:
        z_step = (z_stop - z_start) / (nz - 1.)
    if ny == 1:
        y_step = 0.
    else:
        y_step = (y_stop - y_start) / (ny - 1.)
    if nx == 1:
        x_step = 0.
    else:
        x_step = (x_stop - x_start) / (nx - 1.)

    if not hasattr(roi_func, '__call__'):
        if constant_roi is not None:
            roi_func = 'constant'
        else:
            constant_roi = 500.0
        if roi_func == 'constant':
            roi_func = _gen_roi_func_constant(constant_roi)
        elif roi_func == 'dist':
            roi_func = _gen_roi_func_dist(
                z_factor, xy_factor, min_radius, (0,0,0))
        elif roi_func == 'dist_beam':
            roi_func = _gen_roi_func_dist_beam(
                h_factor, nb, bsp, min_radius, (0,0,0))
        else:
            raise ValueError('unknown roi_func: %s' % roi_func)

    # create array to hold interpolated grid data and roi if requested
    grid_data = np.ma.empty((nz, ny, nx), dtype=np.float64)
    grid_data.set_fill_value(badval)

    if map_roi:
        roi = np.empty((nz, ny, nx), dtype=np.float64)

    # interpolate field values for each point in the grid
    for iz, iy, ix in np.ndindex(nz, ny, nx):

        # calculate the grid point
        x = x_start + x_step * ix
        y = y_start + y_step * iy
        z = z_start + z_step * iz
        r = roi_func(z, y, x)
        if map_roi:
            roi[iz, iy, ix] = r

        # find neighbors and distances
        ind, dist = nnlocator.find_neighbors_and_dists((z, y, x), r)

        if len(ind) == 0:
            # when there are no neighbors, mark the grid point as bad
            grid_data[iz, iy, ix] = np.ma.masked
            grid_data.data[iz, iy, ix] = badval
            continue

        # find the field values for all neighbors
        nn_field_data = filtered_field_data[ind]


        # preforms weighting of neighbors.
        dist2 = dist * dist
        r2 = r * r

        if weighting_function.upper() == 'NEAREST':
            value = nn_field_data[np.argmin(dist2)]
        else:
            if weighting_function.upper() == 'CRESSMAN':
                weights = (r2 - dist2) / (r2 + dist2)
                weights = np.exp(-dist2 / (2.0 * r2)) + 1e-5
            elif weighting_function.upper() == 'BARNES2':
                weights = np.exp(-dist2 / (r2/4)) + 1e-5
            value = np.ma.average(nn_field_data, weights=weights, axis=0)

        grid_data[iz, iy, ix] = value

    # create and return the grid dictionary
    grids = {}
    grids['data'] = grid_data
    if map_roi:
        grids['ROI'] = roi
    return grids

def example_roi_func_constant(zg, yg, xg):
    """
    Example RoI function which returns a constant radius.
    Parameters
    ----------
    zg, yg, xg : float
        Distance from the grid center in meters for the x, y and z axes.
    Returns
    -------
    roi : float
        Radius of influence in meters
    """
    # RoI function parameters
    constant = 500.     # constant 500 meter RoI
    return constant


def _gen_roi_func_constant(constant_roi):
    """
    Return a RoI function which returns a constant radius.
    See :py:func:`map_to_grid` for a description of the parameters.
    """

    def roi(zg, yg, xg):
        """ constant radius of influence function. """
        return constant_roi

    return roi


def example_roi_func_dist(zg, yg, xg):
    """
    Example RoI function which returns a radius which grows with distance.
    Parameters
    ----------
    zg, yg, xg : float
        Distance from the grid center in meters for the x, y and z axes.
    Returns
    -------
    roi : float
    """
    # RoI function parameters
    z_factor = 0.05         # increase in radius per meter increase in z dim
    xy_factor = 0.02        # increase in radius per meter increase in xy dim
    min_radius = 500.       # minimum radius
    offsets = ((0, 0, 0), )  # z, y, x offset of grid in meters from radar(s)

    offsets = np.array(offsets)
    zg_off = offsets[:, 0]
    yg_off = offsets[:, 1]
    xg_off = offsets[:, 2]
    r = np.maximum(z_factor * (zg - zg_off) +
                   xy_factor * np.sqrt((xg - xg_off)**2 + (yg - yg_off)**2),
                   min_radius)
    return min(r)


def _gen_roi_func_dist(z_factor, xy_factor, min_radius, offsets):
    """
    Return a RoI function whose radius grows with distance.
    See :py:func:`map_to_grid` for a description of the parameters.
    """
    def roi(zg, yg, xg):
        """ dist radius of influence function. """
        r = np.maximum(
            z_factor * zg +
            xy_factor * np.sqrt(xg**2 + yg**2),
            min_radius)
        return min(r)

    return roi


def example_roi_func_dist_beam(zg, yg, xg):
    """
    Example RoI function which returns a radius which grows with distance
    and whose parameters are based on virtual beam size.
    Parameters
    ----------
    zg, yg, xg : float
        Distance from the grid center in meters for the x, y and z axes.
    Returns
    -------
    roi : float
    """
    # RoI function parameters
    h_factor = 1.0      # height scaling
    nb = 1.5            # virtual beam width
    bsp = 1.0           # virtual beam spacing
    min_radius = 500.   # minimum radius in meters

    r = np.maximum(
        h_factor * (zg / 20.0) +
        np.sqrt(yg**2 + xg**2) *
        np.tan(nb * bsp * np.pi / 180.0), min_radius)
    return min(r)


def _gen_roi_func_dist_beam(h_factor, nb, bsp, min_radius, offsets):
    """
    Return a RoI function whose radius which grows with distance
    and whose parameters are based on virtual beam size.
    See :py:func:`map_to_grid` for a description of the parameters.
    """
    def roi(zg, yg, xg):
        """ dist_beam radius of influence function. """
        r = np.maximum(
            h_factor * (zg / 20.0) +
            np.sqrt(yg**2 + xg**2) *
            np.tan(nb * bsp * np.pi / 180.0), min_radius)
        return r

    return roi


def makeRadarStruct():
    radar = {}
    radar['txPos'] = np.array([0,-15,0])*1e3 #Tx position
    # x = np.linspace(-30,0,4)
    # y = np.linspace(-30,0,4)
    # x,y = np.meshgrid(x,y)
    # z = np.zeros((4,4))
    #radar['rxPos'] = np.reshape(np.stack((x,y,z)),(3,16)).T*1e3+radar['txPos']
    radar['rxPos'] = np.array([[-13,-7.5,0],[-6.5,-11.25,0],[6.5,-11.25,0],[13,-7.5,0],[0,-15,0]])*1e3 #Rx position 
    radar['lambda'] = 0.1031 #Operating wavelength
    radar['prt'] = 1/1282  #Tx PRT (s)
    radar['tau'] = 1.57e-6 #Pulse width (s)
    radar['fs'] = 1/radar['tau'] #Sampling Rate samples/s
    radar['Pt'] = 750e3 #Peak transmit power (W)
    radar['Tant'] = 63.36 #Antenna noise temp/brightness
    radar['Frx'] = 3 #Rx noise figure
    radar['M'] = 16 #Pulses per dwell
    radar['rxMechTilt'] = 1 #Mechanical tilt of Rx antenna
    radar['txMechEl'] = np.array([0.5])#Mechanical Tx elevation tilt
    radar['txMechAz'] = 0 #Baseline mechanical Tx position
    radar['txAz'] = np.mod(np.linspace(315,404,90),360) #Transmit azimuths
    radar['txEl'] = np.zeros(radar['txAz'].shape) #Electrical tx steering elv.
    radar['txG'] = 45.5 #Transmit antenna gain (dBi)
    radar['rxG'] = 18 #Receive antenna gain (dBi)
    radar['receiverGain'] = 45 #Receiver gain (dB)
    return radar

def makeWxStruct():
    wx = {}
    wx['scatMode'] = 'rayleigh' #Rayleigh or Bragg scattering
    wx['xSize'] = 30e3 #x,y,z dimensions of simulation volume (m)
    wx['ySize'] = 30e3
    wx['zSize'] = 4e3
    wx['getZh'] = 'Zh=35*np.ones((1,nPts))' #Zh/Zv values for uniform test case
    wx['getZv'] = 'Zv=Zh'              #not currently used
    wx['wrfDate'] = '2013-06-01 01:20:10' #NWP data query date: yyyy-mm-dd HH:MM:SS
    wx['wrfOffset'] = np.array([-14,4,0.5])#np.array([-20,-20,0.2]) # offset of simulation volume
                                               # from origin within NWP data                                           
    wx['spw'] = 2 #Spectrum width
    wx['ptsPerMinVol'] = 4 #Number of points in smallest resolution volume

    return wx

def datenum64(d):
    return 366 + d.item().toordinal() + (d.item() - dt.datetime.fromordinal(d.item().toordinal())).total_seconds()/(24*60*60)

# def getWrf(dstr,xq,yq,zq):
#     #Unit tested
#     f = loadmat('wrfDirectory.mat') #load NWP directory
#     YY = f['XX'][:]
#     XX = f['YY'][:]
#     ZZ = f['ZZ'][:]
#     dates = np.stack(f['dates'][:])[:,0]
#     uFiles = np.stack(f['uFiles'][:][0])[:,0]
#     vFiles = np.stack(f['vFiles'][:][0])[:,0]
#     wFiles = np.stack(f['wFiles'][:][0])[:,0]
#     zFiles = np.stack(f['zFiles'][:][0])[:,0]
#     dnumq = datenum64(np.datetime64(dstr)) #convert to datenum
#     #Get boundaries of NWP and query points
#     xmin,xmax = (np.min(XX),np.max(XX))
#     ymin,ymax = (np.min(YY),np.max(YY))
#     zmin,zmax = (np.min(ZZ),np.max(ZZ))
#     xqmin,xqmax = (np.min(xq),np.max(xq))
#     yqmin,yqmax = (np.min(yq),np.max(yq))
#     zqmin,zqmax = (np.min(zq),np.max(zq))
    
#     #Check if all query points fall within NWP volume
#     xgood = (xmin<xqmin) & (xmax>xqmax)
#     ygood = (ymin<yqmin) & (ymax>yqmax)
#     zgood = (zmin<zqmin) & (zmax>zqmax)
#     if xgood & ygood & zgood:
#         if np.any(dates==dnumq): #Interpolate directly if querying sampled instant
#             ind = np.where(dates==dnumq)
            
#             #Load variables from corresponding files and spatially
#             #interpolate
#             uFile = uFiles[ind]
#             vFile = vFiles[ind]
#             wFile = wFiles[ind]
#             zFile = zFiles[ind]
#             f = loadmat('WRF/'+uFile)
#             data_int = f['data_int'][:]
#             uu = interpolate.interpn((XX[:,0,0],YY[0,:,0],ZZ[0,0,:]), data_int,np.vstack((xq.flatten(),yq.flatten(),zq.flatten())))
#             f = loadmat('WRF/'+vFile)
#             data_int = f['data_int'][:]
#             vv = interpolate.interpn((XX,YY,ZZ), data_int,np.vstack((xq.flatten(),yq.flatten(),zq.flatten())))
#             f = loadmat('WRF/'+wFile)
#             data_int = f['data_int'][:]
#             ww = interpolate.interpn((XX,YY,ZZ), data_int,np.vstack((xq.flatten(),yq.flatten(),zq.flatten())))
#             f = loadmat('WRF/'+zFile)
#             data_int = f['data_int'][:]
#             zz = 10*np.log10(interpolate.interpn((XX,YY,ZZ), 10**(data_int/10),np.vstack((xq.flatten(),yq.flatten(),zq.flatten()))))
#         elif (dnumq > np.min(dates)) & (dnumq < np.max(dates)): #Otherwise interpolate in time
#             #Get variables from corresponding files at neighboring times
#             #samples
#             d0 = np.max(dates[dates<dnumq])
#             d1 = np.min(dates[dates>dnumq])
#             ind0 = np.where(dates==d0)[0]
#             ind1 = np.where(dates==d1)[0]
#             uFile0 = uFiles[ind0][0]
#             vFile0 = vFiles[ind0][0]
#             wFile0 = wFiles[ind0][0]
#             zFile0 = zFiles[ind0][0]
#             uFile1 = uFiles[ind1][0]
#             vFile1 = vFiles[ind1][0]
#             wFile1 = wFiles[ind1][0]
#             zFile1 = zFiles[ind1][0]
#             #Load variables from corresponding files and spatially
#             #interpolate
#             f = loadmat('WRF/'+uFile0)
#             data_int = f['data_int'][:]
#             rgi = interpolate.RegularGridInterpolator((XX[:,0,0],YY[0,:,0],ZZ[0,0,:]), np.swapaxes(data_int,0,1))
#             uu0 = rgi(np.array([xq,yq,zq]).T)
#             f = loadmat('WRF/'+vFile0)
#             data_int = f['data_int'][:]
#             rgi = interpolate.RegularGridInterpolator((XX[:,0,0],YY[0,:,0],ZZ[0,0,:]), np.swapaxes(data_int,0,1))
#             vv0 = rgi(np.array([xq,yq,zq]).T)
#             f = loadmat('WRF/'+wFile0)
#             data_int = f['data_int'][:]
#             rgi = interpolate.RegularGridInterpolator((XX[:,0,0],YY[0,:,0],ZZ[0,0,:]), np.swapaxes(data_int,0,1))
#             ww0 = rgi(np.array([xq,yq,zq]).T)
#             f = loadmat('WRF/'+zFile0)
#             data_int = f['data_int'][:]
#             rgi = interpolate.RegularGridInterpolator((XX[:,0,0],YY[0,:,0],ZZ[0,0,:]), np.swapaxes(10**(data_int/10),0,1))
#             zz0 = rgi(np.array([xq,yq,zq]).T)
            
#             f = loadmat('WRF/'+uFile1)
#             data_int = f['data_int'][:]
#             rgi = interpolate.RegularGridInterpolator((XX[:,0,0],YY[0,:,0],ZZ[0,0,:]), np.swapaxes(data_int,0,1))
#             uu1 = rgi(np.array([xq.flatten(),yq.flatten(),zq.flatten()]).T)
#             f = loadmat('WRF/'+vFile1)
#             data_int = f['data_int'][:]
#             rgi = interpolate.RegularGridInterpolator((XX[:,0,0],YY[0,:,0],ZZ[0,0,:]), np.swapaxes(data_int,0,1))
#             vv1 = rgi(np.array([xq.flatten(),yq.flatten(),zq.flatten()]).T)
#             f = loadmat('WRF/'+wFile1)
#             data_int = f['data_int'][:]
#             rgi = interpolate.RegularGridInterpolator((XX[:,0,0],YY[0,:,0],ZZ[0,0,:]), np.swapaxes(data_int,0,1))
#             ww1 = rgi(np.array([xq.flatten(),yq.flatten(),zq.flatten()]).T)
#             f = loadmat('WRF/'+zFile1)
#             data_int = f['data_int'][:]
#             rgi = interpolate.RegularGridInterpolator((XX[:,0,0],YY[0,:,0],ZZ[0,0,:]), np.swapaxes(10**(data_int/10),0,1))
#             zz1 = rgi(np.array([xq.flatten(),yq.flatten(),zq.flatten()]).T)
            
#             #Temporal interpolation
#             uu = uu0 + (dnumq-d0)*(uu1-uu0)/(d1-d0)
#             vv = vv0 + (dnumq-d0)*(vv1-vv0)/(d1-d0)
#             ww = ww0 + (dnumq-d0)*(ww1-ww0)/(d1-d0)
#             zz = 10*np.log10(zz0 + (dnumq-d0)*(zz1-zz0)/(d1-d0))
#             return (zz,uu,vv,ww)
#         else:
#             print('Specified date out of bounds.')
#     else:
#         print('Specified volume out of bounds.')

def getWrf(xq,yq,zq):
    #Unit tested
    fh = Dataset('may20.nc', 'r')
    XX = fh.variables['x'][:]
    YY = fh.variables['y'][:]
    ZZ = fh.variables['z'][:]
    #XX,YY,ZZ = np.meshgrid(XX,YY,ZZ)
    # UU = np.swapaxes(fh.variables['u'][:],0,1)
    # VV = np.swapaxes(fh.variables['v'][:],0,1)
    # WW = np.swapaxes(fh.variables['w'][:],0,1)
    # ref = np.swapaxes(fh.variables['reflectivity'][:],0,1)
    UU = fh.variables['u'][:]
    VV = fh.variables['v'][:]
    WW = fh.variables['w'][:]
    ref = fh.variables['reflectivity'][:]
    # UU = np.ones_like(ref)*10
    # VV = np.ones_like(ref)*10
    # WW = np.zeros_like(ref)
    fh.close()
    #Get boundaries of NWP and query points
    xmin,xmax = (np.min(XX),np.max(XX))
    ymin,ymax = (np.min(YY),np.max(YY))
    zmin,zmax = (np.min(ZZ),np.max(ZZ))
    xqmin,xqmax = (np.min(xq),np.max(xq))
    yqmin,yqmax = (np.min(yq),np.max(yq))
    zqmin,zqmax = (np.min(zq),np.max(zq))
    
    #Check if all query points fall within NWP volume
    xgood = (xmin<xqmin) & (xmax>xqmax)
    ygood = (ymin<yqmin) & (ymax>yqmax)
    zgood = (zmin<zqmin) & (zmax>zqmax)
    if xgood & ygood & zgood:
        #Load variables from corresponding files and spatially
        #interpolate
        rgi = interpolate.RegularGridInterpolator((XX,YY,ZZ),UU)
        uu = rgi(np.array([xq,yq,zq]).T)
        rgi = interpolate.RegularGridInterpolator((XX,YY,ZZ),VV)
        vv = rgi(np.array([xq,yq,zq]).T)
        rgi = interpolate.RegularGridInterpolator((XX,YY,ZZ),WW)
        ww = rgi(np.array([xq,yq,zq]).T)
        rgi = interpolate.RegularGridInterpolator((XX,YY,ZZ),10**(ref/10))
        ref = 10*np.log10(rgi(np.array([xq,yq,zq]).T))
        return (ref,uu,vv,ww)
    else:
        print('Specified volume out of bounds.')

def getRelPos(sensorPos,peakPt,scatPos):
#Gets relative angular position of some target with respect to the sensor
#pointing angle

#Inputs:
#sensorPos - sensor (Tx/Rx) position
#peakPt - point along pointing direction of sensor
#scatPos - scatterer positions

#Outputs:
#rxtheta - angular distance from boresight
#rxphi - angular rotation about boresight
#rxr - range
    scatPos = scatPos - sensorPos[:,np.newaxis]
    peakPt = peakPt - sensorPos
    
    res = cart2sph(peakPt[0],peakPt[1],peakPt[2])
    az = res[0]
    el = res[1]
    zAngle = -az-np.pi/2
    xAngle = el-np.pi/2
    
    rotZ = rotMat('z',zAngle)
    rotX = rotMat('x',xAngle)
    rotXZ = np.matmul(rotX,rotZ)
    scatPos = np.matmul(rotXZ,scatPos)
    es = np.einsum('ij,ij -> j',scatPos,scatPos)
    rxr = ne.evaluate('sqrt(es)')
    z = scatPos[2,:]/rxr
    rxphi = ne.evaluate('arccos(z)')
    x = scatPos[0,:]
    y = scatPos[1,:]
    rxtheta = ne.evaluate('arctan2(y,x)')
    return (rxtheta, rxphi, rxr)


def cart2sph(x,y,z):
    try:
        n = len(x)
    except:
        n = 1
    ptsnew = np.zeros((n,3))
    xy = ne.evaluate('x**2 + y**2')
    sxy = ne.evaluate('sqrt(xy)')
    ptsnew[:,0] = ne.evaluate('arctan2(y, x)')
    #ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    ptsnew[:,1] = ne.evaluate('arctan2(z, sxy)') # for elevation angle defined from XY-plane up
    ptsnew[:,2] = ne.evaluate('sqrt(xy + z**2)')
    return np.squeeze(ptsnew)

def rotMat(axis,angle):
    R = np.zeros((3,3))
    if axis == 'x':
        R[0,0] = 1;
        R[1,1] = np.cos(angle)
        R[1,2] = -np.sin(angle)
        R[2,1] = np.sin(angle)
        R[2,2] = np.cos(angle)
    elif axis == 'y':
        R[0,0] = np.cos(angle)
        R[0,2] = np.sin(angle)
        R[1,1] = 1
        R[2,0] = -np.sin(angle)
        R[2,2] = np.cos(angle)
            
    elif axis == 'z':
        R[0,0] = np.cos(angle)
        R[0,1] = -np.sin(angle)
        R[1,0] = np.sin(angle)
        R[1,1] = np.cos(angle)
        R[2,2] = 1
    return R

def getUlaWts(theta,phi,rxG):
#Unit tested
#GETULAWTS evaluates antenna pattern for rx antenna (assumed it would be a
# ula) through interpolation from table in rxPat
    uu = np.sin(theta)*np.cos(phi)
    vv = np.sin(theta)*np.sin(phi)
    f = loadmat('rxPat.mat')
    UU = f['UU'][:]
    VV = f['VV'][:]
    rxPat = f['rxPat'][:]
    rxPat = rxPat*10**(rxG/20)/np.max(rxPat)  
    rxPatr = np.real(rxPat)
    rxPati = np.imag(rxPat)
    inter = interpolate.RegularGridInterpolator((UU[0,:],VV[:,0]),rxPatr.T)
    wtsr = inter(np.array([uu,vv]).T)
    inter = interpolate.RegularGridInterpolator((UU[0,:],VV[:,0]),rxPati.T)
    wtsi = inter(np.array([uu,vv]).T)
    wts = wtsr+1j*wtsi
    return wts

def getThetaS(pts,txPos,rxPos):
#Unit tested
#GETTHETAS get forward scattering angle
#
#Inputs
#pts - x,y,z coordinates of scatterers
#txPos - position of transmitter
#rxPos - position of receiver
#
#Outputs:
#thetaS - forward scatter angle

    TS = np.zeros(pts.shape)
    SR = np.zeros(pts.shape)
    
    for ii in range(3):
        TS[ii,:] = pts[ii,:] - txPos[ii]
        SR[ii,:] = rxPos[ii] - pts[ii,:]
    TSmag = np.sqrt(np.sum(TS**2,axis=0))
    SRmag = np.sqrt(np.sum(SR**2,axis=0))
    
    thetaS = np.arccos(np.sum(TS*SR,axis=0)/(TSmag*SRmag))
    return thetaS

def bistaticWeights(txPos,rxPos,pts):
    #Unit tested
    #BISTATICWEIGHTS - gets bistatic scattering weights / loss factors from raindrops assuming dipole
    #model and no roll on antennas (explained in simulator section of Byrd 
    #dissertation)
    #
    #Inputs
    #txPos - transmitter position
    #rxPos - receiver position
    #pts - scatterer locations
    #
    #Outputs
    #Hwts - H polarized scattering weights
    #Vwts - V polarized scattering weights
    rpts = pts.T - txPos
    rrxPos = rxPos.T - txPos

    nPts = pts.shape[1]
    res = cart2sph(rpts[:,0],rpts[:,1],rpts[:,2])
    az,el,r = res[:,0],res[:,1],res[:,2]
    rotZ = np.zeros((nPts,3,3))
    rotY = np.zeros((nPts,3,3))
    rotX = np.zeros((3,3))
    
    rotZ[:,0,0] = np.cos(-az)
    rotZ[:,0,1] = np.sin(-az)
    rotZ[:,1,0] = -np.sin(-az)
    rotZ[:,1,1] = np.cos(-az)
    rotZ[:,2,2] = 1
    
    rotY[:,0,0] = np.cos(el)
    rotY[:,0,2] = np.sin(el)
    rotY[:,1,1] = 1
    rotY[:,2,0] = -np.sin(el)
    rotY[:,2,2] = np.cos(el)
    
    rotX[0,0] = 1
    rotX[1,2] = 1
    rotX[2,1] = -1
    
    res = np.matmul(rotZ,rrxPos[:,np.newaxis])
    vRx = np.matmul(rotY,res)
    vRx[:,0,0] = vRx[:,0,0] - r
    hRx = np.matmul(rotX,vRx)
    
    Vwts = 1-vRx[:,2,:]**2/np.sum(vRx**2,axis=1)
    Hwts = 1-hRx[:,2,:]**2/np.sum(hRx**2,axis=1) 
    return (Hwts, Vwts)

def z2eta(Z,_lambda):
#Z2ETA converts dBZ to eta in m^2/m^3
    Ze = 10**(Z/10)*10**(-18)
    Kw = 0.93
    eta = np.pi**5/_lambda**4*Kw**2*Ze
    return eta

def getBistaticRanges(pos1,pos2,pts):
    #GETBISTATICRANGE gets bistatic ranges for pts w/ respect to pos1 and
    #pos2
    if np.all(pos1 == pos2):
        ranges = 2*np.sqrt(np.sum((pts-pos1[:,np.newaxis])**2,axis=0))
    else:
        ranges = np.sqrt(np.sum((pts-pos1[:,np.newaxis])**2,axis=0)) + np.sqrt(np.sum((pts-pos2[:,np.newaxis])**2,axis=0))
    return ranges

def inRange2(r0,tau,ranges):
    #INRANGE returns mask for points within bistatic range shell
    #centered at r0
    
    c = 3e8
    rNear = r0 - c*tau/2
    rFar  = r0 + c*tau/2
    
    (i1,i2) = findinsorted(ranges,rNear,rFar)
    if i2 == 0:
        return np.array([])
    else:
        shellPts = np.arange(i1,i2+1).astype(int)
        return shellPts

def findinsorted(x,LowerBound,UpperBound):
# fast O(log2(N)) computation of the range of indices of x that satify the
# upper and lower bound values using the fact that the x vector is sorted
# from low to high values. Computation is done via a binary search.
#
# Input:
#
# x-            A vector of sorted values from low to high.       
#
# LowerBound-   Lower boundary on the values of x in the search
#
# UpperBound-   Upper boundary on the values of x in the search
#
# Output:
#
# lower_index-  The smallest index such that
#               LowerBound<=x(index)<=UpperBound
#
# upper_index-  The largest index such that
#               LowerBound<=x(index)<=UpperBound

    if (LowerBound>x[-1]) | (UpperBound<x[0]) | (UpperBound<LowerBound):
        # no indices satisfy bounding conditions
        lower_index = 0;
        upper_index = 0;
        return (lower_index, upper_index)
    lower_index_a=0
    lower_index_b=len(x)-1 # x(lower_index_b) will always satisfy lowerbound
    upper_index_a=0     # x(upper_index_a) will always satisfy upperbound
    upper_index_b=len(x)-1
    
    # The following loop increases _a and decreases _b until they differ 
    # by at most 1. Because one of these index variables always satisfies the 
    # appropriate bound, this means the loop will terminate with either 
    # lower_index_a or lower_index_b having the minimum possible index that 
    # satifies the lower bound, and either upper_index_a or upper_index_b 
    # having the largest possible index that satisfies the upper bound. 

    while (lower_index_a+1<lower_index_b) | (upper_index_a+1<upper_index_b):
    
        lw=np.floor((lower_index_a+lower_index_b)/2) # split the upper index
    
        if x[int(lw)] >= LowerBound:
            lower_index_b=lw # decrease lower_index_b (whose x value remains \geq to lower bound)   
        else:
            lower_index_a=lw # increase lower_index_a (whose x value remains less than lower bound)
            if (lw>upper_index_a) & (lw<upper_index_b):
                upper_index_a=lw # increase upper_index_a (whose x value remains less than lower bound and thus upper bound)

        up=np.ceil((upper_index_a+upper_index_b)/2) # split the lower index
        if x[int(up)] <= UpperBound:
            upper_index_a=up # increase upper_index_a (whose x value remains \leq to upper bound) 
        else:
            upper_index_b=up  # decrease upper_index_b
            if (up<lower_index_b) & (up>lower_index_a):
                lower_index_b=up #decrease lower_index_b (whose x value remains greater than upper bound and thus lower bound)

    
    if x[int(lower_index_a)]>=LowerBound:
        lower_index = lower_index_b
    else:
        lower_index = lower_index_a

    if x[int(upper_index_b)]<=UpperBound:
        upper_index = upper_index_a
    else:
        upper_index = upper_index_b
    return (lower_index, upper_index)

# @njit
# def getArrayWts2(theta,phi,interpolator,codeVal,txG):
#     #Unit tested
#     #GETARRAYWTS@ - gets transmit antenna pattern values at scatterer locations
#     #
#     #Inputs
#     #theta - angular distance from boresight
#     #phi - - angular rotation about boresight
#     #interpolator - set of precalculated patterns
#     #codeVal - whitening code bit
#     #txG - transmitter gain
#     #
#     #Output
#     #wts - transmit antenna pattern weights
#     fftres = 4096
#     st = np.sin(theta)
#     sf = (fftres-1)/2
#     uu = np.empty_like(phi)
#     vv = np.empty_like(phi)
#     uu = np.round((st*np.cos(phi)+1)*sf,0,uu).astype(int64)
#     vv = np.round((st*np.sin(phi)+1)*sf,0,vv).astype(int64)

#     codeVal[codeVal==-1] = 0
#     codeVal = 2*codeVal[1] + codeVal[0]
#     txPat = interpolator[:,:,codeVal]
#     txPat = txPat*10**(txG/20)/np.max(txPat)
#     wts = np.zeros(len(uu),dtype=complex128)
#     for i in range(len(uu)):
#         u = uu[i]
#         v = vv[i]
#         wts[i] = txPat[u,v]
#     return wts




def getArrayWts2(theta,phi,interpolator,codeVal,txG):
    #Unit tested
    #GETARRAYWTS@ - gets transmit antenna pattern values at scatterer locations
    #
    #Inputs
    #theta - angular distance from boresight
    #phi - - angular rotation about boresight
    #interpolator - set of precalculated patterns
    #codeVal - whitening code bit
    #txG - transmitter gain
    #
    #Output
    #wts - transmit antenna pattern weights
    fftres = 4096
    st = ne.evaluate('sin(theta)')
    sf = (fftres-1)/2
    uu = np.round((st*ne.evaluate('cos(phi)')+1)*sf).astype(int)
    vv = np.round((st*ne.evaluate('sin(phi)')+1)*sf).astype(int)
    codeVal[codeVal==-1] = 0
    codeVal = int(2*codeVal[1] + codeVal[0])
    txPat = interpolator[:,:,codeVal]
    idx = np.ravel_multi_index((uu,vv),txPat.shape)
    #wts = txPat[uu,vv]
    wts = txPat.flatten()[idx]
    return wts

def getDishWts(scatTheta,patTheta,txPat):
    inter = interpolate.interp1d(patTheta.flatten()*np.pi/180,txPat.flatten(),kind='nearest',fill_value=0,bounds_error=False)
    wts = inter(scatTheta)
    return wts

def calcV(mski,theta,txWts,curVwts,Pt):
    if len(mski) > 0:
        mskTh = np.abs(theta[mski])<np.pi/2
        rvwt = txWts[mski[mskTh]]*curVwts[mskTh]
        return np.sqrt(Pt)*np.sum(rvwt)
    else:
        return 0
    
def localize(aa,ee,tt,rxPos,txPos):
    c = 3e8
    nRx = rxPos[:,np.newaxis].shape[1]
    if np.all(np.isclose(rxPos,txPos)):
        theta = (90 - ee)*np.pi/180
        phi = aa*np.pi/180
        nAngle = len(theta)
        npts = np.zeros((nAngle,len(tt),3))
        ranges = c*tt/2
        RR,PH = np.meshgrid(ranges,phi)
        r,TH = np.meshgrid(ranges,theta)
        npts[:,:,0]  = r*np.sin(TH)*np.cos(PH)+txPos[0]
        npts[:,:,1]  = r*np.sin(TH)*np.sin(PH)+txPos[1]
        npts[:,:,2]  = r*np.cos(TH)+txPos[2]
    else:
        th = (90 - ee)*np.pi/180
        ph = aa*np.pi/180
        nAngle = len(th)
        pointingVector = np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
        rxVector = rxPos - np.tile(txPos,nRx)
        rxMag = np.sqrt(np.sum(rxVector**2))
        pp = np.zeros((nRx,nAngle))
        xx = np.zeros((nRx,len(tt),nAngle))
        yy = np.zeros((nRx,len(tt),nAngle))
        zz = np.zeros((nRx,len(tt),nAngle))
        
        npts = np.zeros((nAngle,len(tt),3))
        ca = np.zeros_like(aa).astype(float)
        ce = np.zeros_like(ee).astype(float)
        rxAz = np.arctan2(rxVector[1],rxVector[0])
        rxEl = np.arcsin(rxVector[2]/rxMag)
        D = rxMag
        crx = rotMat('y',rxEl)@rotMat('z',-rxAz)@rxVector[:,np.newaxis]
        for jj in range(nAngle):
            cp = rotMat('y',rxEl)@rotMat('z',-rxAz)@pointingVector[:,jj][:,np.newaxis]
            ca[jj] = np.arctan2(cp[1],cp[0])*180/np.pi
            ce[jj] = np.arcsin(cp[2]/np.sqrt(np.sum(cp**2)))*180/np.pi
            pp[:,jj] = np.arccos(np.sum(crx*cp)/(np.sqrt(np.sum(crx**2))*np.sqrt(np.sum(cp**2))))*180/np.pi
        PP,TT = np.meshgrid(pp,tt)
        AA = np.tile(ca,np.array([len(tt),1]))
        EE = np.tile(ce,np.array([len(tt),1]))
        xx = ((c*TT)**2 - D**2)/(2*(c*TT - D*np.cos(PP*np.pi/180)))*np.cos(AA*np.pi/180)*np.cos(EE*np.pi/180)
        yy = ((c*TT)**2 - D**2)/(2*(c*TT - D*np.cos(PP*np.pi/180)))*np.sin(AA*np.pi/180)*np.cos(EE*np.pi/180)
        zz = ((c*TT)**2 - D**2)/(2*(c*TT - D*np.cos(PP*np.pi/180)))*np.sin(EE*np.pi/180)
        for jj in range(nAngle):
            for kk in range(len(tt)):
                npts[jj,kk,:] =  rotMat('z',rxAz)@rotMat('y',-rxEl)@np.array([xx[kk,jj], yy[kk,jj],zz[kk,jj]])+txPos
    return npts
    
def freq2vel(df,xx,yy,zz,rxPos,txPos,_lambda):
    if np.all(np.isclose(rxPos,txPos)):
        vv = df*_lambda/2
    else:
        txVecs = np.zeros(xx.shape+(3,))
        txVecs[:,:,0] = -xx
        txVecs[:,:,1] = -yy
        txVecs[:,:,2] = -zz
        rxVecs = txVecs + np.tile(np.transpose(rxPos),np.array(xx.shape+(1,)))
        
        txNorm = np.sqrt(np.sum(txVecs**2,axis=2))
        rxNorm = np.sqrt(np.sum(rxVecs**2,axis=2))
        
        beta = np.arccos(np.sum(rxVecs*txVecs,axis=2)/(rxNorm*txNorm))
        vv = _lambda*df/(2*np.cos(beta/2))
    return vv

def getBa(xx,yy,zz,rxPos):
    txVecs = np.zeros(xx.shape + (3,))
    txVecs[:,:,0] = -xx
    txVecs[:,:,1] = -yy
    txVecs[:,:,2] = -zz
    rxVecs = txVecs + np.tile(rxPos[np.newaxis,:],xx.shape+(1,))
    txDist = np.sqrt(np.sum(txVecs**2,axis=2))
    rxDist = np.sqrt(np.sum(rxVecs**2,axis=2))
    beta = np.arccos(np.sum(rxVecs*txVecs,axis=2)/(rxDist*txDist))*180/np.pi
    return beta

def cressman(x1,y1,v1,xq,yq,R,minPts):
    fs = xq.shape
    xq = xq.flatten()
    yq = yq.flatten()
    v1c = np.zeros_like(xq)
    R = R**2
    for jj in range(len(xq)): 
        v1c[jj] = cressmanVal(x1,y1,v1,xq[jj],yq[jj],R,minPts)
    v1c = np.reshape(v1c,fs)
    return v1c

def barnes2(x1,y1,v1,xq,yq,R,minPts):
    fs = xq.shape
    xq = xq.flatten()
    yq = yq.flatten()
    v1c = np.zeros_like(xq)
    R = R**2
    for jj in range(len(xq)): 
        v1c[jj] = barnesVal(x1,y1,v1,xq[jj],yq[jj],R,minPts)
    v1c = np.reshape(v1c,fs)
    return v1c
@njit
def cressmanVal(x1,y1,v1,xq,yq,R,minPts):
    r1 = (x1 - xq)**2 + (y1 - yq)**2
    msk = r1<R
    vt = v1[msk]
    r1 = r1[msk]

    filt = np.isnan(vt)
    if np.sum(filt)/(len(filt)+1e-6) > 0.8 or np.sum(~filt) < minPts:
        return np.nan
    else:
        vt = vt[~filt]
        r1 = r1[~filt]
        w1 = (R - r1)/(R + r1+1e-6)
        s = np.sum(w1)
        if s == 0:
            return np.nan
        else:
            return np.sum(w1*vt)/np.sum(w1)
        
@njit
def barnesVal(x1,y1,v1,xq,yq,R,minPts):
    r1 = (x1 - xq)**2 + (y1 - yq)**2
    msk = r1<R
    vt = v1[msk]
    r1 = r1[msk]

    filt = np.isnan(vt)
    if np.sum(filt)/(len(filt)+1e-6) > 0.8 or np.sum(~filt) < minPts:
        return np.nan
    else:
        vt = vt[~filt]
        r1 = r1[~filt]
        w1 = np.exp(-r1/(R*0.3))+1e-5
        s = np.sum(w1)
        if s == 0:
            return np.nan
        else:
            return np.sum(w1*vt)/np.sum(w1)

def dealias(vr):
    nr,nb = vr.shape
    vr_d = np.copy(vr)
    va = np.nanmax(vr)
    adj = np.array([0, 2*va, -2*va, 4*va, -4*va])
    # Find lowest non-NaN value in first range ring (least likely to be aliased)
    rad1 = np.where(abs(vr_d[:,0])==np.nanmin(abs(vr_d[:,0])))[0][0]
    # Fill in NaN values in first range ring
    for j in np.arange(rad1, nr):
        if np.isnan(vr_d[j,0]) & np.isfinite(vr_d[j-1,0]):
            if j == 0:
                vr_d[j,0] = vr_d[nr-1,0]
            else:
                vr_d[j,0] = vr_d[j-1,0]
    for j in np.arange(0,rad1):
        if j == 0:
            if np.isnan(vr_d[j,0]) & np.isfinite(vr_d[nr-1,0]):
                vr_d[j,0] = vr_d[nr-1,0]
        else:
            if np.isnan(vr_d[j,0]) & np.isfinite(vr_d[j-1,0]):
                vr_d[j,0] = vr_d[j-1,0]
    # Radially fill all NaN values
    for i in np.arange(1,nb):
        j = np.isnan(vr_d[:,i]) & np.isfinite(vr_d[:,i-1])
        vr_d[j,i] = vr_d[j,i-1]
    nrad = 3
    nbin = 2
    window = np.ones((nrad,nbin))
    for j in np.arange(2,nb):
        vals = vr_d[:,j]+adj[:,np.newaxis]
        m = signal.convolve2d(vr_d[:,j-nbin:j],window,boundary='wrap')[nrad//2:-(nrad//2),0]/(nrad*nbin)
        pos = abs(vals-m).argmin(axis=0)
        vr_d[:,j] += adj[pos]
        
    # 1st radial dealiasing pass
    
        
    # Find least aliased radial
    r = np.mean(abs(vr_d),axis=1).argmin()
    #Azimuthal deliasing pass
    for i in np.arange(r+1,nr):
        vals = vr_d[i,:]+adj[:,np.newaxis]
        pos = abs(vr_d[i-1,:]-vals).argmin(axis=0)
        vr_d[i,:] += adj[pos]
    for i in np.arange(0, r-1):
        vals = vr_d[i,:]+adj[:,np.newaxis]
        if i == 0:
            pos = abs(vr_d[nr-1,:]-vals).argmin(axis=0)
        else:
            pos = abs(vr_d[i-1,:]-vals).argmin(axis=0)
        vr_d[i,:] += adj[pos]
    # 2nd radial dealiasing pass with higher constraints
    for j in np.arange(2,nb):
        vals = vr_d[:,j]+adj[:,np.newaxis]
        pos = abs(vals-np.mean(vr_d[:,j-2:j],axis=1)).argmin(axis=0)
        vr_d[:,j] += adj[pos]
    return vr_d

def simulate(radarStruct,wxStruct):
    os.chdir('/Users/semmerson/Documents/MATLAB/bistaticSimulator')
    f = loadmat('gold64.mat')
    c1 = np.squeeze(f['c1'][:])
    c2 = np.squeeze(f['c2'][:])
    c1 = np.ones(c1.shape)
    c2 = np.ones(c1.shape)
    txPos = radarStruct['txPos'] #Transmitter pos
    rxPos = radarStruct['rxPos'] #Receiver pos
    _lambda = radarStruct['lambda']#Operating wavelength
    prt = radarStruct['prt'] #Pulse Repetition Time
    tau = radarStruct['tau'] #pulse width
    fs = radarStruct['fs'] #sampling rate
    Pt = radarStruct['Pt'] #transmit power
    Tant = radarStruct['Tant'] #Antenna noise/brightness temp
    Frx = radarStruct['Frx'] #Receiver noise figure
    M = radarStruct['M'] #Samples per dwell
    txAz = radarStruct['txAz'] #Transmit azimuths
    txEl = radarStruct['txEl'] #Transmit elevations
    rxMechTilt = radarStruct['rxMechTilt'] #Mechanical tilt of rx antenna
    txMechEl = radarStruct['txMechEl'] #Mechanical tilt of tx antenna
    txG = radarStruct['txG'] #Transmit antenna gain (dBI)
    rxG = radarStruct['rxG'] #Receive antenna gain
    
    scatMode = wxStruct['scatMode'] #Wx or Bragg scatter
    xSize = wxStruct['xSize'] #Sizes of each dimension of Wx volume
    ySize = wxStruct['ySize']
    zSize = wxStruct['zSize']
    wrfDate = wxStruct['wrfDate'] #Date to be queried from WRF data
    
    #Spatial offset of sim volume within WRF volume
    wrfOffset = wxStruct['wrfOffset']
    
    #minimum number of scatterers in smallest volume
    ptsPerMinVol = wxStruct['ptsPerMinVol']    
    
    f = loadmat('win88d.mat')
    baseWin = np.squeeze(f['baseWin'][:])
    f = loadmat('win88dWht5.mat')
    codingWeights1 = np.squeeze(f['totalWts1'][:])
    codingWeights2 = np.squeeze(f['totalWts2'][:])
    codingWeights1 = baseWin
    codingWeights2 = baseWin
    idealPat = True
    dishMode = False
    fftres = 4096;
    wts = np.zeros((len(codingWeights1),len(codingWeights1),4)).astype('complex128')
    wts[:,:,0] = codingWeights1*codingWeights1[:,np.newaxis]
    wts[:,:,1] = codingWeights1*codingWeights2[:,np.newaxis]
    wts[:,:,2] = codingWeights2*codingWeights1[:,np.newaxis]
    wts[:,:,3]= codingWeights2*codingWeights2[:,np.newaxis]
    patMatrix = np.zeros((fftres,fftres,4)).astype('complex128')
    
    for ii in range(4):
        txPat = fftshift(fft2(wts[:,:,ii],shape = (fftres,fftres)))
        if idealPat: #Zero all sidelobes in ideal case
            maxi = round((0+1)/2*(fftres-1))+1
            maxj = round((-0+1)/2*(fftres-1))+1
            uslice = txPat[maxi,maxj:]
            vslice = txPat[maxi:,maxj]
            negIndsu = np.where((np.diff(np.abs(uslice))>0) & (np.abs(uslice[1:])<(0.5*np.max(np.abs(uslice[1:])))))[0]
            firstNullu = negIndsu[0]
            negIndsv = np.where((np.diff(np.abs(vslice))>0) & (np.abs(vslice[1:])<(0.5*np.max(np.abs(vslice[1:])))))[0]
            firstNullv = negIndsv[0]
            iu = maxi + firstNullv - 1
            il = maxi - firstNullv
            ju = maxj + firstNullu - 1
            jl = maxj - firstNullu
            txPat[:il,:] = 0
            txPat[iu:,:] = 0
            txPat[:,:jl] = 0
            txPat[:,ju:] = 0
        txPat = txPat*10**(txG/20)/np.max(txPat)
        patMatrix[:,:,ii] = txPat
    c = 3e8 #speed of light
    nRx = rxPos.shape[0] ## of Rx
    Ts = 1/fs #sampling interval
    T0 = 290 #reference noise temp
    Trec = T0*(10**(Frx/10)-1) #Rx noise temp
    B = fs #receiver bandwidth (assumes IQ sampling)
    k = 1.381e-23 #Boltzmann
    N0 = k*(Tant+Trec)*B #Noise power at receiver
    cn = 1e-14
    
    # Populate scattering centers
    domainVolume = xSize*ySize*zSize
    minVolSize = (np.sin(0.5*np.pi/180)*20e3)**2*c*tau/2
    nPts = np.round(ptsPerMinVol * domainVolume/minVolSize)
    ptVolume = domainVolume/nPts
    np.random.seed(777);
    
    
    pts = np.zeros((3,int(nPts)))
    pts[0,:] = xSize*np.random.uniform(size=(int(nPts))) - xSize/2
    pts[1,:] = ySize*np.random.uniform(size=(int(nPts))) - ySize/2
    pts[2,:] = zSize*np.random.uniform(size=(int(nPts)))
    Vpower = []
    rr = []
    nAz = txAz.shape[0]
    nEl = txMechEl.shape[0]
    for ll in range(nEl):
        os.chdir('/Users/semmerson/Documents/cm1r19.10/grinder')
        #os.chdir('/Users/semmerson/Documents/python/data/may7/may_7_250m')
        #Zv,uu,vv,ww = getWrf(wrfDate,pts[0,:]/1000+wrfOffset[0],pts[1,:]/1000+wrfOffset[1],pts[2,:]/1000+wrfOffset[2])
        Zv,uu,vv,ww = getWrf(pts[0,:]/1000+wrfOffset[0],pts[1,:]/1000+wrfOffset[1],pts[2,:]/1000+wrfOffset[2])
        os.chdir('/Users/semmerson/Documents/MATLAB/bistaticSimulator')
        # inds = np.where(Zv != 0)[0]
        # nPts = len(inds)
        # pts = pts[:,inds]
        # Zv = Zv[inds]
        # uu = uu[inds]
        # vv = vv[inds]
        # ww = ww[inds]
        #Vectorize wind velocities
        windSpeed = np.zeros((3,int(nPts)))
        windSpeed[0,:] = uu
        windSpeed[1,:] = vv
        windSpeed[2,:] = ww
        
        del uu,vv,ww
        
        staticVwts = np.zeros((int(nRx),int(nPts))).astype('complex128')
        print('Calculating Bistatic RCS:')
        rxr = np.sqrt(np.sum(rxPos**2,axis=1))
        rxz = rxr*np.sin(rxMechTilt*np.pi/180)
        
        rxRef = np.vstack((np.zeros((2,int(nRx))), rxz))
        
        txPhi = (90-txAz)*np.pi/180;
        txMechTheta = (90-txMechEl)*np.pi/180;
        nAz = txAz.shape[0]
        nEl = txMechTheta.shape[0]
        
        nr = np.zeros((nRx),dtype=int)
        for i in range(nRx):
            brmin = np.sqrt(np.sum((rxPos[i,:]-txPos)**2))
            brmax = 1e5
            # if np.all(np.isclose(rxPos[i,:],txPos)):
            #     brmax = brmax/2
            rr.append(np.arange(brmin,brmax,c*Ts/2))
            nr[i] = len(rr[i])
            Vpower.append(np.zeros((nAz,nEl,nr[i],M)).astype('complex128'))
        
        #Calculate portion of complex scatterer weights that will not change
        #with Tx pointing angle.
        #
        #Makes simplifying assumption that Tx scan happens instantaneously
        if dishMode:
            f = loadmat('dishWts.mat')
            patTheta = f['patTheta'][:]
            txPat = f['txPat'][:]
        for qq in range(M): #For samples in dwell
            pts = pts + prt*windSpeed
            br = np.zeros(staticVwts.shape)
            #Reference pointing direction vector for transmitter
    
            #Initialize variables
            nPts = pts.shape[1]
            rxPhi = np.zeros((nRx,nPts))
            rxTheta = np.zeros((nRx,nPts))
            rxr = np.zeros((nRx,nPts))
            for ii in range(nRx): #For each receiver          
                #Find effective receive ant area in direction of scatterers
                print('Receiver ',ii+1,' of ',nRx)
        
                #Get scatterer positions relative to receiver
                (rxTheta[ii,:], rxPhi[ii,:], rxr[ii,:]) = getRelPos(rxPos[ii,:],rxRef[:,ii],np.squeeze(pts))
                #Get Rx antenna pattern weightings 
                Ae = getUlaWts(rxTheta[ii,:],rxPhi[ii,:],rxG)*_lambda**2/(4*np.pi)
        
                #Get forward scatter angles
                #thetaS = getThetaS(pts,txPos,rxPos[ii,:])
                #Get dual-pol scattering amplitudes based on infinitesimal
                #dipole-like reradiation
                vwt = np.squeeze(bistaticWeights(txPos,rxPos[ii,:],pts)[1])
        
                # Do the "receive" half of the radar range equation (doesn't
                # change with Tx pointing angle.  Convert from Z or Cn^2 to
                # reflectivity (eta) as necessary
                if scatMode == 'rayleigh':
                    staticVwts[ii,:] = np.sqrt(vwt*ptVolume*z2eta(Zv,0.1)*Ae)/(4*np.pi*rxr[ii,:])*np.exp(1j*2*np.pi*rxr[ii,:]/_lambda)
                elif scatMode == 'bragg':
                    print('bragg?')
                    #staticVwts[ii,:] = np.sqrt(vwt*ptVolume*cn2eta(_lambda,cn,thetaS)*Ae)/(4*np.pi*rxr[ii,:,:])
                # Get total distance traveled from tx to each point back to rx
                br[ii,:] = getBistaticRanges(txPos,rxPos[ii,:],pts)
            #Format bistatic range info 
            br = np.squeeze(br)
            if nRx == 1:
                br = br.T
                
            # #Filter out unused scatterers based on bistatic range
            rmsk = np.min(br,axis=0)>brmax
            br = br[:,~rmsk]
            staticVwts = staticVwts[:,~rmsk]
            pts = pts[:,~rmsk]
            windSpeed = windSpeed[:,~rmsk]
            Zv = Zv[~rmsk]
            rxr = rxr[:,~rmsk]
                
            sortInds = np.zeros(br.shape).astype(int)
            for ii in range(nRx):
                sortInds[ii,:] = np.argsort(br[ii,:])
                br[ii,:] = np.sort(br[ii,:])
                staticVwts[ii,:] = staticVwts[ii,sortInds[ii,:]]
            staticVwts[np.isnan(staticVwts)] = 0
            del Ae,rxPhi,rxTheta 
            print('Simulating Observations:')
    
            for jj in range(nRx): #for each receiver
                  #Get sorted pts by bistatic range for current receiver
                  ptsr = pts[:,sortInds[jj,:]]
    
                  #Initialize cell arrays for storing range bin data
                  ncp = np.zeros((nr[jj]))
                  mskIndices = []
                  curVwts = []
                  txRef = np.vstack((np.sin(txMechTheta[ll])*np.cos(txPhi)+txPos[0],
                                     np.sin(txMechTheta[ll])*np.sin(txPhi)+txPos[1],
                                     np.cos(txMechTheta[ll])*np.ones(txPhi.shape)+txPos[2]))
                  for kk in range(nr[jj]):
                      mski = inRange2(rr[jj][kk],tau,br[jj,:]) #Find points in bin
                      mskIndices.append(mski)
                      ncp[kk] = len(mski)
                      if len(mski) > 0:
                          curVwts.append(staticVwts[jj,mski]) #Get wts for those points
                      else:
                          curVwts.append(np.array([0]))
                  for ii in range(nAz): #For each azimuth angle
                      if np.mod(ii+1,5) == 0: #Progress display
                          print('Pt', qq+1, 'Rx', jj+1, ': Elevation',ll+1,'/',nEl,'Azimuth',ii+1,'/',nAz)
           
                      #Get position relative to receiver
                      (phi, theta,r) = getRelPos(txPos,txRef[:,ii],ptsr)
                      #Calculate terms for phase delay and attenuation
                      rf = -1j*2*np.pi*r/_lambda
                      rangeFactor = r*ne.evaluate('exp(rf)') 
                      #Get transmit antenna weights
                      if dishMode:
                          txWts = getDishWts(theta,patTheta,txPat)/rangeFactor
                      else:
                          txWts = getArrayWts2(theta,phi,patMatrix,np.array((c1[qq], c2[qq])),txG)/rangeFactor
           
                      #Sum across samples and scale to form IQ
                      for kk in range(nr[jj]):
                          Vpower[jj][ii,ll,kk,qq] = calcV(mskIndices[kk],theta,txWts,curVwts[kk],Pt)
        
    return Vpower, rr
# lp = LineProfiler()
# lp_wrap = lp(simulate)
# lp_wrap()
# lp.print_stats()
    
radarStruct = makeRadarStruct()
wxStruct = makeWxStruct()
Vpower, rr = simulate(radarStruct,wxStruct)

txPos = radarStruct['txPos']
rxPos = radarStruct['rxPos']
_lambda = radarStruct['lambda']
prt = radarStruct['prt']
tau = radarStruct['tau']
fs = radarStruct['fs'] 
Pt = radarStruct['Pt'] 
Tant = radarStruct['Tant'] 
Frx = radarStruct['Frx'] 
M = radarStruct['M'] 
txAz = radarStruct['txAz'] 
txEl = radarStruct['txEl'] 
rxMechTilt = radarStruct['rxMechTilt'] 
txMechEl = radarStruct['txMechEl'] 
txG = radarStruct['txG']
rxG = radarStruct['rxG'] 

scatMode = wxStruct['scatMode'] 
xSize = wxStruct['xSize'] 
ySize = wxStruct['ySize']
zSize = wxStruct['zSize']
wrfDate = wxStruct['wrfDate'] 
wrfOffset = wxStruct['wrfOffset']

c = 3e8 
nRx = rxPos.shape[0]
nEl = txMechEl.shape[0]
Ts = 1/fs 
T0 = 290
Trec = T0*(10**(Frx/10)-1) 
B = fs 
k = 1.381e-23 
N0 = k*(Tant+Trec)*B 
cn = 1e-14

f = loadmat('wrfDirectory.mat') #load NWP directory
XX = f['XX'][:]-wrfOffset[0]
YY = f['YY'][:]-wrfOffset[1]
ZZ = f['ZZ'][:]

saveMode = False

vpp = []
snr = []
df = []
vels = []
xx = []
yy = []
zz = []
br = []
beta = []
snrThresh = 10**(20/10)
brThresh = 1e3

txrxDist = np.sqrt(np.sum((txPos-rxPos)**2,axis=1))
rxPhi = np.arctan2(rxPos[:,1]-txPos[1],rxPos[:,0]-txPos[0])

rxPos2 = np.array([np.cos(rxPhi), np.sin(rxPhi), np.zeros(len(txrxDist))])*txrxDist
txPos2 = np.zeros(3)
for i in range(nRx):
    noiseMat = np.sqrt(N0)/np.sqrt(2)*(np.random.normal(0,1,Vpower[i].shape) + 1j*np.random.normal(0,1,Vpower[i].shape))
    Vpower[i] += noiseMat
    vpp.append(np.sum(np.abs(Vpower[i])**2,axis=3)/radarStruct['M'])
    snr.append(vpp[i]/N0)
    f = unwrap_phase(np.angle(np.sum(np.conj(Vpower[i][:,:,:,:-1])*Vpower[i][:,:,:,1:],axis=3)))
    vpp[i][snr[i] < snrThresh] = np.nan
    f[snr[i] < snrThresh] = np.nan
    df.append(f/(2*np.pi*radarStruct['prt']))
    # br.append(np.reshape(getBistaticRanges(txPos2,rxPos2[:,i],np.array((xx[i].flatten(),yy[i].flatten(),zz[i].flatten()))),xx[i].shape))
    # beta.append(getBa(xx[i],yy[i],zz[i],rxPos2[:,i]))
    vel = np.zeros(df[i].shape)
    xx0 = np.zeros(df[i].shape)
    yy0 = np.zeros(df[i].shape)
    zz0 = np.zeros(df[i].shape)
    for l in range(nEl):
        pts = localize(90-txAz,radarStruct['txMechEl'][l]*np.ones(txAz.shape),rr[i]/c,rxPos2[:,i],txPos2)
        #pts = localize(90-txAz,radarStruct['txMechEl'][l]*np.ones(txAz.shape),br[i][0,:]/c,rxPos2[:,i],txPos2)
        vel[:,l,:] = freq2vel(df[i][:,l,:],pts[:,:,0],pts[:,:,1],pts[:,:,2],rxPos2[:,i],txPos2,_lambda)
        xx0[:,l,:] = pts[:,:,0]
        yy0[:,l,:] = pts[:,:,1]
        zz0[:,l,:] = pts[:,:,2]
    vels.append(vel)
    xx.append(xx0)
    yy.append(yy0)
    zz.append(zz0)
#RR = np.sqrt(np.sum(pts**2,axis=2))
#br = np.reshape(getBistaticRanges(txPos2,rxPos2[:,rxN],np.array((xx0.flatten(),yy0.flatten(),zz0.flatten()))),xx0.shape)
#vels0 = freq2vel(df[rxN],xx0,yy0,zz0,rxPos2[:,rxN],txPos2,_lambda)
#vels0[snr[rxN,:,:] < snrThresh] = np.nan
# vpp[rxN,snr[rxN,:,:] < snrThresh] = np.nan
#vpp[rxN,br < brThresh] = np.nan
#vels0[br < brThresh] = np.nan
xc = 0
yc = 12
window = 5
x = (xc-window, xc+window)
y = (yc-window, yc+window)
l = 0
figsize = (14,8)
if nRx > 5:
    fig = plt.figure(figsize=figsize)
    m = int(np.ceil(np.sqrt(nRx)))-1
    n = int(np.ceil(nRx/m))-1
    axs = fig.subplots(m,n).flatten()
    for i in range(nRx-1):
        pc = axs[i].pcolormesh(xx[i][:,l,:]/1000,yy[i][:,l,:]/1000,vels[i][:,l,:],vmin=-40,vmax=40,cmap='pyart_balance',shading='auto')
        #ax.quiver(XX[::2,::2,1],YY[::2,::2,1],UU[::2,::2,1],VV[::2,::2,1],scale = 600)
        axs[i].set_xlim(x)
        axs[i].set_ylim(y)
        axs[i].scatter(x=0,y=0,marker='x',color='k',s=100)
        axs[i].scatter(x=rxPos2[0,i]/1000,y=rxPos2[1,i]/1000,marker='+',color='r',s=100)
        if i == nRx-1:
            axs[i].set_title('Tx Simulated Radial Velocity')
        else:
            axs[i].set_title('Rx '+ str(i+1) +' Simulated Bistatic Velocity')
        axs[i].set_xlabel('Zonal Distance from Tx Radar (km)')
        axs[i].set_ylabel('Meridional Distance from Tx Radar (km)')
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.05, 0.02, 0.92])
    cb = fig.colorbar(pc, cax=cbar_ax)
    cb.set_label('Bistatic Velocity (m/s)')
    # for ax in fig.get_axes():
    #     ax.label_outer()

    
    fig = plt.figure(figsize=figsize)
    axs = fig.subplots(m,n).flatten()
    for i in range(nRx-1):
        rFact = (rxPos2[0,i]-xx[i][:,l,:])**2+(rxPos2[1,i]-yy[i][:,l,:])**2++(rxPos2[2,i]-zz[i][:,l,:])**2
        #pc = axs[i].pcolormesh(xx[i][:,l,:]/1000,yy[i][:,l,:]/1000,10*np.log10(vpp[i][:,l,:]*rFact),vmin=-70,vmax=10,cmap='pyart_HomeyerRainbow')
        pc = axs[i].pcolormesh(xx[i][:,l,:]/1000,yy[i][:,l,:]/1000,10*np.log10(snr[i][:,l,:]),vmin=-10,vmax=60,cmap='pyart_HomeyerRainbow')
        axs[i].set_xlim(x)
        axs[i].set_ylim(y)
        axs[i].scatter(x=0,y=0,marker='x',color='k',s=100)
        axs[i].scatter(x=rxPos2[0,i]/1000,y=rxPos2[1,i]/1000,marker='+',color='r',s=100)
        if i == nRx-1:
            axs[i].set_title('Tx Simulated Range-Corrected Power')
        else:
            axs[i].set_title('Rx '+ str(i+1) +' Simulated Range-Corrected Power')
        axs[i].set_xlabel('Zonal Distance from Tx Radar (km)')
        axs[i].set_ylabel('Meridional Distance from Tx Radar (km)')
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.05, 0.02, 0.92])
    cb = fig.colorbar(pc, cax=cbar_ax)
    cb.set_label('Range-Corrected Power (dB)')
    # for ax in fig.get_axes():
    #     ax.label_outer()


else:
    fig = plt.figure(figsize=figsize)
    axs = fig.subplots(2,nRx)
    for i in range(nRx):
        pc = axs[0,i].pcolormesh(xx[i][:,l,:]/1000,yy[i][:,l,:]/1000,vels[i][:,l,:],vmin=-40,vmax=40,cmap='pyart_balance',shading='auto')
        #ax.quiver(XX[::2,::2,1],YY[::2,::2,1],UU[::2,::2,1],VV[::2,::2,1],scale = 600)
        axs[0,i].set_xlim(x)
        axs[0,i].set_ylim(y)
        axs[0,i].scatter(x=0,y=0,marker='x',s=100)
        axs[0,i].scatter(x=rxPos2[0,i]/1000,y=rxPos2[1,i]/1000,marker='+',s=100)
        if i == nRx-1:
            axs[0,i].set_title('Tx Simulated Radial Velocity')
            cb = plt.colorbar(pc,ax=axs[0,i])
            cb.set_label('Bistatic Velocity (m/s)')
        else:
            axs[0,i].set_title('Rx '+ str(i+1) +' Simulated Bistatic Velocity')
        axs[0,i].set_xlabel('Zonal Distance from Tx Radar (km)')
        axs[0,i].set_ylabel('Meridional Distance from Tx Radar (km)')
    
        rFact = (rxPos2[1,i]-yy[i][:,l,:])**2+(rxPos2[0,i]-xx[i][:,l,:])**2
        pc = axs[1,i].pcolormesh(xx[i][:,l,:]/1000,yy[i][:,l,:]/1000,10*np.log10(vpp[i][:,l,:]*rFact),vmin=-80,vmax=40,cmap=z_cmap)
        axs[1,i].scatter(x=0,y=0,marker='x',s=100)
        axs[1,i].scatter(x=rxPos2[0,i]/1000,y=rxPos2[1,i]/1000,marker='+',s=100)
        if i == nRx-1:
            axs[1,i].set_title('Tx Simulated Range-Corrected Power')
            cb = plt.colorbar(pc,ax=axs[1,i])
            cb.set_label('Range-Corrected Power (dB)')
        else:
            axs[1,i].set_title('Rx '+ str(i+1) +' Simulated Range-Corrected Power')
        axs[1,i].set_xlabel('Zonal Distance from Tx Radar (km)')
        axs[1,i].set_ylabel('Meridional Distance from Tx Radar (km)')
        axs[1,i].set_xlim(x)
        axs[1,i].set_ylim(y)
    for ax in fig.get_axes():
        ax.label_outer()
    plt.tight_layout()
# rxN = 1
# ax = fig.add_subplot(232)
# pc = ax.pcolormesh(xx[rxN]/1000,yy[rxN]/1000,vels[rxN],vmin=-40,vmax=40,cmap='pyart_balance',shading='auto')
# #ax.quiver(XX[::2,::2,1],YY[::2,::2,1],UU[::2,::2,1],VV[::2,::2,1],scale = 600)
# ax.set_xlim(x)
# ax.set_ylim(y)
# ax.scatter(x=0,y=0,marker='x',s=100)
# ax.scatter(x=rxPos2[0,rxN]/1000,y=rxPos2[1,rxN]/1000,marker='+',s=100)
# ax.set_title('Rx 2 Simulated Bistatic Velocity')
# ax.set_xlabel('Zonal Distance from Tx Radar (km)')
# ax.set_ylabel('Meridional Distance from Tx Radar (km)')
# cb = plt.colorbar(pc,ax=ax)
# cb.set_label('Bistatic Velocity (m/s)')
# ax = fig.add_subplot(235)
# rFact = (rxPos2[1,rxN]-yy[rxN])**2+(rxPos2[0,rxN]-xx[rxN])**2
# pc = ax.pcolormesh(xx[rxN]/1000,yy[rxN]/1000,10*np.log10(vpp[rxN]*rFact),vmin=-70,vmax=10,cmap='pyart_HomeyerRainbow')
# cb = plt.colorbar(pc,ax=ax)
# cb.set_label('Range-Corrected Power (dB)')
# ax.set_title('Rx 2 Simulated Range-Corrected Power')
# ax.set_xlabel('Zonal Distance from Tx Radar (km)')
# ax.set_ylabel('Meridional Distance from Tx Radar (km)')
# ax.set_xlim(x)
# ax.set_ylim(y)

# rxN = 2
# ax = fig.add_subplot(233)
# pc = ax.pcolormesh(xx[rxN]/1000,yy[rxN]/1000,vels[rxN],vmin=-40,vmax=40,cmap='pyart_balance',shading='auto')
# #ax.quiver(XX[::2,::2,1],YY[::2,::2,1],UU[::2,::2,1],VV[::2,::2,1],scale = 600)
# ax.set_xlim(x)
# ax.set_ylim(y)
# ax.scatter(x=0,y=0,marker='x',s=100)
# ax.scatter(x=rxPos2[0,rxN]/1000,y=rxPos2[1,rxN]/1000,marker='+',s=100)
# ax.set_title('Tx Simulated Radial Velocity')
# ax.set_xlabel('Zonal Distance from Tx Radar (km)')
# ax.set_ylabel('Meridional Distance from Tx Radar (km)')
# cb = plt.colorbar(pc,ax=ax)
# cb.set_label('Bistatic Velocity (m/s)')
# ax = fig.add_subplot(236)
# rFact = (rxPos2[1,rxN]-yy[rxN])**2+(rxPos2[0,rxN]-xx[rxN])**2
# pc = ax.pcolormesh(xx[rxN]/1000,yy[rxN]/1000,10*np.log10(vpp[rxN]*rFact),vmin=-70,vmax=10,cmap='pyart_HomeyerRainbow')
# cb = plt.colorbar(pc,ax=ax)
# cb.set_label('Range-Corrected Power (dB)')
# ax.set_title('Tx Simulated Range-Corrected Power')
# ax.set_xlabel('Zonal Distance from Tx Radar (km)')
# ax.set_ylabel('Meridional Distance from Tx Radar (km)')
# ax.set_xlim(x)
# ax.set_ylim(y)
# plt.tight_layout()
snrt = 3
#

rilThresh = 10*1e3
hscThresh = 10*1e3
betaThresh = [20,160]

# rPI = np.reshape(interpolate.griddata(np.vstack((xx0.flatten(),yy0.flatten())).T,vpp[0].flatten(),np.vstack((xx1.flatten(),yy1.flatten())).T,method='nearest'),xx1.shape)
# hPI = np.reshape(interpolate.griddata(np.vstack((xx1.flatten(),yy1.flatten())).T,vpp[1].flatten(),np.vstack((xx0.flatten(),yy0.flatten())).T,method='nearest'),xx0.shape)
# rBI = np.reshape(interpolate.griddata(np.vstack((xx0.flatten(),yy0.flatten())).T,brRil.flatten(),np.vstack((xx1.flatten(),yy1.flatten())).T,method='nearest'),xx1.shape)
# hBI = np.reshape(interpolate.griddata(np.vstack((xx1.flatten(),yy1.flatten())).T,brHsc.flatten(),np.vstack((xx0.flatten(),yy0.flatten())).T,method='nearest'),xx0.shape)
# rBeI = np.reshape(interpolate.griddata(np.vstack((xx0.flatten(),yy0.flatten())).T,betaRil.flatten(),np.vstack((xx1.flatten(),yy1.flatten())).T,method='nearest'),xx1.shape)
# hBeI = np.reshape(interpolate.griddata(np.vstack((xx1.flatten(),yy1.flatten())).T,betaHsc.flatten(),np.vstack((xx0.flatten(),yy0.flatten())).T,method='nearest'),xx0.shape)
# tvh = np.nanmin(np.stack((10*np.log10(rPI),vpp[1])),axis=0)
# tvr = np.nanmin(np.stack((10*np.log10(hPI),vpp[0])),axis=0)
# #hscMsk = np.any(np.stack((tvh < 7,np.isnan(tvh),brHsc < hscThresh,rBI < rilThresh, hsc['yy'] > hscPos[1] + 1)),axis=0)
# #rilMsk = np.any(np.stack((tvr < 7,np.isnan(tvr),brRil < rilThresh,hBI < hscThresh, ril['yy'] < rilPos[1] - 1)),axis=0)
# hscMsk = np.any(np.stack((tvh < -160,np.isnan(tvh),brHsc < hscThresh,rBI < rilThresh,rBeI < betaThresh[0], rBeI > betaThresh[1])),axis=0)
# rilMsk = np.any(np.stack((tvr < -160,np.isnan(tvr),brRil < rilThresh,hBI < hscThresh,hBeI < betaThresh[0], hBeI > betaThresh[1])),axis=0)

# nmsk = np.isnan(vels[0])
# vels[0][nmsk] = np.inf*(-1*np.ones((np.sum(nmsk))))**np.round(np.random.uniform(size=np.sum(nmsk)))
# #vels0 = ndimage.median_filter(vels0,footprint = np.ones((3,5)),mode='constant')
# vels[0][np.isinf(vels[0])] = np.nan

# nmsk = np.isnan(vpp[0])
# vpp[0][nmsk] = np.inf*(-1*np.ones((np.sum(nmsk))))**np.round(np.random.uniform(size=np.sum(nmsk)))
# #vpp[0,:,:] = ndimage.median_filter(vpp[0,:,:],footprint = np.ones((3,5)),mode='constant')
# vpp[0][np.isinf(vpp[0])] = np.nan

# nmsk = np.isnan(vels[1])
# vels[1][nmsk] = np.inf*(-1*np.ones((np.sum(nmsk))))**np.round(np.random.uniform(size=np.sum(nmsk)))
# #vels1 = ndimage.median_filter(vels1,footprint = np.ones((3,5)),mode='constant')
# vels[1][np.isinf(vels1)] = np.nan

# nmsk = np.isnan(vpp[1])
# vpp[1][nmsk] = np.inf*(-1*np.ones((np.sum(nmsk))))**np.round(np.random.uniform(size=np.sum(nmsk)))
# #vpp[1,:,:] = ndimage.median_filter(vpp[1,:,:],footprint = np.ones((3,5)),mode='constant')
# vpp[1][np.isinf(vpp[1])] = np.nan

# vels[0][rilMsk] = np.nan
# vpp[0][rilMsk] = np.nan
# vels[1][hscMsk] = np.nan
# vpp[1][hscMsk] = np.nan

rxc = 0e3
ryc = 15e3
rwindow = 15*1e3
roiLimsX = (rxc-rwindow ,rxc+rwindow)
roiLimsY = (ryc-rwindow ,ryc+rwindow)
roiLimsZ = (0,4e3)
res = 0.125e3
nx = int(np.diff(roiLimsX)[0]//res)
ny = int(np.diff(roiLimsY)[0]//res)
nz = int(np.diff(roiLimsZ)[0]//res)
xq = np.linspace(roiLimsX[0],roiLimsX[1]-res,nx)
yq = np.linspace(roiLimsY[0],roiLimsY[1]-res,ny)
zq = np.linspace(roiLimsZ[0]+res,roiLimsZ[1],nz)
zq,yq,xq = np.meshgrid(zq,yq,xq,indexing='ij')
refgrids = -999*np.ones((nRx,nz,ny,nx)) 
velgrids = -999*np.ones((nRx,nz,ny,nx)) 
for i in range(nRx):
    print(i)
    grid = map_to_grid(xx[i],yy[i],zz[i],vpp[i],(nz,ny,nx),(roiLimsZ,roiLimsY,roiLimsX),h_factor=1,nb=1,bsp=1,min_radius=400)
    refgrids[i,:,:,:] = grid['data']
    grid = map_to_grid(xx[i],yy[i],zz[i],vels[i],(nz,ny,nx),(roiLimsZ,roiLimsY,roiLimsX),h_factor=1,nb=1,bsp=1,min_radius=400)
    velgrids[i,:,:,:] = grid['data']
refgrids[refgrids < 0] = np.nan
refgrids[refgrids > 50] = np.nan
velgrids[velgrids < -999] = np.nan
uu = -999*np.ones((nz,ny,nx))
vv = -999*np.ones((nz,ny,nx))
ww = -999*np.ones((nz,ny,nx))
for j in range(nz):
    pwrmsk = np.any(np.vstack((np.isnan(refgrids[:,j,:,:]),np.isnan(velgrids[:,j,:,:]))),axis=0)
    npts = np.sum(~pwrmsk)
    VR = np.zeros((nRx,1,npts))
    UV = np.zeros((3,1,npts))
    MMinv = np.zeros((nRx,3,npts))
    
    ptPos = np.stack((xq[j,~pwrmsk],yq[j,~pwrmsk],zq[j,~pwrmsk]))
    #ptPos = np.stack((xq,yq,zq))
    gPts = ptPos[np.newaxis,:,:]-rxPos2.T[:,:,np.newaxis]
    #rilPts = ptPos - rilPos[:,np.newaxis,np.newaxis]
    #hscPts = ptPos - hscPos[:,np.newaxis,np.newaxis]
    
    az = np.pi/2 - np.arctan2(gPts[:,1,:],gPts[:,0,:])
    dist = np.sqrt(np.sum(gPts**2,axis=1))
    el = np.arcsin(gPts[:,2,:]/dist)
    beta = np.arccos(np.cos(el[:-1,:])*np.cos(el[-1,:])*np.cos(az[:-1,:]-az[-1,:]) + np.sin(el[:-1,:])*np.sin(el[-1,:]))
    
    #MMinv[0,:,:] = np.stack(((np.sin(rilAz)*np.cos(rilEl) + np.sin(ktlxAz)*np.cos(ktlxEl))/(2*np.cos(rilBeta/2)), (np.cos(rilAz)*np.cos(rilEl) + np.cos(ktlxAz)*np.cos(ktlxEl))/(2*np.cos(rilBeta/2)), (np.sin(rilEl)+np.sin(ktlxEl))/(2*np.cos(rilBeta/2))))
    for i in range(nRx-1):
        MMinv[i,:,:] = np.stack(((np.sin(az[i,:])*np.cos(el[i,:]) + np.sin(az[-1,:])*np.cos(el[-1,:]))/(2*np.cos(beta[i,:]/2)), 
                                  (np.cos(az[i,:])*np.cos(el[i,:]) + np.cos(az[-1,:])*np.cos(el[-1,:]))/(2*np.cos(beta[i,:]/2)), 
                                  (np.sin(el[i,:])+np.sin(el[-1,:]))/(2*np.cos(beta[i,:]/2))))
    MMinv[-1,:,:] = np.stack((np.sin(az[-1,:])*np.cos(el[-1,:]),np.cos(az[-1,:])*np.cos(el[-1,:]),np.sin(el[-1,:])))
    VR[:,0,:] = velgrids[:,j,~pwrmsk]
    
    for ii in range(npts):
        try:
            MM = np.linalg.lstsq(MMinv[:,:,ii],np.identity(nRx))[0]
            UV[:,0,ii] = MM@VR[:,0,ii]
        except:
            UV[:,0,ii] = np.zeros(3)
        
    uu[j,~pwrmsk] = np.squeeze(UV[0,0,:])
    vv[j,~pwrmsk] = np.squeeze(UV[1,0,:])
    ww[j,~pwrmsk] = np.squeeze(UV[2,0,:])
uu[np.abs(uu) > 100] = np.nan
vv[np.abs(vv) > 100] = np.nan
ww[np.abs(ww) > 100] = np.nan

azi = 122
xqs = np.tile(xq,(nEl,1,1))
yqs = np.tile(yq,(nEl,1,1))
fig = plt.figure(figsize=(8,9))
ax = fig.add_subplot(311)
pc = ax.pcolormesh(yq[:,:,azi]/1000,zq[:,:,azi]/1000,uu[:,:,azi],vmin=-40,vmax=40,cmap='pyart_balance')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('Retrieved U Wind (m/s)')
ax.set_aspect('equal')
ax.set_xlim(0,30)
ax.set_ylim(0,5)
ax.set_xlabel('Range (km)')
ax.set_ylabel('Altitude (km)')
ax.set_title('Tx Simulated '+str(radarStruct['txAz'][45])+ ' Deg RHI (U wind)')
ax = fig.add_subplot(312)
pc = ax.pcolormesh(yq[:,:,azi]/1000,zq[:,:,azi]/1000,vv[:,:,azi],vmin=-40,vmax=40,cmap='pyart_balance')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('Retrieved V Wind (m/s)')
ax.set_aspect('equal')
ax.set_xlim(0,30)
ax.set_ylim(0,5)
ax.set_xlabel('Range (km)')
ax.set_ylabel('Altitude (km)')
ax.set_title('Tx Simulated '+str(radarStruct['txAz'][45])+ ' Deg RHI (V wind)')
ax = fig.add_subplot(313)
pc = ax.pcolormesh(yq[:,:,azi]/1000,zq[:,:,azi]/1000,ww[:,:,azi],vmin=-40,vmax=40,cmap='pyart_balance')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('Retrieved W Wind (m/s)')
ax.set_aspect('equal')
ax.set_xlim(0,30)
ax.set_ylim(0,5)
ax.set_xlabel('Range (km)')
ax.set_ylabel('Altitude (km)')
ax.set_title('Tx Simulated '+str(radarStruct['txAz'][45])+ ' Deg RHI (W wind)')
plt.tight_layout()

#os.chdir('/Users/semmerson/Documents/python/data/may7/may_7_250m')
os.chdir('/Users/semmerson/Documents/cm1r19.10/grinder')
#fh = Dataset('line.nc', 'r')
fh = Dataset('may20.nc', 'r')
XX = fh.variables['x'][:]
YY = fh.variables['y'][:]
ZZ = fh.variables['z'][:]
#XX,YY,ZZ = np.meshgrid(XX,YY,ZZ)
UU = fh.variables['u'][:]
VV = fh.variables['v'][:]
WW = fh.variables['w'][:]
rref = fh.variables['reflectivity'][:]
fh.close()
alt = 1
# msk = zz[:,:,alt] < 2
# WW[msk] = np.nan
# UU[msk] = np.nan
# VV[msk] = np.nan



# ax=fig.add_subplot(223)
# pc = ax.pcolormesh(XX-wrfOffset[0]-txPos[0]/1000,YY-wrfOffset[1]-txPos[1]/1000,WW[:,:,alt].T,vmin=-10,vmax=10,cmap='pyart_balance',shading='auto')
# cb = plt.colorbar(pc,ax=ax)
# cb.set_label('Model Vertical Velocity (m/s)')
# #ax.quiver(yq[::2,::2]/1000,xq[::2,::2]/1000,vv[::2,::2],uu[::2,::2],scale=1000,width=0.0015)
# #ax.quiver(XX[::2]-wrfOffset[0]-txPos[0]/1000,YY[::2]-wrfOffset[1]-txPos[1]/1000,UU[::2,::2,0].T,VV[::2,::2,0].T,scale=1000,width=0.0015)
# ax.set_xlim(plims)
# ax.set_ylim(plims)
# ax.scatter(x=0,y=0,marker='x',color='k',s=100)
# ax.scatter(x=rxPos2[1,0]/1000,y=rxPos2[0,0]/1000,marker='+',color='r',s=100)
# ax.scatter(x=rxPos2[1,1]/1000,y=rxPos2[0,1]/1000,marker='+',color='b',s=100)
# ax=fig.add_subplot(224)
# pc = ax.pcolormesh(yq/1000,xq/1000,windVectors[2,:,:]/8,vmin=-10,vmax=10,cmap='pyart_balance',shading='auto')
# cb = plt.colorbar(pc,ax=ax)
# cb.set_label('Derived Vertical Velocity (m/s)')
# #ax.quiver(yq[::s,::s]/1000,xq[::s,::s]/1000,uu[::s,::s],vv[::s,::s],scale=1000,width=0.0015)
# #plt.quiver(XX[::2]-wrfOffset[0]-txPos[0]/1000,YY[::2]-wrfOffset[1]-txPos[1]/1000,UU[::2,::2,0].T,VV[::2,::2,0].T,color='blue',scale=1000,width=0.0015)
# ax.set_xlim(plims)
# ax.set_ylim(plims)
# ax.scatter(x=0,y=0,marker='x',color='k',s=100)
# ax.scatter(x=rxPos2[1,0]/1000,y=rxPos2[0,0]/1000,marker='+',color='r',s=100)
# ax.scatter(x=rxPos2[1,1]/1000,y=rxPos2[0,1]/1000,marker='+',color='b',s=100)
# plt.tight_layout()
# kv = np.copy(ktlxVr)
# kv[np.isnan(pvrTheory)]=np.nan
l = 10
xq = xq[l,:,:]
yq = yq[l,:,:]
zq = zq[l,:,:]
uq = uu[l,:,:]
vq = vv[l,:,:]
wq = ww[l,:,:]
alts = zq
alts[alts >= np.max(ZZ)*1e3] = np.max(ZZ)*1e3-1
#alts[alts <= np.max(ZZ)*1e3] = np.min(ZZ)*1e3+1
rgi = interpolate.RegularGridInterpolator((XX-wrfOffset[0]-txPos[0]/1000,YY-wrfOffset[1]-txPos[1]/1000,ZZ-txPos[2]/1000),UU)
uinterp = np.reshape(rgi(np.array([xq.flatten()/1000,yq.flatten()/1000,alts.flatten()/1000]).T),xq.shape)
rgi = interpolate.RegularGridInterpolator((XX-wrfOffset[0]-txPos[0]/1000,YY-wrfOffset[1]-txPos[1]/1000,ZZ-txPos[2]/1000),VV)
vinterp = np.reshape(rgi(np.array([xq.flatten()/1000,yq.flatten()/1000,alts.flatten()/1000]).T),xq.shape)
rgi = interpolate.RegularGridInterpolator((XX-wrfOffset[0]-txPos[0]/1000,YY-wrfOffset[1]-txPos[1]/1000,ZZ-txPos[2]/1000),WW)
winterp = np.reshape(rgi(np.array([xq.flatten()/1000,yq.flatten()/1000,alts.flatten()/1000]).T),xq.shape)
rgi = interpolate.RegularGridInterpolator((XX-wrfOffset[0]-txPos[0]/1000,YY-wrfOffset[1]-txPos[1]/1000,ZZ-txPos[2]/1000),rref)
zinterp = np.reshape(rgi(np.array([xq.flatten()/1000,yq.flatten()/1000,alts.flatten()/1000]).T),xq.shape)


xc = 0
yc = 14
window = 15
x = (xc-window, xc+window)
y = (yc-window, yc+window)
s = 4
fig = plt.figure(figsize=(14,6))
ax=fig.add_subplot(121)
pc = ax.pcolormesh(xq/1000,yq/1000,zinterp,vmin=0,vmax=60,cmap='pyart_HomeyerRainbow',shading='auto')
#pc = ax.pcolormesh(xq/1000,yq/1000,winterp,vmin=-40,vmax=40,cmap='pyart_balance',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('Model Reflectivity (dBZ)')
#ax.quiver(yq[::2,::2]/1000,xq[::2,::2]/1000,vv[::2,::2],uu[::2,::2],scale=1000,width=0.0015)
ax.quiver(xq[::s,::s]/1000,yq[::s,::s]/1000,uinterp[::s,::s],vinterp[::s,::s],scale=1000,width=0.0015)
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Model Input Reflectivity and Horizontal Winds')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
#s = 4
ax=fig.add_subplot(122)
pc = ax.pcolormesh(xq/1000,yq/1000,10*np.log10(np.mean(refgrids,axis=0)[l,:,:])+140,vmin=0,vmax=60,cmap='pyart_HomeyerRainbow',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('Range-Corrected Power (dB)')
ax.quiver(xq[::s,::s]/1000,yq[::s,::s]/1000,uu[l,::s,::s],vv[l,::s,::s],scale=1000,width=0.0015)
#plt.quiver(XX[::2]-wrfOffset[0]-txPos[0]/1000,YY[::2]-wrfOffset[1]-txPos[1]/1000,UU[::2,::2,0].T,VV[::2,::2,0].T,color='blue',scale=1000,width=0.0015)
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Retrieved Range-Corrected Power and Horizontal Winds')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
plt.tight_layout()

# xc = -15
# yc = -15
# window = 15
# x = (xc-window, xc+window)
# y = (yc-window, yc+window)
fig = plt.figure(figsize=(14,12))
ax=fig.add_subplot(331)
pc = ax.pcolormesh(xq/1000,yq/1000,uinterp,vmin=-50,vmax=50,cmap='pyart_balance',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('V Wind (m/s)')
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Model Input U Wind')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
ax=fig.add_subplot(332)
pc = ax.pcolormesh(xq/1000,yq/1000,uu[l,:,:],vmin=-50,vmax=50,cmap='pyart_balance',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('V Wind (m/s)')
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Retrieved U Wind')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
plt.tight_layout()
ax=fig.add_subplot(333)
pc = ax.pcolormesh(xq/1000,yq/1000,uu[l,:,:]-uinterp,vmin=-10,vmax=10,cmap='pyart_balance',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('U Wind Error (m/s)')
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Retrieved U Wind Error')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
ax=fig.add_subplot(334)
pc = ax.pcolormesh(xq/1000,yq/1000,vinterp,vmin=-50,vmax=50,cmap='pyart_balance',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('V Wind (m/s)')
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Model Input V Wind')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
ax=fig.add_subplot(335)
pc = ax.pcolormesh(xq/1000,yq/1000,vv[l,:,:],vmin=-50,vmax=50,cmap='pyart_balance',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('V Wind (m/s)')
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Retrieved V Wind')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
plt.tight_layout()
ax=fig.add_subplot(336)
pc = ax.pcolormesh(xq/1000,yq/1000,vv[l,:,:]-vinterp,vmin=-10,vmax=10,cmap='pyart_balance',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('V Wind Error (m/s)')
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Retrieved V Wind Error')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
ax=fig.add_subplot(337)
pc = ax.pcolormesh(xq/1000,yq/1000,winterp,vmin=-50,vmax=50,cmap='pyart_balance',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('W Wind (m/s)')
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Model Input W Wind')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
ax=fig.add_subplot(338)
pc = ax.pcolormesh(xq/1000,yq/1000,ww[l,:,:],vmin=-50,vmax=50,cmap='pyart_balance',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('V Wind (m/s)')
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Retrieved W Wind')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
plt.tight_layout()
ax=fig.add_subplot(339)
pc = ax.pcolormesh(xq/1000,yq/1000,ww[l,:,:]-winterp,vmin=-10,vmax=10,cmap='pyart_balance',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('V Wind Error (m/s)')
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Retrieved W Wind Error')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
plt.tight_layout()

if saveMode:
    radars = {}
    radars['vpp'] = vpp
    radars['snr'] = snr
    radars['df'] = df
    radars['vel'] = vels
    radars['xx'] = xx
    radars['yy'] = yy
    radars['zz'] = zz
    radars['br'] = br
    radars['beta'] = beta
    os.chdir('/Users/semmerson/Documents/python/data/PRS')
    if len(radarStruct['txMechEl']) > 1:
        fName = str(radarStruct['rxPos'].shape[0])+'rx_'+str(len(radarStruct['txAz']))+'az_'+str(radarStruct['txMechEl'][0])+'-'+str(radarStruct['txMechEl'][-1])+'el_'+str(radarStruct['M'])+'M'+'.xz'
    else:
        fName = str(radarStruct['rxPos'].shape[0])+'rx_'+str(len(radarStruct['txAz']))+'az_'+str(radarStruct['txMechEl'])+'el_'+str(radarStruct['M'])+'M'+'.xz'
    with lzma.open(fName,'wb') as f:
        pickle.dump(radarStruct,f)
        pickle.dump(wxStruct,f)
        pickle.dump(radars,f)
    del radars
    

def animate(t):
    s = 2
    plt.clf()
    ax = fig.add_subplot(111)
    pc = ax.pcolormesh(yq[:,:,t]/1000,zq[:,:,t]/1000,uu[:,:,t],vmin=-100,vmax=100,cmap=v_cmap)
    ax.quiver(yq[::s,::s,t]/1000,zq[::s,::s,t]/1000,vv[::s,::s,t],ww[::s,::s,t],scale=1600,width=0.001)
    cb = plt.colorbar(pc,ax=ax)
    cb.set_label('Retrieved U Wind (m/s)')
    ax.set_aspect('equal')
    ax.set_xlim(0,30)
    ax.set_ylim(0,5)
    ax.set_xlabel('N-S Distance from Tx (km)')
    ax.set_ylabel('Altitude (km)')
    #ax.set_title('Tx Simulated '+str(radarStruct['txAz'][t])+ ' Deg RHI')
    plt.tight_layout()

    
#os.chdir('/Users/semmerson/Documents/python/data')
fig = plt.figure(figsize=(12,4))
anim = mani.FuncAnimation(fig, animate, interval=1, frames=240)
anim.save('retrieval.gif',writer='imagemagick',fps=24)

fig = plt.figure(figsize=(12,4))
t = 120
s = 1
ax = fig.add_subplot(111)
pc = ax.pcolormesh(yqs[:,:,t]/1000,zqs[:,:,t]/1000,us[:,:,t],vmin=-50,vmax=50,cmap='pyart_balance')
ax.quiver(yqs[::s,::s,t]/1000,zqs[::s,::s,t]/1000,vs[::s,::s,t],ws[::s,::s,t],scale=1600,width=0.001)
cb = plt.colorbar(pc,ax=ax)
cb.set_label('Retrieved U Wind (m/s)')
ax.set_aspect('equal')
ax.set_xlim(0,10)
ax.set_ylim(0,5)
ax.set_xlabel('N-S Distance from Tx (km)')
ax.set_ylabel('Altitude (km)')
#ax.set_title('Tx Simulated '+str(radarStruct['txAz'][t])+ ' Deg RHI')
plt.tight_layout()
    
    