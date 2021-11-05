#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:07:06 2021

@author: semmerson
"""
import os,re, pyart, pickle, lzma
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from netCDF4 import Dataset
from scipy import interpolate
from skimage.restoration import unwrap_phase
from scipy.io import loadmat
from pyart.map.grid_mapper import NNLocator


def loadRadars(file):
    with lzma.open(file,'rb') as f:
        radarStruct = pickle.load(f)
        wxStruct = pickle.load(f)
        radars = pickle.load(f)
    return radarStruct,wxStruct,radars

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

def map_to_grid(x,y,z,data, grid_shape, grid_limits, grid_origin=None,
                grid_origin_alt=None, grid_projection=None,
                fields=None, gatefilters=False,
                map_roi=True, weighting_function='Barnes2', toa=17000.0,
                copy_field_data=True, algorithm='kd_tree', leafsize=10.,
                roi_func='dist_beam_bistatic', constant_roi=None,
                z_factor=0.05, xy_factor=0.02, min_radius=500.0,
                h_factor=1.0, nb=1.5, bsp=1.0, rxPos=np.array([0,0,0]), **kwargs):
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
    count = 0
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
            constant_roi = 1000.0
        if roi_func == 'constant':
            roi_func = _gen_roi_func_constant(constant_roi)
        elif roi_func == 'dist':
            roi_func = _gen_roi_func_dist(
                z_factor, xy_factor, min_radius, (0,0,0))
        elif roi_func == 'dist_beam':
            roi_func = _gen_roi_func_dist_beam(
                h_factor, nb, bsp, min_radius, (0,0,0))
        elif roi_func == 'dist_beam_bistatic':
            roi_func = _gen_roi_func_dist_beam_bistatic(
                h_factor, nb, bsp, min_radius, rxPos, (0,0,0))
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
            count += 1
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
                value = np.ma.average(nn_field_data, weights=weights, axis=0)
            elif weighting_function.upper() == 'BARNES2':
                kappa = 3.4e-1
                gamma = 0.1
                weights = np.exp(-dist2 / (r2*kappa)) + 1e-5
                value = np.ma.average(nn_field_data, weights=weights, axis=0)
                weights = np.exp(-dist2 / (r2*kappa*gamma)) + 1e-5
                value = value + np.ma.average(nn_field_data-value, weights=weights, axis=0)
                
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

def _gen_roi_func_dist_beam_bistatic(h_factor, nb, bsp, min_radius, rxPos, offsets):
    """
    Return a RoI function whose radius which grows with distance
    and whose parameters are based on virtual beam size.
    See :py:func:`map_to_grid` for a description of the parameters.
    """
    def roi(zg, yg, xg):
        """ dist_beam radius of influence function. """
        ba = getBa1(xg,yg,zg,rxPos)
        r = np.maximum(
            h_factor * (zg / 20.0) +
            np.sqrt(yg**2 + xg**2) *
            np.tan(nb * bsp * np.pi / 180.0)*1/(2*np.cos(np.pi/180*ba/2)), min_radius)
        return r

    return roi

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

def getBa1(xx,yy,zz,rxPos):
    txVec = np.array([-xx,-yy,zz])
    rxVec = txVec + rxPos
    txDist = np.sqrt(np.sum(txVec**2))
    rxDist = np.sqrt(np.sum(rxVec**2))
    beta = np.arccos(np.sum(rxVec*txVec)/(rxDist*txDist))*180/np.pi
    return beta

def gridMoments(xx,yy,zz,vpp,vels,roiLimsX,roiLimsY,roiLimsZ,nx,ny,nz):
    nRx = len(vpp)
    refgrids = np.ma.ones((nRx,nz,ny,nx)) 
    velgrids = np.ma.ones((nRx,nz,ny,nx)) 
    for i in range(nRx):
        # hpp[i][snr[i] < snrThresh] = np.nan
        # vpp[i][snr[i] < snrThresh] = np.nan
        # for j in range(vpp[0].shape[1]):
        #     ba = getBa(xx[i][:,j,:],yy[i][:,j,:],zz[i][:,j,:],rxPos2[:,i])
        #     msk = np.logical_or(ba < betaThresh[0], ba > betaThresh[1])
        #     vpp[i][:,j,:][msk] = np.nan
        #     vels[i][:,j,:][msk] = np.nan
        print('Gridding Power')
        if i == nRx - 1:
            grid = map_to_grid(xx[i],yy[i],zz[i],vpp[i],(nz,ny,nx),(roiLimsZ,roiLimsY,roiLimsX),weighting_function='Cressman',roi_func='dist_beam',h_factor=0.5,nb=1,bsp=1,min_radius=res*2)
        else:
            grid = map_to_grid(xx[i],yy[i],zz[i],vpp[i],(nz,ny,nx),(roiLimsZ,roiLimsY,roiLimsX),weighting_function='Cressman',roi_func='dist_beam_bistatic',h_factor=0.5,nb=1,bsp=1,rxPos=rxPos2[:,i],min_radius=res*2)
        refgrids[i,:,:,:] = grid['data']
        print('Gridding Velocity')
        if i == nRx - 1:
            grid = map_to_grid(xx[i],yy[i],zz[i],vels[i],(nz,ny,nx),(roiLimsZ,roiLimsY,roiLimsX),weighting_function='Cressman',roi_func='dist_beam',h_factor=0.5,nb=1,bsp=1,min_radius=res*2)
        else:
            grid = map_to_grid(xx[i],yy[i],zz[i],vels[i],(nz,ny,nx),(roiLimsZ,roiLimsY,roiLimsX),weighting_function='Cressman',roi_func='dist_beam_bistatic',h_factor=0.5,nb=1,bsp=1,rxPos=rxPos2[:,i],min_radius=res*2)
        velgrids[i,:,:,:] = grid['data']
    return refgrids,velgrids

def convertGrids(refgrids,velgrids,xq,yq,zq,rxPos,rxc,ryc):
    grids = []
    for i in range(refgrids.shape[0]):
        time = pyart.config.get_metadata('grid_time')
        time['data'] = 0
        x = pyart.config.get_metadata('x')
        x['data'] = xq
        y = pyart.config.get_metadata('y')
        y['data'] = yq
        z = pyart.config.get_metadata('z')
        z['data'] = zq
        origin_latitude = pyart.config.get_metadata('origin_latitude')
        origin_latitude['data'] = np.array([35.1812+ryc/1000*0.008979])
        origin_longitude = pyart.config.get_metadata('origin_longitude')
        origin_longitude['data'] = np.array([-97.4368+rxc/1000*0.0109045])
        origin_altitude = pyart.config.get_metadata('origin_altitude')
        radar_latitude = pyart.config.get_metadata('radar_latitude')
        radar_latitude['data'] = np.array([35.1812+rxPos[1,i]/1000*0.008979])
        radar_longitude = pyart.config.get_metadata('radar_longitude')
        radar_longitude['data'] = np.array([-97.4368+rxPos[0,i]/1000*0.0109045])
        radar_altitude = pyart.config.get_metadata('radar_altitude')
        radar_altitude['data'] = np.array([rxPos[2,i]])
        radar_time = pyart.config.get_metadata('radar_time')
        radar_time['data'] = np.array([0])
        radar_name = pyart.config.get_metadata('radar_name')
        radar_name['data'] = np.array(['nope'+str(i)])
        metadata = pyart.config.get_metadata('metadata')
        fields = {}
        fields['reflectivity'] = pyart.config.get_metadata('reflectivity')
        fields['reflectivity']['data'] = np.ma.masked_invalid(refgrids[i,:,:,:])
        fields['reflectivity']['_FillValue'] = -32768
        fields['corrected_velocity'] = pyart.config.get_metadata('velocity')
        fields['corrected_velocity']['data'] = np.ma.masked_invalid(velgrids[i,:,:,:])
        g = pyart.core.Grid(time, fields, metadata,
        origin_latitude, origin_longitude, origin_altitude, x, y, z,
        radar_latitude=radar_latitude, radar_longitude=radar_longitude,
        radar_altitude=radar_altitude, radar_name=radar_name,
        radar_time=radar_time, projection=None)
        grids.append(g)
    return grids

def multiDop(xq,yq,zq,rxPos2,refgrids,velgrids,numRad,a,b,c):
    uu = -999*np.ones((nz,ny,nx))
    vv = -999*np.ones((nz,ny,nx))
    ww = -999*np.ones((nz,ny,nx))
    nRx = refgrids.shape[0]
    for j in range(nz):
        ptPos = np.stack((xq[j,:,:],yq[j,:,:],zq[j,:,:]))
        gPts = ptPos[np.newaxis,:,:,:]-rxPos2.T[:,:,np.newaxis,np.newaxis]
        az = np.pi/2 - np.arctan2(gPts[:,1,:],gPts[:,0,:])
        dist = np.sqrt(np.sum(gPts**2,axis=1))
        el = np.arcsin(gPts[:,2,:]/dist)
        beta = np.arccos(np.cos(el[:-1,:,:])*np.cos(el[-1,:,:])*np.cos(az[:-1,:,:]-az[-1,:,:]) + np.sin(el[:-1,:])*np.sin(el[-1,:,:]))
        ptPos = np.stack((xq[j,:,:].flatten(),yq[j,:,:].flatten(),zq[j,:,:].flatten()))
        br = np.reshape(np.sqrt(np.sum((ptPos)**2,axis=0)) + np.sqrt(np.sum((ptPos[:,np.newaxis,:]-rxPos2[:,:-1,np.newaxis])**2,axis=0)),beta.shape)
        sigma = (1/np.tan(beta/2)+np.tan(beta/2))**2+1/np.tan(beta/2)
        Pr = 0.00025*(100000-br)
        R = 0.5*np.cos(beta/2)**2
        # qualI =(np.exp((np.pi/2-beta)**2/(np.pi**2))-1)*br
    
        qualI = a*Pr+b*R+c*sigma
        qualmsk = qualI < 0
        qualmsk = np.all(qualmsk,axis=0)
        pwrmsk = np.sum(np.isnan(refgrids[:,j,:,:]),axis=0)> 1#,np.nanvar(np.log10(refgrids[:,j,:,:]),axis=0) > 2)
        pwrmsk = pwrmsk | qualmsk
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
        inds = np.vstack((np.argsort(qualI,axis=0)[:,~pwrmsk],(nRx-1)*np.ones(npts))).astype(int)
        betaMsk = np.sum((beta > np.pi/6) & (beta < 5*np.pi/6),axis=0)+1
        for ii in range(npts):
            try:
                num = np.min((numRad,betaMsk[ii]))
                if num < 3:
                    UV[:,0,ii] = np.ones(3)*np.nan
                else:
                    MM = np.linalg.lstsq(MMinv[inds[-num:,ii],:,ii],np.identity(num))[0]
                    UV[:,0,ii] = MM@VR[inds[-num:,ii],0,ii]
            except:
                UV[:,0,ii] = np.ones(3)*np.nan
            
        uu[j,~pwrmsk] = np.squeeze(UV[0,0,:])
        vv[j,~pwrmsk] = np.squeeze(UV[1,0,:])
        ww[j,~pwrmsk] = np.squeeze(UV[2,0,:])
        #refgrids[:,j,pwrmsk] = np.nan
    inds = np.abs(np.sqrt(uu**2+vv**2)) > 60
    uu[inds] = np.nan
    vv[inds] = np.nan
    ww[np.abs(ww) > 50] = np.nan
    # rm = np.mean(10*np.log10(refgrids),axis=0)+145
    # msk = rm < 22
    # uu[msk] = np.nan
    # vv[msk] = np.nan
    # ww[msk] = np.nan
    return uu,vv,ww

def interpWrf(xq,yq,zq,el):
    os.chdir('/Users/semmerson/Documents/cm1r19.10/grinder')
    #fh = Dataset('line.nc', 'r')
    fh = Dataset('may20.nc', 'r')
    XX = fh.variables['x'][:]
    YY = fh.variables['y'][:]
    ZZ = fh.variables['z'][:]
    XX,YY,ZZ = np.meshgrid(XX,YY,ZZ)
    UU = np.swapaxes(fh.variables['u'][:],0,1)
    VV = np.swapaxes(fh.variables['v'][:],0,1)
    WW = np.swapaxes(fh.variables['w'][:],0,1)
    rref = np.swapaxes(fh.variables['reflectivity'][:],0,1)
    fh.close()
    xq = xq[el,:,:]
    yq = yq[el,:,:]
    zq = zq[el,:,:]
    alts = zq
    alts[alts >= np.max(ZZ)*1e3] = np.max(ZZ)*1e3-1
    #alts[alts <= np.max(ZZ)*1e3] = np.min(ZZ)*1e3+1
    # rgi = interpolate.RegularGridInterpolator((XX-wrfOffset[0]-txPos[0]/1000,YY-wrfOffset[1]-txPos[1]/1000,ZZ-wrfOffset[2]-txPos[2]/1000),UU)
    # uinterp = np.reshape(rgi(np.array([xq.flatten(),yq.flatten(),alts.flatten()]).T),xq.shape)
    # rgi = interpolate.RegularGridInterpolator((XX-wrfOffset[0]-txPos[0]/1000,YY-wrfOffset[1]-txPos[1]/1000,ZZ-wrfOffset[2]-txPos[2]/1000),VV)
    # vinterp = np.reshape(rgi(np.array([xq.flatten(),yq.flatten(),alts.flatten()]).T),xq.shape)
    # rgi = interpolate.RegularGridInterpolator((XX-wrfOffset[0]-txPos[0]/1000,YY-wrfOffset[1]-txPos[1]/1000,ZZ-wrfOffset[2]-txPos[2]/1000),WW)
    # winterp = np.reshape(rgi(np.array([xq.flatten(),yq.flatten(),alts.flatten()]).T),xq.shape)
    # rgi = interpolate.RegularGridInterpolator((XX-wrfOffset[0]-txPos[0]/1000,YY-wrfOffset[1]-txPos[1]/1000,ZZ-wrfOffset[2]-txPos[2]/1000),rref)
    # zinterp = np.reshape(rgi(np.array([xq.flatten(),yq.flatten(),alts.flatten()]).T),xq.shape)
    rgi = interpolate.RBFInterpolator(np.array((XX.ravel()-wrfOffset[0]-txPos[0]/1000,YY.ravel()-wrfOffset[1]-txPos[1]/1000,ZZ.ravel()-wrfOffset[2]-txPos[2]/1000)).T,UU.ravel(),neighbors=4,kernel='linear')
    uinterp = np.reshape(rgi(np.array([xq.flatten(),yq.flatten(),alts.flatten()]).T),xq.shape)
    rgi = interpolate.RBFInterpolator(np.array((XX.ravel()-wrfOffset[0]-txPos[0]/1000,YY.ravel()-wrfOffset[1]-txPos[1]/1000,ZZ.ravel()-wrfOffset[2]-txPos[2]/1000)).T,VV.ravel(),neighbors=4,kernel='linear')
    vinterp = np.reshape(rgi(np.array([xq.flatten(),yq.flatten(),alts.flatten()]).T),xq.shape)
    rgi = interpolate.RBFInterpolator(np.array((XX.ravel()-wrfOffset[0]-txPos[0]/1000,YY.ravel()-wrfOffset[1]-txPos[1]/1000,ZZ.ravel()-wrfOffset[2]-txPos[2]/1000)).T,WW.ravel(),neighbors=4,kernel='linear')
    winterp = np.reshape(rgi(np.array([xq.flatten(),yq.flatten(),alts.flatten()]).T),xq.shape)
    rgi = interpolate.RBFInterpolator(np.array((XX.ravel()-wrfOffset[0]-txPos[0]/1000,YY.ravel()-wrfOffset[1]-txPos[1]/1000,ZZ.ravel()-wrfOffset[2]-txPos[2]/1000)).T,rref.ravel(),neighbors=4,kernel='linear')
    zinterp = np.reshape(rgi(np.array([xq.flatten(),yq.flatten(),alts.flatten()]).T),xq.shape)
    
    # uinterp = 10*np.ones(xq.shape)
    # vinterp = 10*np.ones(xq.shape)
    # winterp = 1*np.ones(xq.shape)
    # zinterp = np.ones(xq.shape)*40
    # inds = np.logical_and(np.mod(np.round(xq),12) < 6,np.mod(np.round(yq),12) < 6)
    # zinterp[inds] = 60
    # uinterp[inds] = 20
    # vinterp[inds] = 20
    return uinterp,vinterp,winterp,zinterp

def verifyStats(uu,uinterp,vv,vinterp,ww,winterp,el):
    msk = np.isnan(uu[el,:,:])
    uinterp[0,0] += 1e-3
    u_mae = np.nansum(np.abs(uu[el,:,:]-uinterp))/np.sum(~msk)
    u_rmse = np.sqrt(np.nansum((uu[el,:,:]-uinterp)**2)/np.sum(~msk))
    u_cc = np.corrcoef(np.stack((uinterp[~msk],uu[el,:,:][~msk])))[0,1]
    u_pct = np.sum(~msk)/len(msk.flatten())
    msk = np.isnan(vv[el,:,:])
    vinterp[0,0] += 1e-3
    v_mae = np.nansum(np.abs(vv[el,:,:]-vinterp))/np.sum(~msk)
    v_rmse = np.sqrt(np.nansum((vv[el,:,:]-vinterp)**2)/np.sum(~msk))
    v_cc = np.corrcoef(np.stack((vinterp[~msk],vv[el,:,:][~msk])))[0,1]
    v_pct = np.sum(~msk)/len(msk.flatten())
    msk = np.isnan(ww[el,:,:])
    winterp[0,0] += 1e-3
    w_mae = np.nansum(np.abs(ww[el,:,:]-winterp))/np.sum(~msk)
    w_rmse = np.sqrt(np.nansum((ww[el,:,:]-winterp)**2)/np.sum(~msk))
    w_cc = np.corrcoef(np.stack((winterp[~msk],ww[el,:,:][~msk])))[0,1]
    w_pct = np.sum(~msk)/len(msk.flatten())
    return np.array([u_mae,v_mae,w_mae]),np.array([u_rmse,v_rmse,w_rmse]),np.array([u_cc,v_cc,w_cc]),np.array([u_pct,v_pct,w_pct])

def plotMoments(x,y,xx,yy,vpp,vels,nRx,figsize):
    if nRx > 5:
        fig = plt.figure(figsize=figsize)
        m = int(np.ceil(np.sqrt(nRx)))-1
        n = int(np.ceil(nRx/m))
        axs = fig.subplots(m,n).flatten()
        for i in range(nRx-1):
            pc = axs[i].pcolormesh(xx[i][:,el,:]/1000,yy[i][:,el,:]/1000,vels[i][:,el,:],vmin=-40,vmax=40,cmap='pyart_balance',shading='auto')
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
            rFact = (rxPos2[0,i]-xx[i][:,el,:])**2+(rxPos2[1,i]-yy[i][:,el,:])**2++(rxPos2[2,i]-zz[i][:,el,:])**2
            pc = axs[i].pcolormesh(xx[i][:,el,:]/1000,yy[i][:,el,:]/1000,10*np.log10(vpp[i][:,el,:]*rFact)+45,vmin=-30,vmax=90,cmap=z_cmap)
            #pc = axs[i].pcolormesh(xx[i][:,el,:]/1000,yy[i][:,el,:]/1000,10*np.log10(snr[i][:,el,:]),vmin=-10,vmax=60,cmap='pyart_HomeyerRainbow')
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
        # for ax in fig.get_axes():
        #     ax.set_aspect('equal')
        for i in range(nRx):
            # j = 0
            rFact = (rxPos2[1,i]-yy[i][:,el,:])**2+(rxPos2[0,i]-xx[i][:,el,:])**2
            # pc = axs[0,i].pcolormesh(xx[i][:,el,:]/1000,yy[i][:,el,:]/1000,10*np.log10(hpp[i][:,el,:]*rFact)+45,vmin=-30,vmax=90,cmap=z_cmap)
            # axs[0,i].scatter(x=0,y=0,marker='x',s=100)
            # axs[0,i].scatter(x=rxPos2[0,i]/1000,y=rxPos2[1,i]/1000,marker='+',s=100)
            # if i == nRx-1:
            #     axs[0,i].set_title('Tx Simulated Horiz. Range-Corrected Power')
            #     cb = plt.colorbar(pc,ax=axs[0,i])
            #     cb.set_label('Range-Corrected Power (dB)')
            # else:
            #     axs[0,i].set_title('Rx '+ str(i+1) +' Horiz. Simulated Range-Corrected Power')
            # axs[0,i].set_xlabel('Zonal Distance from Tx Radar (km)')
            # axs[0,i].set_ylabel('Meridional Distance from Tx Radar (km)')
            # axs[0,i].set_xlim(x)
            # axs[0,i].set_ylim(y)
        
            j = 0
            ax = axs[j,i]
            pc = ax.pcolormesh(xx[i][:,el,:]/1000,yy[i][:,el,:]/1000,10*np.log10(vpp[i][:,el,:]*rFact)+55,vmin=-30,vmax=90,cmap=z_cmap)
            ax.scatter(x=0,y=0,marker='x',s=100)
            ax.scatter(x=rxPos2[0,i]/1000,y=rxPos2[1,i]/1000,marker='+',s=100)
            if i == nRx-1:
                ax.set_title('Tx Simulated Vert. Range-Corrected Power')
                cb = plt.colorbar(pc,ax=ax)
                cb.set_label('Range-Corrected Power (dB)')
            else:
                ax.set_title('Rx '+ str(i+1) +' Vert. Simulated Range-Corrected Power')
            ax.set_xlabel('Zonal Distance from Tx Radar (km)')
            ax.set_ylabel('Meridional Distance from Tx Radar (km)')
            ax.set_xlim(x)
            ax.set_ylim(y)
            # plt.tight_layout()
            # pc = axs[j,i].pcolormesh(xx[i][:,el,:]/1000,yy[i][:,el,:]/1000,rhohv[i][:,el,:],vmin=0.5,vmax=1,cmap=cc_cmap)
            # axs[j,i].scatter(x=0,y=0,marker='x',s=100)
            # axs[j,i].scatter(x=rxPos2[0,i]/1000,y=rxPos2[1,i]/1000,marker='+',s=100)
            # if i == nRx-1:
            #     axs[j,i].set_title('Tx Simulated Vert. Range-Corrected Power')
            #     cb = plt.colorbar(pc,ax=axs[j,i])
            #     cb.set_label('Range-Corrected Power (dB)')
            # else:
            #     axs[j,i].set_title('Rx '+ str(i+1) +' Vert. Simulated Range-Corrected Power')
            # axs[j,i].set_xlabel('Zonal Distance from Tx Radar (km)')
            # axs[j,i].set_ylabel('Meridional Distance from Tx Radar (km)')
            # axs[j,i].set_xlim(x)
            # axs[j,i].set_ylim(y)
            
            j=1
            ax = axs[j,i]
            pc = ax.pcolormesh(xx[i][:,el,:]/1000,yy[i][:,el,:]/1000,vels[i][:,el,:],vmin=-100,vmax=100,cmap=v_cmap,shading='auto')
            #ax.quiver(XX[::2,::2,1],YY[::2,::2,1],UU[::2,::2,1],VV[::2,::2,1],scale = 600)
            ax.set_xlim(x)
            ax.set_ylim(y)
            ax.scatter(x=0,y=0,marker='x',s=100)
            ax.scatter(x=rxPos2[0,i]/1000,y=rxPos2[1,i]/1000,marker='+',s=100)
            if i == nRx-1:
                ax.set_title('Tx Simulated Radial Velocity')
                cb = plt.colorbar(pc,ax=ax)
                cb.set_label('Bistatic Velocity (m/s)')
            else:
                ax.set_title('Rx '+ str(i+1) +' Simulated Bistatic Velocity')
                
            # rFact = (rxPos2[1,i]-yy[i][:,el,:])**2+(rxPos2[0,i]-xx[i][:,el,:])**2
            # pc = ax.pcolormesh(xx[i][:,el,:]/1000,yy[i][:,el,:]/1000,10*np.log10(hpp[i][:,el,:]/vpp[i][:,el,:]),vmin=-6,vmax=6,cmap=z_cmap)
            # ax.scatter(x=0,y=0,marker='x',s=100)
            # ax.scatter(x=rxPos2[0,i]/1000,y=rxPos2[1,i]/1000,marker='+',s=100)
            # if i == nRx-1:
            #     ax.set_title('Tx Simulated Differential Range-Corrected Power')
            #     cb = plt.colorbar(pc,ax=ax)
            #     cb.set_label('Diff. Range-Corrected Power (dB)')
            # else:
            #     ax.set_title('Rx '+ str(i+1) +' Differential Simulated Range-Corrected Power')
            # ax.set_xlabel('Zonal Distance from Tx Radar (km)')
            # ax.set_ylabel('Meridional Distance from Tx Radar (km)')
            # ax.set_xlim(x)
            # ax.set_ylim(y)
            ax.label_outer()
        plt.tight_layout()

def plotMultiDop(x,y,xc,yc,xq,yq,zq,uu,vv,ww,uinterp,vinterp,winterp,el,rxPos2,rmse):
    fig = plt.figure(figsize=(16,10))
    ax=fig.add_subplot(231)
    pc = ax.pcolormesh(xq[el,:,:]/1000,yq[el,:,:]/1000,uinterp,vmin=-50,vmax=50,cmap='pyart_balance',shading='auto')
    cb = plt.colorbar(pc,ax=ax)
    cb.set_label('U Wind (m/s)')
    ax.set_xlim(x)
    ax.set_ylim(y)
    ax.scatter(x=0,y=0,marker='x',color='k',s=100)
    ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
    ax.set_title('Model Input U Wind')
    ax.set_xlabel('Zonal Distance from Tx Radar (km)')
    ax.set_ylabel('Meridional Distance from Tx Radar (km)')
    ax=fig.add_subplot(232)
    pc = ax.pcolormesh(xq[el,:,:]/1000,yq[el,:,:]/1000,uu[el,:,:],vmin=-50,vmax=50,cmap='pyart_balance',shading='auto')
    cb = plt.colorbar(pc,ax=ax)
    cb.set_label('V Wind (m/s)')
    ax.set_xlim(x)
    ax.set_ylim(y)
    ax.scatter(x=0,y=0,marker='x',color='k',s=100)
    ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
    ax.set_title('Retrieved U Wind')
    ax.set_xlabel('Zonal Distance from Tx Radar (km)')
    ax.set_ylabel('Meridional Distance from Tx Radar (km)')
    ax=fig.add_subplot(233)
    pc = ax.pcolormesh(xq[el,:,:]/1000,yq[el,:,:]/1000,uu[el,:,:]-uinterp,vmin=-10,vmax=10,cmap='pyart_balance',shading='auto')
    cb = plt.colorbar(pc,ax=ax)
    cb.set_label('U Wind Error (m/s)')
    ax.set_xlim(x)
    ax.set_ylim(y)
    ax.scatter(x=0,y=0,marker='x',color='k',s=100)
    ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
    ax.set_title('Retrieved U Wind Error')
    ax.set_xlabel('Zonal Distance from Tx Radar (km)')
    ax.set_ylabel('Meridional Distance from Tx Radar (km)')
    window = (x[1]-x[0])/2
    ax.annotate(f'U MAE: {np.round(rmse[0],2)} m/s',(xc,yc-(window*1.35)),annotation_clip=False,ha='center')
    #plt.tight_layout()
    
    # #fig = plt.figure(figsize=(16,6))
    ax=fig.add_subplot(234)
    pc = ax.pcolormesh(xq[el,:,:]/1000,yq[el,:,:]/1000,vinterp,vmin=-50,vmax=50,cmap='pyart_balance',shading='auto')
    cb = plt.colorbar(pc,ax=ax)
    cb.set_label('V Wind (m/s)')
    ax.set_xlim(x)
    ax.set_ylim(y)
    ax.scatter(x=0,y=0,marker='x',color='k',s=100)
    ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
    ax.set_title('Model Input V Wind')
    ax.set_xlabel('Zonal Distance from Tx Radar (km)')
    ax.set_ylabel('Meridional Distance from Tx Radar (km)')
    ax=fig.add_subplot(235)
    pc = ax.pcolormesh(xq[el,:,:]/1000,yq[el,:,:]/1000,vv[el,:,:],vmin=-50,vmax=50,cmap='pyart_balance',shading='auto')
    cb = plt.colorbar(pc,ax=ax)
    cb.set_label('V Wind (m/s)')
    ax.set_xlim(x)
    ax.set_ylim(y)
    ax.scatter(x=0,y=0,marker='x',color='k',s=100)
    ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
    ax.set_title('Retrieved V Wind')
    ax.set_xlabel('Zonal Distance from Tx Radar (km)')
    ax.set_ylabel('Meridional Distance from Tx Radar (km)')
    ax=fig.add_subplot(236)
    pc = ax.pcolormesh(xq[el,:,:]/1000,yq[el,:,:]/1000,vv[el,:,:]-vinterp,vmin=-10,vmax=10,cmap='pyart_balance',shading='auto')
    cb = plt.colorbar(pc,ax=ax)
    cb.set_label('V Wind Error (m/s)')
    ax.set_xlim(x)
    ax.set_ylim(y)
    ax.scatter(x=0,y=0,marker='x',color='k',s=100)
    ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
    ax.set_title('Retrieved V Wind Error')
    ax.set_xlabel('Zonal Distance from Tx Radar (km)')
    ax.set_ylabel('Meridional Distance from Tx Radar (km)')
    ax.annotate(f'V MAE: {np.round(rmse[1],2)} m/s',(xc,yc-(window*1.35)),annotation_clip=False,ha='center')
    plt.tight_layout()
    
    fig = plt.figure(figsize=(16,6))
    ax=fig.add_subplot(131)
    pc = ax.pcolormesh(xq[el,:,:]/1000,yq[el,:,:]/1000,winterp,vmin=-50,vmax=50,cmap='pyart_balance',shading='auto')
    cb = plt.colorbar(pc,ax=ax)
    cb.set_label('W Wind (m/s)')
    ax.set_xlim(x)
    ax.set_ylim(y)
    ax.scatter(x=0,y=0,marker='x',color='k',s=100)
    ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
    ax.set_title('Model Input W Wind')
    ax.set_xlabel('Zonal Distance from Tx Radar (km)')
    ax.set_ylabel('Meridional Distance from Tx Radar (km)')
    ax=fig.add_subplot(132)
    pc = ax.pcolormesh(xq[el,:,:]/1000,yq[el,:,:]/1000,ww[el,:,:],vmin=-50,vmax=50,cmap='pyart_balance',shading='auto')
    cb = plt.colorbar(pc,ax=ax)
    cb.set_label('W Wind (m/s)')
    ax.set_xlim(x)
    ax.set_ylim(y)
    ax.scatter(x=0,y=0,marker='x',color='k',s=100)
    ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
    ax.set_title('Retrieved W Wind')
    ax.set_xlabel('Zonal Distance from Tx Radar (km)')
    ax.set_ylabel('Meridional Distance from Tx Radar (km)')
    ax=fig.add_subplot(133)
    pc = ax.pcolormesh(xq[el,:,:]/1000,yq[el,:,:]/1000,ww[el,:,:]-winterp,vmin=-10,vmax=10,cmap=v_cmap,shading='auto')
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
                   [0.333,  0.105, 0.105],
                   [0.499,  1.0, 1.0],
                   [0.5,  0.938, 0.938],
                   [0.583,  0.725, 0.725],
                   [0.666,  0.196, 0.196],
                   [0.749,  0.0, 0.0],
                   [0.75,  0.0, 0.0],
                   [0.875,  1.0, 1.0],
                   [1.0,  0.098, 0.098]],
         'blue':  [[0.0,  0.0, 0.0],
                   [0.25,  0.782, 0.782],
                   [0.333,  0.902, 0.902],
                   [0.499,  1.0, 1.0],
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
cdict = {'red':   [[0.0,  0.0, 0.0],
                   [0.5,  0.0, 0.0],
                   [0.85,  1.0, 1.0],
                   [1.0,  1.0, 1.0]],
         'green': [[0.0,  0.0, 0.0],
                   [0.25, 0.0, 0.0],
                   [0.5,  0.0, 0.0],
                   [0.85,  1.0, 1.0],
                   [1.0,  0.0, 0.0]],
         'blue':  [[0.0,  1.0, 1.0],
                   [0.5,  1.0, 1.0],
                   [1.0,  0.1, 0.1]]}
cc_cmap = colors.LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)

maes = np.zeros((14,3))
rmses = np.zeros((14,3))
ccs = np.zeros((14,3))
pcts = np.zeros((14,3))
fp = '/Users/semmerson/Documents/python/data/PRS/abeil'
os.chdir(fp)
l = np.sort([x for x in os.listdir('.') if re.search('ideal',x)])
#l = l[np.arange(50)!=30]
#l = l[3:4]
# l = np.sort(os.listdir('.'))
#l = l[np.array([7,19,55,80,92])]
for num,file in enumerate(l): 
    os.chdir(fp)
    print(file)
    radarStruct,wxStruct,radars = loadRadars(file)
    
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
    # idealPat = radarStruct['idealPat']
    # whitening = radarStruct['whitening']
    # dishMode = radarStruct['dishMode']
    # tmMode = radarStruct['tmMode']
    # saveMode = radarStruct['saveMode']
    
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
    
    os.chdir('/Users/semmerson/Documents/MATLAB/bistaticSimulator')
    f = loadmat('wrfDirectory.mat') #load NWP directory
    XX = f['XX'][:]-wrfOffset[0]
    YY = f['YY'][:]-wrfOffset[1]
    ZZ = f['ZZ'][:]
    
    hpp = radars['hpp']
    vpp = radars['vpp']
    snr = radars['snr']
    rr = radars['rr']
    df = radars['df']
    br = radars['br']
    beta = radars['beta']
    vels = []
    xx = []
    yy = []
    zz = []
    
    snrThresh = 10**(12/10)
    brThresh = 1e3
    
    txrxDist = np.sqrt(np.sum((txPos-rxPos)**2,axis=1))
    rxPhi = np.arctan2(rxPos[:,1]-txPos[1],rxPos[:,0]-txPos[0])
    rxTheta = np.arccos(rxPos[:,2]/txrxDist)
    rxTheta[-1] = 0
    rxPos2 = np.array([np.cos(rxPhi)*np.sin(rxTheta), np.sin(rxPhi)*np.sin(rxTheta),np.cos(rxTheta)])*txrxDist
    txPos2 = np.zeros(3)
    for i in range(nRx):
        snr.append(vpp[i]/N0)
        hpp[i][snr[i] < snrThresh] = np.nan
        vpp[i][snr[i] < snrThresh] = np.nan
        df[i][snr[i] < snrThresh] = np.nan
        vel = np.zeros(df[i].shape)
        xx0 = np.zeros(df[i].shape)
        yy0 = np.zeros(df[i].shape)
        zz0 = np.zeros(df[i].shape)
        inds = np.isnan(df[i])
        df[i][inds] = 0
        for el in range(nEl):
            pts = localize(90-txAz,radarStruct['txMechEl'][el]*np.ones(txAz.shape),rr[i]/c,rxPos2[:,i],txPos2)
            vel[:,el,:] = freq2vel(df[i][:,el,:],pts[:,:,0],pts[:,:,1],pts[:,:,2],rxPos2[:,i],txPos2,_lambda)
            xx0[:,el,:] = pts[:,:,0]
            yy0[:,el,:] = pts[:,:,1]
            zz0[:,el,:] = pts[:,:,2]
        vel[inds] = np.nan
        vels.append(vel)
        xx.append(xx0)
        yy.append(yy0)
        zz.append(zz0)
    el = 0
    figsize = (14,8)
    
    rxc = 0e3
    ryc = 0e3
    rwindow = 15*1e3
    roiLimsX = (rxc-rwindow ,rxc+rwindow)
    roiLimsY = (ryc-rwindow ,ryc+rwindow)
    roiLimsZ = (0,2e3)
    betaThresh = [20,160]
    brThresh = 5e3
    res = 0.3e3
    
    # rmses = np.zeros((3,7,5))
    # ccs = np.zeros((3,7,5))
    # for d in np.arange(7):
    #     print(d)
    #     for n in np.arange(2,7):
    #         print(n)
    #         snrThresh = 10**d
    nx = int(np.diff(roiLimsX)[0]//res)
    ny = int(np.diff(roiLimsY)[0]//res)
    nz = int(np.diff(roiLimsZ)[0]//res)
    
    refgrids,velgrids = gridMoments(xx,yy,zz,vpp,vels,roiLimsX,roiLimsY,roiLimsZ,nx,ny,nz)
    xq = np.linspace(roiLimsX[0],roiLimsX[1]-res,nx)
    yq = np.linspace(roiLimsY[0],roiLimsY[1]-res,ny)
    zq = np.linspace(roiLimsZ[0]+res,roiLimsZ[1],nz)
    zq,yq,xq = np.meshgrid(zq,yq,xq,indexing='ij')
    refgrids[refgrids.mask] = np.nan
    # refgrids[refgrids > 50] = np.nan
    # velgrids[velgrids < -999] = np.nan
    # for i in np.arange(3,15):
    #     print(i)
    
    uinterp,vinterp,winterp,zinterp = interpWrf(xq/1000,yq/1000,zq/1000,el)
    
    # for i in np.arange(3,9):
    #     print(i)
    # for i in np.arange(3,6):
    uu,vv,ww = multiDop(xq,yq,zq,rxPos2,refgrids,velgrids,nRx,1,250,-15)
    uinterp = 10*np.ones_like(uu[el,:,:])
    vinterp = 10*np.ones_like(uu[el,:,:])
    winterp = 1*np.ones_like(uu[el,:,:])
    inds = np.logical_and(np.mod(np.round(xq[el,:,:]/1000),4) < 2,np.mod(np.round(yq[el,:,:]/1000),4) < 2)
    inds = np.logical_and(np.mod(np.round(xq[el,:,:]/1000+wrfOffset[0]+txPos[0]/1000),12) < 6,np.mod(np.round(yq[el,:,:]/1000+wrfOffset[1]+txPos[1]/1000),12) < 6)
    uinterp[inds] = 20
    vinterp[inds] = 20
    el = 0
    mae,rmse,cc,pct = verifyStats(uu,uinterp,vv,vinterp,ww,winterp,el)
    print('U MAE:',np.round(mae[0],2))
    print('U RMSE:',np.round(rmse[0],2))
    print('U CC:',np.round(cc[0],2))
    print('V MAE:',np.round(mae[1],2))
    print('V RMSE:',np.round(rmse[1],2))
    print('V CC:',np.round(cc[1],2))
    print('W MAE:',np.round(mae[2],2))
    print('W RMSE:',np.round(rmse[2],2))
    print('W CC:',np.round(cc[2],2))
    
    # numc = num % 4
    # numr = num // 4
    maes[num,:] = mae
    rmses[num,:] = rmse
    ccs[num,:] = cc
    pcts[num,:] = pct
        
    
xc = 0
yc = 0
window = 25
x = (xc-window, xc+window)
y = (yc-window, yc+window)
s = 1
plotMoments(x,y,xx,yy,vpp,vels,nRx,figsize)
plotMultiDop(x,y,xc,yc,xq,yq,zq,uu,vv,ww,uinterp,vinterp,winterp,el,rxPos2,mae)

# fig = plt.figure(figsize=(14,6))
# ax=fig.add_subplot(121)
# pc = ax.pcolormesh(xq/1000,yq/1000,zinterp,vmin=10,vmax=60,cmap='pyart_HomeyerRainbow',shading='auto')
# #pc = ax.pcolormesh(xq/1000,yq/1000,winterp,vmin=-40,vmax=40,cmap='pyart_balance',shading='auto')
# cb = plt.colorbar(pc,ax=ax)
# cb.set_label('Model Reflectivity (dBZ)')
# #ax.quiver(yq[::2,::2]/1000,xq[::2,::2]/1000,vv[::2,::2],uu[::2,::2],scale=1000,width=0.0015)
# ax.quiver(xq[::s,::s]/1000,yq[::s,::s]/1000,uinterp[::s,::s],vinterp[::s,::s],scale=1000,width=0.0015)
# ax.set_xlim(x)
# ax.set_ylim(y)
# ax.scatter(x=0,y=0,marker='x',color='k',s=100)
# ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
# ax.set_title('Model Input Reflectivity and Horizontal Winds')
# ax.set_xlabel('Zonal Distance from Tx Radar (km)')
# ax.set_ylabel('Meridional Distance from Tx Radar (km)')
# #s = 4
# ax=fig.add_subplot(122)
# pc = ax.pcolormesh(xq/1000,yq/1000,10*np.log10(np.nanmean(refgrids,axis=0)[l,:,:])+145,vmin=10,vmax=60,cmap='pyart_HomeyerRainbow',shading='auto')
# cb = plt.colorbar(pc,ax=ax)
# cb.set_label('Range-Corrected Power (dB)')
# ax.quiver(xq[::s,::s]/1000,yq[::s,::s]/1000,uu[l,::s,::s],vv[l,::s,::s],scale=1000,width=0.0015)
# #plt.quiver(XX[::2]-wrfOffset[0]-txPos[0]/1000,YY[::2]-wrfOffset[1]-txPos[1]/1000,UU[::2,::2,0].T,VV[::2,::2,0].T,color='blue',scale=1000,width=0.0015)
# ax.set_xlim(x)
# ax.set_ylim(y)
# ax.scatter(x=0,y=0,marker='x',color='k',s=100)
# ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
# ax.set_title('Retrieved Range-Corrected Power and Horizontal Winds')
# ax.set_xlabel('Zonal Distance from Tx Radar (km)')
# ax.set_ylabel('Meridional Distance from Tx Radar (km)')
# plt.tight_layout()


# scene=dict(camera=dict(eye=dict(x=1.15, y=1.15, z=0.8)), #the default values are 1.25, 1.25, 1.25
#             # xaxis = dict(nticks=5, range=[-60,60],),
#             # yaxis = dict(nticks=5, range=[-60,60],),
#             # zaxis = dict(nticks=5, range=[0,20],),
#             aspectmode='data', #this string can be 'data', 'cube', 'auto', 'manual'
#             #a custom aspectratio is defined as follows:
#             #aspectratio=dict(x=1, y=1, z=1/6)
#             )
# layout = go.Layout(scene=scene)
# z,y,x = np.mgrid[roiLimsZ[0]:roiLimsZ[1]:nz*1j,roiLimsY[0]:roiLimsY[1]:ny*1j,roiLimsX[0]:roiLimsX[1]:nx*1j]
# fig = go.Figure(data=go.Volume(
#     x=x.flatten(),
#     y=y.flatten(),
#     z=z.flatten(),
#     value=(ww-winterp).flatten(),
#     opacity=0.2,  
#     opacityscale=[[0, 1], [0.5, 0.1], [1,1]], 
#     colorscale='RdBu',
#     isomin = -30,
#     isomax = 30,
#     surface_count=7,
#     caps=dict(x_show=False, y_show=False, z_show=False)
#     ),layout=layout)
# plot(fig)


