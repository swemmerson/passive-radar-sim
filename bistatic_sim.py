#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:53:46 2020

@author: semmerson
"""
import os, pickle, lzma
import numpy as np
import numexpr as ne
import matplotlib.colors as colors
import datetime as dt
from netCDF4 import Dataset
from scipy.fftpack import fft2, fftshift
from scipy import interpolate, signal
#from skimage.restoration import unwrap_phase
from scipy.io import loadmat
from numba import njit
#from pyart.map.grid_mapper import NNLocator
from timeit import default_timer as timer

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
cdict = {'red':   [[0.0,  0.196, 0.196],
                   [0.2,  0.196, 0.0],
                   [0.7,  0.0, 0.0],
                   [0.8,  0.588, 0.588],
                   [0.9,  0.902, 0.902],
                   [1.0,  0.471, 0.471]],
         'green': [[0.0,  0.196, 0.196],
                   [0.2,  0.196, 0.196],
                   [0.7, 0.0, 0.0],
                   [0.8,  0.51, 1.0],
                   [0.9,  0.588, 0.471],
                   [1.0,  0.0, 0.0]],
         'blue':  [[0.0,  0.196, 0.196],
                   [0.2,  0.196, 0.196],
                   [0.7,  1.0, 1.0],
                   [0.8,  1.0, 0.275],
                   [0.9,  0.0, 0.235],
                   [1.0,  0.231, 0.231]]}
cc_cmap = colors.LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
cdict = {'red':   [[0.0,     0.020, 0.020],
                   [0.1875,  0.039, 0.039],
                   [0.3438,  0.110, 0.078],
                   [0.3672,  0.110, 0.110],
                   [0.375,   0.471, 0.471],
                   [0.3828,  0.0,   0.0],
                   [0.4375,  0.392, 1.0],
                   [0.4688,  0.855, 0.855],
                   [0.5313,  0.502, 0.502],
                   [0.5625,  0.882, 0.882],
                   [0.6094,  0.471, 0.471],
                   [0.6875,  0.353, 0.353],
                   [1.0,     0.961, 0.961]],
         
         'green': [[0.0,     0.118, 0.118],
                   [0.1875,  0.251, 0.251],
                   [0.3438,  0.706, 0.706],
                   [0.3672,  0.902, 0.902],
                   [0.375,   0.471, 0.471],
                   [0.3828,  0.0,   0.0],
                   [0.4375,  0.863, 1.0],
                   [0.4688,  0.471, 0.471],
                   [0.5313,  0.0,   0.0],
                   [0.5625,  0.345, 0.345],
                   [0.6094,  0.0,   0.0],
                   [0.6875,  0.0,   0.0],
                   [1.0,     0.961, 0.961]],
         
         'blue':  [[0.0,     0.357, 0.357],
                   [0.1875,  0.251, 0.251],
                   [0.3438,  0.706, 0.706],
                   [0.3672,  0.902, 0.902],
                   [0.375,   0.471, 0.471],
                   [0.3828,  0.596, 0.596],
                   [0.4375,  1.0,   0.376],
                   [0.4688,  0.0,   0.0],
                   [0.5313,  0.0,   0.0],
                   [0.5625,  0.882, 0.882],
                   [0.6094,  0.471, 0.471],
                   [0.6875,  0.353, 0.353],
                   [1.0,     0.784, 0.784]]}
zdr_cmap = colors.LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)

# Edit these to configure your Tx radar
def makeRadarStruct(txPos,rxPos,modes=[True,False,False]):
    radar = {}
    radar['txPos'] = txPos #Tx position
    radar['rxPos'] = rxPos #Rx position 
    radar['lambda'] = 0.1031 #Operating wavelength
    radar['prt'] = 1/2000  #Tx PRT (s)
    radar['tau'] = 1.57e-6 #Pulse width (s)
    radar['fs'] = 1/radar['tau'] #Sampling Rate samples/s
    radar['Pt'] = 1000e3 #Peak transmit power (W)
    radar['Tant'] = 63.36 #Antenna noise temp/brightness
    radar['Frx'] = 3 #Rx noise figure
    radar['M'] = 8 #Pulses per dwell
    radar['rxMechTilt'] = 1.0 #Mechanical tilt of Rx antenna
    radar['txMechEl'] = np.array([0.5])#Mechanical Tx elevation tilt 
    radar['txMechAz'] = 1 #Baseline mechanical Tx position
    radar['txAz'] = np.mod(np.linspace(0,360,721),360) #Transmit azimuths
    radar['txEl'] = np.zeros(radar['txAz'].shape) #Electrical tx steering elv.
    radar['txG'] = 45.5 #Transmit antenna gain (dBi)
    radar['rxG'] = 25 #Receive antenna gain (dBi)
    radar['receiverGain'] = 50 #Receiver gain (dB)
    radar['idealPat'] = modes[0] #Ideal pattern (pencil beam + no sidelobes)?
    radar['whitening'] = modes[1] #Whiten sidelobes?
    radar['dishMode'] = modes[2] #Emulate WSR-88D transmit pattern?
    radar['tmMode'] = True #T-matrix scattering?
    radar['saveMode'] = True #Save radar outputs?
    radar['ptypes'] = ['rain','hail'] #Precipitation type(s) for T-matrix scattering
    radar['attenuationMode'] = True #Include attenuation + phase shift?
    radar['scanType'] = 'ppi'
    return radar

# Edit these to configure the radar simulation domain
def makeWxStruct():
    wx = {}
    wx['scatMode'] = 'rayleigh' #Rayleigh or Bragg scattering
    wx['xSize'] = 50e3 #x,y,z dimensions of simulation volume (m)
    wx['ySize'] = 50e3
    wx['zSize'] = 0.5e3
    wx['getZh'] = 'Zh=35*np.ones((1,nPts))' #Zh/Zv values for uniform test case
    wx['getZv'] = 'Zv=Zh'              #not currently used
    wx['wrfDate'] = '2013-06-01 01:20:10' #NWP data query date: yyyy-mm-dd HH:MM:SS
    wx['wrfOffset'] = np.array([-10.0,0.0,0.25]) # offset of simulation volume np.array([-16,-6,0.5])
                                               # from origin within NWP data                                           
    wx['spw'] = 2 #Spectrum width
    wx['ptsPerMinVol'] = 3 #Number of points in smallest resolution volume
    return wx

def datenum64(d):
    return 366 + d.item().toordinal() + (d.item() - dt.datetime.fromordinal(d.item().toordinal())).total_seconds()/(24*60*60)

def getWrf(xq,yq,zq,ptypes):
    #Unit tested    
    fh = Dataset('cm1out_000150.nc', 'r')
    XX = fh.variables['xh'][:]
    YY = fh.variables['yh'][:]
    ZZ = fh.variables['zh'][:]
    UU = np.squeeze(fh.variables['uinterp'][:])
    VV = np.squeeze(fh.variables['vinterp'][:])
    WW = np.squeeze(fh.variables['winterp'][:])
    ref = np.squeeze(fh.variables['dbz'][:])
    # ntr = fh.variables['ncr'][0,:,:,:]
    # qr = fh.variables['qr'][0,:,:,:]
    # ntg = fh.variables['ncg'][0,:,:,:]
    # qg = fh.variables['qg'][0,:,:,:]
    # nts = fh.variables['ncs'][0,:,:,:]
    # qs = fh.variables['qs'][0,:,:,:]
    # fh = Dataset('cm1out.nc', 'r')
    # XX = fh.variables['x'][:]
    # YY = fh.variables['y'][:]
    # ZZ = fh.variables['z'][:]
    rhow = 997
    rhos = 100
    #rhog = fh.variables['qh'][:]/fh.variables['vhl'][:]
    rhog = np.squeeze(fh.variables['qhl'][:])/np.squeeze(fh.variables['vhl'][:])
    cr = rhow*np.pi/6
    cs = rhos*np.pi/6
    dr = 3
    ds = 3
    pr = 0
    ps = 0
    lambda_r = (cr*np.squeeze(fh.variables['crw'][:])*np.math.factorial(pr+dr)/np.squeeze(fh.variables['qr'][:]))**(1/dr)/1000
    #lambda_r = np.ones(lambda_r.shape)*3
    n0_r = np.squeeze(fh.variables['crw'][:])*lambda_r
    #n0_r = np.ones(n0_r.shape)*1000
    #lambda_g = (np.pi*rhog*fh.variables['chl'][:]/fh.variables['qh'][:])**(1/3)/1000
    lambda_g = (np.pi*rhog*np.squeeze(fh.variables['chl'][:])/np.squeeze(fh.variables['qhl'][:]))**(1/3)/1000
    n0_g = np.squeeze(fh.variables['chl'][:])*lambda_g
    lambda_s = (cs*np.squeeze(fh.variables['csw'][:])*np.math.factorial(ps+ds)/np.squeeze(fh.variables['qs'][:]))**(1/ds)/1000
    n0_s = np.squeeze(fh.variables['csw'][:])*lambda_s
    # UU = fh.variables['u'][:]
    # VV = fh.variables['v'][:]
    # WW = fh.variables['w'][:]
    #ref = fh.variables['reflectivity'][:]

    # rhow = 997
    # rhog = 900
    # rhos = 100
    # cr = rhow*np.pi/6
    # cg = rhog*np.pi/6
    # cs = rhos*np.pi/6
    # dr = 3
    # dg = 3
    # ds = 3
    # pr = 0
    # pg = 0
    # ps = 0
    # lambda_r = (cr*ntr*np.math.factorial(pr+dr)/qr)**(1/dr)/1000
    # n0_r = ntr*lambda_r
    # lambda_g = (cg*ntg*np.math.factorial(pg+dg)/qg)**(1/dg)/1000
    # n0_g = ntg*lambda_g
    # lambda_s = (cs*nts*np.math.factorial(ps+ds)/qs)**(1/ds)/1000
    # n0_s = nts*lambda_s
    # fh = Dataset('wrfwof_d01_2020-08-10_18:00:00','r')
    # UU = np.swapaxes(np.squeeze(fh.variables['U'][:]),0,2)[:-1,:,:]
    # VV = np.swapaxes(np.squeeze(fh.variables['V'][:]),0,2)[:,:-1,:]
    # WW = np.swapaxes(np.squeeze(fh.variables['W'][:]),0,2)[:,:,:-1]
    # ref = np.swapaxes(np.squeeze(fh.variables['REFL_10CM'][:]),0,2)
    # XX = (np.arange(ref.shape[0])-ref.shape[0]/2)*3
    # YY = (np.arange(ref.shape[1])-ref.shape[1]/2)*3
    # ZZ = np.arange(ref.shape[2])*0.5
    #XX,YY,ZZ = np.meshgrid(XX,YY,ZZ)
    # UU = np.swapaxes(fh.variables['u'][:],0,1)
    # VV = np.swapaxes(fh.variables['v'][:],0,1)
    # WW = np.swapaxes(fh.variables['w'][:],0,1)
    # ref = np.swapaxes(fh.variables['reflectivity'][:],0,1)

    # UU = np.ones_like(ref)*10
    # VV = np.ones_like(ref)*10
    # WW = np.zeros_like(ref)
    fh.close()
    #Get boundaries of NWP and query points
    pds = {}
    for ptype in ptypes:
        pds[ptype] = {}
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
        rgi = interpolate.RegularGridInterpolator((ZZ,YY,XX),UU)
        uu = rgi(np.array([zq,yq,xq]).T)
        # uu = 50*np.ones(xq.shape)
        
        rgi = interpolate.RegularGridInterpolator((ZZ,YY,XX),VV)
        vv = rgi(np.array([zq,yq,xq]).T)
        # vv = 0*np.ones(xq.shape)
        
        rgi = interpolate.RegularGridInterpolator((ZZ,YY,XX),WW)
        ww = rgi(np.array([zq,yq,xq]).T)
        # ww = 0*np.ones(xq.shape)
        rgi = interpolate.RegularGridInterpolator((ZZ,YY,XX),10**(ref/10))
        Zv = rgi(np.array([zq,yq,xq]).T)
        for ptype in ptypes:
            if ptype in ['dry_hail','hail','wet_hail']:
                rgi = interpolate.RegularGridInterpolator((ZZ,YY,XX),np.log10(lambda_g))
                pds[ptype]['lambda'] = rgi(np.array([zq,yq,xq]).T)
                rgi = interpolate.RegularGridInterpolator((ZZ,YY,XX),np.log10(n0_g))
                pds[ptype]['n0'] = rgi(np.array([zq,yq,xq]).T)  
            elif ptype == 'rain':
                rgi = interpolate.RegularGridInterpolator((ZZ,YY,XX),np.log10(lambda_r))
                pds[ptype]['lambda'] = rgi(np.array([zq,yq,xq]).T)
                rgi = interpolate.RegularGridInterpolator((ZZ,YY,XX),np.log10(n0_r))
                pds[ptype]['n0'] = rgi(np.array([zq,yq,xq]).T)
            elif ptype in ['snow','dry_snow']:
                rgi = interpolate.RegularGridInterpolator((ZZ,YY,XX),np.log10(lambda_s))
                pds[ptype]['lambda'] = rgi(np.array([zq,yq,xq]).T)
                rgi = interpolate.RegularGridInterpolator((ZZ,YY,XX),np.log10(n0_s))
                pds[ptype]['n0'] = rgi(np.array([zq,yq,xq]).T)
        # inds = np.logical_and(np.mod(np.round(xq),12) < 6,np.mod(np.round(yq),12) < 6)
        # ref[inds] = 60
        # uu[inds] = 20
        # vv[inds] = 20
        #ref = np.ones(xq.shape)*60
        return (uu,vv,ww,Zv,pds)
    else:
        print('Specified volume out of bounds.')
    
def init_KA_table(Lambda,ptype,_type):
    fn = f'{np.round(Lambda*1000,1)}mm_{_type}_{ptype}.xz'
    if os.path.exists(fn):
        print('T-matrix file found!')
        with lzma.open(fn,'rb') as f:
            ref = pickle.load(f)
            v = pickle.load(f)
            h = pickle.load(f)
    H_inter = interpolate.RegularGridInterpolator(ref,h)
    V_inter = interpolate.RegularGridInterpolator(ref,v)
    return (H_inter,V_inter)

def getKA(pts,pos,nump,ptypes,table):
    npts = pts.shape[1]
    qpts = np.zeros((3,npts,nump))
    dr = np.zeros((3,npts))
    Ah = np.zeros((npts))
    Av = np.zeros((npts))
    Kh = np.zeros((npts))
    Kv = np.zeros((npts))
    for i in range(3):
        qpts[i,:,:],dr[i,:] = np.linspace(pos[i,np.newaxis],pts[i,:],num=nump,retstep=True,axis=1)
    DR = np.sqrt(np.sum(dr**2,axis=0))
    # qpts = np.reshape(qpts,(3,npts*nump))
    # Aih = np.zeros(npts*nump)
    # Aiv = np.zeros(npts*nump)
    
    fh = Dataset('cm1out.nc', 'r')
    XX = fh.variables['x'][:]
    YY = fh.variables['y'][:]
    ZZ = fh.variables['z'][:]
    rhow = 997
    rhos = 100
    rhog = fh.variables['qh'][:]/fh.variables['vhl'][:]
    cr = rhow*np.pi/6
    cs = rhos*np.pi/6
    dr = 3
    ds = 3
    pr = 0
    ps = 0
    lambda_r = (cr*fh.variables['crw'][:]*np.math.factorial(pr+dr)/fh.variables['qr'][:])**(1/dr)/1000
    n0_r = fh.variables['crw'][:]*lambda_r
    lambda_g = (np.pi*rhog*fh.variables['chl'][:]/fh.variables['qh'][:])**(1/3)/1000
    n0_g = fh.variables['chl'][:]*lambda_g
    lambda_s = (cs*fh.variables['csw'][:]*np.math.factorial(ps+ds)/fh.variables['qs'][:])**(1/ds)/1000
    n0_s = fh.variables['csw'][:]*lambda_s
    # fh = Dataset('cm1out_000489.nc', 'r')
    # XX = fh.variables['xh'][:]
    # YY = fh.variables['yh'][:]
    # ZZ = fh.variables['zh'][:]
    # ntr = fh.variables['ncr'][0,:,:,:]
    # qr = fh.variables['qr'][0,:,:,:]
    # ntg = fh.variables['ncg'][0,:,:,:]
    # qg = fh.variables['qg'][0,:,:,:]
    # nts = fh.variables['ncs'][0,:,:,:]
    # qs = fh.variables['qs'][0,:,:,:]
    # rhow = 997
    # rhog = 900
    # rhos = 100
    # cr = rhow*np.pi/6
    # cg = rhog*np.pi/6
    # cs = rhos*np.pi/6
    # dr = 3
    # dg = 3
    # ds = 3
    # pr = 0
    # pg = 0
    # ps = 0
    # lambda_r = (cr*ntr*np.math.factorial(pr+dr)/qr)**(1/dr)/1000
    # n0_r = ntr*lambda_r
    # lambda_g = (cg*ntg*np.math.factorial(pg+dg)/qg)**(1/dg)/1000
    # n0_g = ntg*lambda_g
    # lambda_s = (cs*nts*np.math.factorial(ps+ds)/qs)**(1/ds)/1000
    # n0_s = nts*lambda_s
    fh.close()

    #Load variables from corresponding files and spatially
    #interpolate

    l_r_i = interpolate.RegularGridInterpolator((ZZ,YY,XX),np.log10(lambda_r),method='nearest')
    n_r_i = interpolate.RegularGridInterpolator((ZZ,YY,XX),np.log10(n0_r),method='nearest')
    l_g_i = interpolate.RegularGridInterpolator((ZZ,YY,XX),np.log10(lambda_g),method='nearest')
    n_g_i = interpolate.RegularGridInterpolator((ZZ,YY,XX),np.log10(n0_g),method='nearest')
    l_s_i = interpolate.RegularGridInterpolator((ZZ,YY,XX),np.log10(lambda_s),method='nearest')
    n_s_i = interpolate.RegularGridInterpolator((ZZ,YY,XX),np.log10(n0_s),method='nearest')    
    for i in range(nump):
        print("Point ",i+1)
        l_r = l_r_i(qpts[::-1,:,i].T)
        n_r = n_r_i(qpts[::-1,:,i].T)
        l_g = l_g_i(qpts[::-1,:,i].T)
        n_g = n_g_i(qpts[::-1,:,i].T)
        l_s = l_s_i(qpts[::-1,:,i].T)
        n_s = n_s_i(qpts[::-1,:,i].T)
        l_r[l_r < -0.5] = -0.5
        l_r[l_r > 1.7] = 1.7
        n_r[n_r < -10] = -10
        n_r[n_r > 7] = 7
        l_g[l_g < -0.5] = -0.5
        l_g[l_g > 1.7] = 1.7
        n_g[n_g < -10] = -10
        n_g[n_g > 7] = 7
        l_s[l_s < -0.5] = -0.5
        l_s[l_s > 1.7] = 1.7
        n_s[n_s < -10] = -10
        n_s[n_s > 7] = 7
        for t in ptypes:
            inters = table[t]['A_inters']
            H_inter = inters[0]
            V_inter = inters[1]
            if t == 'rain':
                Ah += H_inter(np.stack((n_r,l_r)).T)*DR
                Av += V_inter(np.stack((n_r,l_r)).T)*DR
            elif t == 'snow':
                Ah += H_inter(np.stack((n_s,l_s)).T)*DR
                Av += V_inter(np.stack((n_s,l_s)).T)*DR
            else:
                Ah += H_inter(np.stack((n_g,l_g)).T)*DR
                Av += V_inter(np.stack((n_g,l_g)).T)*DR
            inters = table[t]['K_inters']
            H_inter = inters[0]
            V_inter = inters[1]
            if t == 'rain':
                Kh += H_inter(np.stack((n_r,l_r)).T)*DR
                Kv += V_inter(np.stack((n_r,l_r)).T)*DR
            elif t == 'snow':
                Kh += H_inter(np.stack((n_s,l_s)).T)*DR
                Kv += V_inter(np.stack((n_s,l_s)).T)*DR
            else:
                Kh += H_inter(np.stack((n_g,l_g)).T)*DR
                Kv += V_inter(np.stack((n_g,l_g)).T)*DR
    #Compute attenuation field (dB/km)
    
    del qpts
    return (Kh,Kv,Ah,Av)

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
    
    rxPat[0,:] = np.mean(np.stack((rxPat[1,:],rxPat[-1,:])),axis=0)
    rxPat[:,0] = np.mean(np.stack((rxPat[:,1],rxPat[:,-1])),axis=0)
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
#thetaS - forward scatter angle relative to incident beam
    if np.all(np.isclose(rxPos,txPos)): #Pure backscattering geometry
        thetaS =  np.ones(pts.shape[1])*0
    else: #Bistatic geometry
        TS = np.zeros(pts.shape)
        SR = np.zeros(pts.shape)
        
        for ii in range(2):
            TS[ii,:] = txPos[ii] - pts[ii,:]
            SR[ii,:] = pts[ii,:] - rxPos[ii]
        TSmag = np.sqrt(np.sum(TS**2,axis=0))
        SRmag = np.sqrt(np.sum(SR**2,axis=0))
        thetaS = np.arccos(np.sum(TS*SR,axis=0)/(TSmag*SRmag))
        thetaS[np.isnan(thetaS)] = 0.0
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

def init_tmatrix_table(Lambda,ptype,_type):
    fn = f'{np.round(Lambda*1000,1)}mm_{_type}_{ptype}_new.xz'
    print(fn)
    if os.path.exists(fn):
        print('T-matrix file found!')
        with lzma.open(fn,'rb') as f:
            ref = pickle.load(f)
            Svv = pickle.load(f)
            Shh = pickle.load(f)
    else:
        print('T-matrix file not found, generating table...')
        #This is where the T-matrix table gen code goes
        print('womp womp')
    ref = (ref[0],ref[1],ref[2],ref[4],ref[3])
    H_inter = interpolate.RegularGridInterpolator(ref,Shh)
    V_inter = interpolate.RegularGridInterpolator(ref,Svv)
    return (H_inter, V_inter)

def getTmatrixWeights(table,txPos,rxPos,pts,lamb,n0):
    rpts = pts.T - txPos
    rxpts = rxPos - pts.T

    res = cart2sph(rpts[:,0],rpts[:,1],rpts[:,2])
    ize = res[:,1] #Incoming zenith angle is relative to xy-plane in the T-matrix file, so no need to change
    res = cart2sph(rxpts[:,0],rxpts[:,1],rxpts[:,2])
    saz = getThetaS(pts,txPos,rxPos)
    saz[saz > np.pi] = np.pi-np.abs(saz[saz > np.pi]-np.pi)
    saz[saz < 0] = np.abs(saz[saz > np.pi]-np.pi)-np.pi
    
    sze = np.pi/2-res[:,1] #Scattering zenith angle is defined from z-axis up, so we invert and add pi/2
    ize[ize < 0] = 0.0 #Clamp negative zenith angles to 0 
    ize[ize > np.pi] = np.pi #Clamp high zenith angles to pi
    sze[sze < 0] = 0.0 #Clamp negative zenith angles to 0 
    sze[sze > np.pi] = np.pi #Clamp high zenith angles to pi
    lamb[lamb < -0.5] = -0.5
    lamb[lamb > 1.7] = 1.7
    n0[n0 < -10] = -10
    n0[n0 > 7] = 7
    
    HZ_inter = table['Z_inters'][0]
    VZ_inter = table['Z_inters'][1]
    # HD_inter = table['D_inters'][0]
    # VD_inter = table['D_inters'][1]
    Hwts = HZ_inter(np.vstack([saz,sze,ize,lamb,n0]).T)
    Vwts = VZ_inter(np.vstack([saz,sze,ize,lamb,n0]).T)
    # Hsp =  HD_inter(np.vstack([saz,sze,ize,lamb,n0]).T)
    # Vsp =  VD_inter(np.vstack([saz,sze,ize,lamb,n0]).T)
    # Hwts = Hwts*np.exp(-1j*Hsp)
    # Vwts = Vwts*np.exp(-1j*Vsp)
    return Hwts,Vwts

def z2eta(Z,_lambda):
#Z2ETA converts dBZ to eta in m^2/m^3
    Ze = 10**(Z/10)*10**(-18)
    Kw = 0.93
    eta = np.pi**5*Kw*Ze/_lambda**4
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
    idealPat = radarStruct['idealPat']
    whitening = radarStruct['whitening']
    dishMode = radarStruct['dishMode']
    tmMode = radarStruct['tmMode']
    saveMode = radarStruct['saveMode']
    ptypes = radarStruct['ptypes']
    att = radarStruct['attenuationMode']
    scanType = radarStruct['scanType']
    
    scatMode = wxStruct['scatMode'] #Wx or Bragg scatter
    xSize = wxStruct['xSize'] #Sizes of each dimension of Wx volume
    ySize = wxStruct['ySize']
    zSize = wxStruct['zSize']
    wrfDate = wxStruct['wrfDate'] #Date to be queried from WRF data
    
    #Spatial offset of sim volume within WRF volume
    wrfOffset = wxStruct['wrfOffset']
    
    #minimum number of scatterers in smallest volume
    ptsPerMinVol = wxStruct['ptsPerMinVol']    
    
    #os.chdir('/Users/semmerson/Documents/MATLAB/bistaticSimulator')
    if idealPat:
        print('Idealized Gain Pattern')
    elif dishMode:
        print('WSR-88D Gain Pattern')
    elif whitening:
        print('PAWR Gain Pattern w/ Sidelobe Whitening')
    else:
        print('PAWR Gain Pattern w/o Sidelobe Whitening')
    if whitening:
        f = loadmat('gold64.mat')
        c1 = np.squeeze(f['c1'][:])
        c2 = np.squeeze(f['c2'][:])
    else:
        f = loadmat('gold64.mat')
        c1 = np.squeeze(f['c1'][:])
        c1 = np.ones(c1.shape)
        c2 = np.ones(c1.shape)
    
    f = loadmat('win88d.mat')
    baseWin = np.squeeze(f['baseWin'][:])
    f = loadmat('win88dWht5.mat')
    codingWeights1 = np.squeeze(f['totalWts1'][:])
    codingWeights2 = np.squeeze(f['totalWts2'][:])
    
    if dishMode:
        codingWeights1 = baseWin
        codingWeights2 = baseWin

    fftres = 4096;
    wts = np.zeros((len(codingWeights1),len(codingWeights1),4)).astype('complex128')
    wts[:,:,0] = codingWeights1*codingWeights1[:,np.newaxis]
    wts[:,:,1] = codingWeights1*codingWeights2[:,np.newaxis]
    wts[:,:,2] = codingWeights2*codingWeights1[:,np.newaxis]
    wts[:,:,3]= codingWeights2*codingWeights2[:,np.newaxis]
    patMatrix = np.zeros((fftres,fftres,4)).astype('complex128')
    
    #data = loadmat('Pencil.mat')
    # data = loadmat('Spoiled15.mat')
    # newPat = data['pat'].astype('complex128')
    for ii in range(4):
        txPat = fftshift(fft2(wts[:,:,ii],shape = (fftres,fftres)))
        # txPat = newPat
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
    minVolSize = (np.sin(0.5*np.pi/180)*5e3)**2*c*tau/2
    nPts = np.round(ptsPerMinVol * domainVolume/minVolSize)
    ptVolume = domainVolume/nPts
    np.random.seed(777)
    
    #os.chdir('/Users/semmerson/Documents/python')
    niZe = 46
    nsZe = 46
    nsAz = 46
    nr = 40
    if tmMode:
        print("Initializing T-matrix scattering tables...")
        S_table = {}
        for t in ptypes:    
            inters = init_tmatrix_table(_lambda, t,'S')
            S_table[t] = {}
            S_table[t]['Z_inters'] = inters
            # inters = init_tmatrix_table(_lambda, t,'D')
            # S_table[t]['D_inters'] = inters
        if att:
            A_table = {}
            print("Initializing T-matrix attenuation tables...")
            for t in ptypes:
                A_table[t] = {}
                inters = init_KA_table(_lambda, t,'K')
                A_table[t]['K_inters'] = inters
                inters = init_KA_table(_lambda, t,'A')
                A_table[t]['A_inters'] = inters
    if dishMode:
        f = loadmat('dishWts.mat')
        patTheta = f['patTheta'][:]
        txPat = f['txPat'][:]
        dishInter = interpolate.interp1d(patTheta.flatten()*np.pi/180,txPat.flatten(),kind='nearest',fill_value=0,bounds_error=False)

    
    pts = np.zeros((3,int(nPts)))
    pts[0,:] = xSize*np.random.uniform(size=(int(nPts))) - xSize/2
    pts[1,:] = ySize*np.random.uniform(size=(int(nPts))) - ySize/2
    pts[2,:] = zSize*np.random.uniform(size=(int(nPts)))
    Hpower = []
    Vpower = []
    rr = []
    nAz = txAz.shape[0]
    nEl = txMechEl.shape[0]
    if scanType == 'ppi':
        for ll in range(nEl):
            print(f'Elevation {ll+1} of {nEl}')
            #os.chdir('/Users/semmerson/Documents/cm1r19.10/grinder')
            #os.chdir('/Users/semmerson/Documents/python/data/may7/may_7_250m')
            #Zv,uu,vv,ww = getWrf(wrfDate,pts[0,:]/1000+wrfOffset[0],pts[1,:]/1000+wrfOffset[1],pts[2,:]/1000+wrfOffset[2])
            uu,vv,ww,Zv,pds = getWrf(pts[0,:]/1000+wrfOffset[0],pts[1,:]/1000+wrfOffset[1],pts[2,:]/1000+wrfOffset[2],ptypes)
    
            inds = np.where(np.log10(Zv) > 0)[0]
            inds = np.ones(Zv.shape).astype(bool)
            nPts = len(inds)
            pts = pts[:,inds]
            Zv = Zv[inds]
            uu = uu[inds]
            vv = vv[inds]
            ww = ww[inds]
            for ptype in ptypes:
                pds[ptype]['lambda'] = pds[ptype]['lambda'][inds]
                pds[ptype]['n0'] = pds[ptype]['n0'][inds]
            #Vectorize wind velocities
            windSpeed = np.zeros((3,int(nPts)))
            windSpeed[0,:] = uu
            windSpeed[1,:] = vv
            windSpeed[2,:] = ww
            
            inds = np.arange(nPts)
            np.random.shuffle(inds)
            nt = len(ptypes)
            ind_l = [inds[i::nt] for i in range(nt)]
            
            del uu,vv,ww
            if att:
                natt = 50
                piahr = np.zeros((int(nRx),int(nPts)))
                piavr = np.zeros((int(nRx),int(nPts)))
                khr = np.zeros((int(nRx),int(nPts)))
                kvr = np.zeros((int(nRx),int(nPts)))
                print('Calculating Attenuation and Phase Shift:')
                kht,kvt,piaht,piavt = getKA(pts/1000+wrfOffset[:,np.newaxis],txPos/1000+wrfOffset,natt,ptypes,A_table)
                piahr[-1,:] = np.copy(piaht)
                piavr[-1,:] = np.copy(piavt)
                khr[-1,:] = np.copy(kht)
                kvr[-1,:] = np.copy(kvt)
                for i in range(nRx-1):
                    print(f'Receiver {i+1} of {nRx}')
                    khr[i,:],kvr[i,:],piahr[i,:], piavr[i,:] = getKA(pts/1000+wrfOffset[:,np.newaxis],rxPos[i,:]/1000+wrfOffset,natt,ptypes,A_table)
            #os.chdir('/Users/semmerson/Documents/MATLAB/bistaticSimulator')
            staticHwts = np.zeros((int(nRx),int(nPts))).astype('complex64')
            staticVwts = np.zeros((int(nRx),int(nPts))).astype('complex64')
    
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
                Hpower.append(np.zeros((nAz,nEl,nr[i],M)).astype('complex64'))
                Vpower.append(np.zeros((nAz,nEl,nr[i],M)).astype('complex64'))
            
            #Calculate portion of complex scatterer weights that will not change
            #with Tx pointing angle.
            #
            #Makes simplifying assumption that Tx scan happens instantaneously
            azTime = 1
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
    
                    # else:
                    #     Ae = dishInter(rxPhi[ii,:])*_lambda**2/(4*np.pi)
                    #Get forward scatter angles
                    #thetaS = getThetaS(pts,txPos,rxPos[ii,:])
                    #Get dual-pol scattering amplitudes based on infinitesimal
                    #dipole-like reradiation
                    hwt = np.zeros(Zv.shape).astype('complex64')
                    vwt = np.zeros(Zv.shape).astype('complex64')
    
                    if tmMode:
                        for n,t in enumerate(ptypes):
                            inds = ind_l[n]
                            weights = getTmatrixWeights(S_table[t],txPos,rxPos[ii,:],pts[:,inds],pds[t]['lambda'][inds],pds[t]['n0'][inds])
                            hwt[inds] = np.squeeze(weights[0])
                            vwt[inds] = np.squeeze(weights[1])
    
    
                    else:
                        weights = bistaticWeights(txPos,rxPos[ii,:],pts)
                        hwt = np.squeeze(weights[0])
                        vwt = np.squeeze(weights[1])
            
                    # Do the "receive" half of the radar range equation (doesn't
                    # change with Tx pointing angle.  Convert from Z or Cn^2 to
                    # reflectivity (eta) as necessary
                    if scatMode == 'rayleigh':
                        staticHwts[ii,:] = np.sqrt(hwt*ptVolume)/(4*np.pi*rxr[ii,:])*np.exp(1j*2*np.pi*rxr[ii,:]/_lambda)
                        staticVwts[ii,:] = np.sqrt(vwt*ptVolume)/(4*np.pi*rxr[ii,:])*np.exp(1j*2*np.pi*rxr[ii,:]/_lambda)
                    elif scatMode == 'bragg':
                        print('bragg?')
                    if att:
                        staticHwts[ii,:] *= 10**(-piahr[ii,:]/10)*np.exp(1j*np.deg2rad(kvr[ii,:]))
                        staticVwts[ii,:] *= 10**(-piavr[ii,:]/10)*np.exp(1j*np.deg2rad(khr[ii,:]))
                    if ii < nRx-1:
                        Ae = getUlaWts(rxTheta[ii,:],rxPhi[ii,:],rxG)
                        staticHwts[ii,:] *= np.sqrt(Ae)
                        staticVwts[ii,:] *= np.sqrt(Ae)  
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
                staticHwts = staticHwts[:,~rmsk]
                staticVwts = staticVwts[:,~rmsk]
                pts = pts[:,~rmsk]
                windSpeed = windSpeed[:,~rmsk]
                Zv = Zv[~rmsk]
                rxr = rxr[:,~rmsk]
                    
                sortInds = np.zeros(br.shape).astype(int)
                for ii in range(nRx):
                    sortInds[ii,:] = np.argsort(br[ii,:])
                    br[ii,:] = np.sort(br[ii,:])
                    staticHwts[ii,:] = staticHwts[ii,sortInds[ii,:]]
                    staticVwts[ii,:] = staticVwts[ii,sortInds[ii,:]]
                staticHwts[np.isnan(staticHwts)] = 0
                staticVwts[np.isnan(staticVwts)] = 0
                del rxPhi,rxTheta 
                print('Simulating Observations:')
        
                for jj in range(nRx): #for each receiver
                      #Get sorted pts by bistatic range for current receiver
                      ptsr = pts[:,sortInds[jj,:]]
        
                      #Initialize object lists for storing range bin data
                      ncp = np.zeros((nr[jj]))
                      mskIndices = []
                      curHwts = []
                      curVwts = []
                      txRef = np.vstack((np.sin(txMechTheta[ll])*np.cos(txPhi)+txPos[0],
                                         np.sin(txMechTheta[ll])*np.sin(txPhi)+txPos[1],
                                         np.cos(txMechTheta[ll])*np.ones(txPhi.shape)+txPos[2]))
                      for kk in range(nr[jj]):
                          mski = inRange2(rr[jj][kk],tau,br[jj,:]) #Find points in bin
                          mskIndices.append(mski)
                          ncp[kk] = len(mski)
                          if len(mski) > 0:
                              if att:
                                  curHwts.append(staticHwts[jj,mski]*10**(-piaht[mski]/10)*np.exp(1j*np.deg2rad(kht[mski]))) #Get wts for those points
                                  curVwts.append(staticVwts[jj,mski]*10**(-piavt[mski]/10)*np.exp(1j*np.deg2rad(kvt[mski]))) 
                              else:
                                  curHwts.append(staticHwts[jj,mski]) #Get wts for those points
                                  curVwts.append(staticVwts[jj,mski]) 
                          else:
                              curHwts.append(np.array([0]))
                              curVwts.append(np.array([0]))
                      for ii in range(nAz): #For each azimuth angle
                          start = timer()
                          if np.mod(ii+1,1) == 0: #Progress display
                              timeLeft = ((nEl-ll-1)*M*nRx*nAz+(M-qq-1)*nRx*nAz+(nRx-jj-1)*nAz+(nAz-ii-1))*azTime
                              h,rem = divmod(timeLeft,3600)
                              m,sec = divmod(rem,60)
                              print(f'Pt {qq+1} Rx {jj+1}: Elevation {ll+1}/{nEl} Azimuth {ii+1}/{nAz} ETA: {int(h)} h {int(m)} m {int(sec)} s')
               
                          #Get position relative to receiver
                          (theta, phi,r) = getRelPos(txPos,txRef[:,ii],ptsr)
                          #Calculate terms for phase delay and attenuation
                          # rf = -1j*2*np.pi*r/_lambda
                          # rangeFactor = r*ne.evaluate('exp(rf)') 
                          rangeFactor = r*np.exp(-1j*2*np.pi*r/_lambda)
                          #Get transmit antenna weights
                          if dishMode:
                              if jj < nRx-1: #bistatic cases
                                  txWts = dishInter(phi)*_lambda**2/(4*np.pi*rangeFactor)
                              else: #monostatic case
                                  txWts = (dishInter(phi)*_lambda)**2/(4*np.pi*rangeFactor)
                          else:
                              if jj < nRx-1: #bistatic cases
                                  txWts = getArrayWts2(phi,theta,patMatrix,np.array((c1[qq], c2[qq])),txG)*_lambda**2/(4*np.pi*rangeFactor)
                              else:
                                  txWts = (getArrayWts2(phi,theta,patMatrix,np.array((c1[qq], c2[qq])),txG)*_lambda)**2/(4*np.pi*rangeFactor)
                          #Sum across samples and scale to form IQ
                          for kk in range(nr[jj]):
                              Hpower[jj][ii,ll,kk,qq] = calcV(mskIndices[kk],phi,txWts,curHwts[kk],Pt)
                              Vpower[jj][ii,ll,kk,qq] = calcV(mskIndices[kk],phi,txWts,curVwts[kk],Pt)
                          end = timer()
                          azTime = end-start
    elif scanType == 'rhi':
        for aa in range(nAz): #For each azimuth angle
            print(f'Azimuth {aa+1} of {nAz}')
            #os.chdir('/Users/semmerson/Documents/cm1r19.10/grinder')
            #os.chdir('/Users/semmerson/Documents/python/data/may7/may_7_250m')
            #Zv,uu,vv,ww = getWrf(wrfDate,pts[0,:]/1000+wrfOffset[0],pts[1,:]/1000+wrfOffset[1],pts[2,:]/1000+wrfOffset[2])
            uu,vv,ww,Zv,pds = getWrf(pts[0,:]/1000+wrfOffset[0],pts[1,:]/1000+wrfOffset[1],pts[2,:]/1000+wrfOffset[2],ptypes)
    
            inds = np.where(np.log10(Zv) > 0)[0]
            nPts = len(inds)
            pts = pts[:,inds]
            Zv = Zv[inds]
            uu = uu[inds]
            vv = vv[inds]
            ww = ww[inds]
            for ptype in ptypes:
                pds[ptype]['lambda'] = pds[ptype]['lambda'][inds]
                pds[ptype]['n0'] = pds[ptype]['n0'][inds]
            #Vectorize wind velocities
            windSpeed = np.zeros((3,int(nPts)))
            windSpeed[0,:] = uu
            windSpeed[1,:] = vv
            windSpeed[2,:] = ww
            
            inds = np.arange(nPts)
            np.random.shuffle(inds)
            nt = len(ptypes)
            ind_l = [inds[i::nt] for i in range(nt)]
            
            del uu,vv,ww
            if att:
                natt = 50
                piahr = np.zeros((int(nRx),int(nPts)))
                piavr = np.zeros((int(nRx),int(nPts)))
                khr = np.zeros((int(nRx),int(nPts)))
                kvr = np.zeros((int(nRx),int(nPts)))
                print('Calculating Attenuation and Phase Shift:')
                kht,kvt,piaht,piavt = getKA(pts/1000+wrfOffset[:,np.newaxis],txPos/1000+wrfOffset,natt,ptypes,A_table)
                piahr[-1,:] = np.copy(piaht)
                piavr[-1,:] = np.copy(piavt)
                khr[-1,:] = np.copy(kht)
                kvr[-1,:] = np.copy(kvt)
                for i in range(nRx-1):
                    print(f'Receiver {i+1} of {nRx}')
                    khr[i,:],kvr[i,:],piahr[i,:], piavr[i,:] = getKA(pts/1000+wrfOffset[:,np.newaxis],rxPos[i,:]/1000+wrfOffset,natt,ptypes,A_table)
            #os.chdir('/Users/semmerson/Documents/MATLAB/bistaticSimulator')
            staticHwts = np.zeros((int(nRx),int(nPts))).astype('complex64')
            staticVwts = np.zeros((int(nRx),int(nPts))).astype('complex64')
    
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
                Hpower.append(np.zeros((nAz,nEl,nr[i],M)).astype('complex64'))
                Vpower.append(np.zeros((nAz,nEl,nr[i],M)).astype('complex64'))
            
            #Calculate portion of complex scatterer weights that will not change
            #with Tx pointing angle.
            #
            #Makes simplifying assumption that Tx scan happens instantaneously
            elTime = 1
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
    
                    # else:
                    #     Ae = dishInter(rxPhi[ii,:])*_lambda**2/(4*np.pi)
                    #Get forward scatter angles
                    #thetaS = getThetaS(pts,txPos,rxPos[ii,:])
                    #Get dual-pol scattering amplitudes based on infinitesimal
                    #dipole-like reradiation
                    hwt = np.zeros(Zv.shape).astype('complex64')
                    vwt = np.zeros(Zv.shape).astype('complex64')
    
                    if tmMode:
                        for n,t in enumerate(ptypes):
                            inds = ind_l[n]
                            weights = getTmatrixWeights(S_table[t],txPos,rxPos[ii,:],pts[:,inds],pds[t]['lambda'][inds],pds[t]['n0'][inds])
                            hwt[inds] = np.squeeze(weights[0])
                            vwt[inds] = np.squeeze(weights[1])
    
                    else:
                        weights = bistaticWeights(txPos,rxPos[ii,:],pts)
                        hwt = np.squeeze(weights[0])
                        vwt = np.squeeze(weights[1])
            
                    # Do the "receive" half of the radar range equation (doesn't
                    # change with Tx pointing angle.  Convert from Z or Cn^2 to
                    # reflectivity (eta) as necessary
                    if scatMode == 'rayleigh':
                        staticHwts[ii,:] = np.sqrt(hwt*ptVolume)/(4*np.pi*rxr[ii,:])*np.exp(1j*2*np.pi*rxr[ii,:]/_lambda)
                        staticVwts[ii,:] = np.sqrt(vwt*ptVolume)/(4*np.pi*rxr[ii,:])*np.exp(1j*2*np.pi*rxr[ii,:]/_lambda)
                    elif scatMode == 'bragg':
                        print('bragg?')
                    if att:
                        staticHwts[ii,:] *= 10**(-piahr[ii,:]/10)*np.exp(1j*np.deg2rad(kvr[ii,:]))
                        staticVwts[ii,:] *= 10**(-piavr[ii,:]/10)*np.exp(1j*np.deg2rad(khr[ii,:]))
                    if ii < nRx-1:
                        Ae = getUlaWts(rxTheta[ii,:],rxPhi[ii,:],rxG)*_lambda**2/(4*np.pi)
                        staticHwts[ii,:] *= np.sqrt(Ae)
                        staticVwts[ii,:] *= np.sqrt(Ae)  
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
                    staticHwts[ii,:] = staticHwts[ii,sortInds[ii,:]]
                    staticVwts[ii,:] = staticVwts[ii,sortInds[ii,:]]
                staticHwts[np.isnan(staticHwts)] = 0
                staticVwts[np.isnan(staticVwts)] = 0
                del rxPhi,rxTheta 
                print('Simulating Observations:')
        
                for jj in range(nRx): #for each receiver
                      #Get sorted pts by bistatic range for current receiver
                      ptsr = pts[:,sortInds[jj,:]]
        
                      #Initialize object lists for storing range bin data
                      ncp = np.zeros((nr[jj]))
                      mskIndices = []
                      curHwts = []
                      curVwts = []
                      for kk in range(nr[jj]):
                          mski = inRange2(rr[jj][kk],tau,br[jj,:]) #Find points in bin
                          mskIndices.append(mski)
                          ncp[kk] = len(mski)
                          if len(mski) > 0:
                              if att:
                                  curHwts.append(staticHwts[jj,mski]*10**(-piaht[mski]/10)*np.exp(-1j*np.deg2rad(kht[mski]))) #Get wts for those points
                                  curVwts.append(staticVwts[jj,mski]*10**(-piavt[mski]/10)*np.exp(-1j*np.deg2rad(kvt[mski]))) 
                              else:
                                  curHwts.append(staticHwts[jj,mski]) #Get wts for those points
                                  curVwts.append(staticVwts[jj,mski]) 
                          else:
                              curHwts.append(np.array([0]))
                              curVwts.append(np.array([0]))
                      for ll in range(nEl): #For each elevation angle
                          start = timer()
                          txRef = np.vstack((np.sin(txMechTheta[ll])*np.cos(txPhi)+txPos[0],
                                         np.sin(txMechTheta[ll])*np.sin(txPhi)+txPos[1],
                                         np.cos(txMechTheta[ll])*np.ones(txPhi.shape)+txPos[2]))
                          if np.mod(ii+1,1) == 0: #Progress display
                              timeLeft = ((nAz-ii-1)*M*nRx*nEl+(M-qq-1)*nRx*nEl+(nRx-jj-1)*nEl+(nEl-ll-1))*elTime
                              h,rem = divmod(timeLeft,3600)
                              m,sec = divmod(rem,60)
                              print(f'Pt {qq+1} Rx {jj+1}: Elevation {ll+1}/{nEl} Azimuth {aa+1}/{nAz} ETA: {int(h)} h {int(m)} m {int(sec)} s')
               
                          #Get position relative to receiver
                          (theta, phi,r) = getRelPos(txPos,txRef[:,ii],ptsr)
                          #Calculate terms for phase delay and attenuation
                          # rf = -1j*2*np.pi*r/_lambda
                          # rangeFactor = r*ne.evaluate('exp(rf)') 
                          rangeFactor = r*np.exp(-1j*2*np.pi*r/_lambda)
                          #Get transmit antenna weights
                          if dishMode:
                              if jj < nRx-1: #bistatic cases
                                  txWts = dishInter(phi)/rangeFactor
                              else: #monostatic case
                                  txWts = (dishInter(phi)*_lambda)**2/(4*np.pi*rangeFactor)
                          else:
                              if jj < nRx-1: #bistatic cases
                                  txWts = getArrayWts2(phi,theta,patMatrix,np.array((c1[qq], c2[qq])),txG)/rangeFactor
                              else:
                                  txWts = (getArrayWts2(phi,theta,patMatrix,np.array((c1[qq], c2[qq])),txG)*_lambda)**2/(4*np.pi*rangeFactor)
                          #Sum across samples and scale to form IQ
                          for kk in range(nr[jj]):
                              Hpower[jj][aa,ll,kk,qq] = calcV(mskIndices[kk],phi,txWts,curHwts[kk],Pt)
                              Vpower[jj][aa,ll,kk,qq] = calcV(mskIndices[kk],phi,txWts,curVwts[kk],Pt)
                          end = timer()
                          azTime = end-start
    
    return Hpower, Vpower, rr

#beams = np.array([[False,True,False],[False,False,False],[False,False,True],[True,False,False]])
beams = np.array([[True,False,False]])
mode = beams[0,:]

# This is where the Tx and Rx positions are configured, relative to the simulation origin
# The last x,y,z entries for Rx MUST be 0 to provide a monostatic location 
x = np.array([-1.0,0.0])*1e3
y = np.array([0.0,0.0])*1e3
z = np.zeros(len(x))
# Tx position
txPos = np.array([0.0,-10.0,0.0])*1e3

pos = np.stack((x,y,z)).T+txPos
radarStruct = makeRadarStruct(txPos=txPos,rxPos=pos,modes=mode)
wxStruct = makeWxStruct()

Hpower, Vpower, rr = simulate(radarStruct,wxStruct)

txPos = radarStruct['txPos']
radarStruct['rxPos'] = pos
rxPos = pos
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
idealPat = radarStruct['idealPat']
whitening = radarStruct['whitening']
dishMode = radarStruct['dishMode']
tmMode = radarStruct['tmMode']
saveMode = radarStruct['saveMode']

c = 3e8 
nRx = rxPos.shape[0]
nEl = txMechEl.shape[0]
Ts = 1/fs 
T0 = 290
Trec = T0*(10**(Frx/10)-1) 
B = fs 
k = 1.381e-23 
N0 = k*(Tant+Trec)*B*1e9
cn = 1e-14
noiseApplied = True
# f = loadmat('wrfDirectory.mat') #load NWP directory
# XX = f['XX'][:]-wrfOffset[0]
# YY = f['YY'][:]-wrfOffset[1]
# ZZ = f['ZZ'][:]

hpp = []
vpp = []
rhohv = []
snr = []
df = []
phi = []

vels = []
xx = []
yy = []
zz = []
br = []
beta = []
snrThresh = 10**(0/10)
brThresh = 1e3

txrxDist = np.sqrt(np.sum((txPos-rxPos)**2,axis=1))
rxPhi = np.arctan2(rxPos[:,1]-txPos[1],rxPos[:,0]-txPos[0])

rxPos2 = np.array([np.cos(rxPhi), np.sin(rxPhi), np.zeros(len(txrxDist))])*txrxDist
txPos2 = np.zeros(3)
for i in range(nRx):
    if ~noiseApplied:
        noiseMatH = np.sqrt(N0)/np.sqrt(2)*(np.random.normal(0,1,Hpower[i].shape) + 1j*np.random.normal(0,1,Hpower[i].shape))
        #noiseMatH = np.zeros(Hpower[i].shape)
        Hpower[i] = Hpower[i]+noiseMatH
        noiseMatV = np.sqrt(N0)/np.sqrt(2)*(np.random.normal(0,1,Vpower[i].shape) + 1j*np.random.normal(0,1,Vpower[i].shape))
        #noiseMatV = np.zeros(Vpower[i].shape)
        Vpower[i] = Vpower[i]+noiseMatV

    hpp.append(np.sum(np.abs(Hpower[i])**2,axis=3)/radarStruct['M'])
    vpp.append(np.sum(np.abs(Vpower[i])**2,axis=3)/radarStruct['M'])
    rho = (np.sum(np.abs((Hpower[i])*np.conj(Vpower[i])),axis=3)/radarStruct['M'])/np.sqrt(hpp[i]*vpp[i])
    rhohv.append(rho)
    phidp = 0.5*np.rad2deg(np.angle(1/M*np.sum(np.conj(Hpower[i])*Vpower[i],axis=3)))
    phi.append(phidp)
    snr.append(vpp[i]/N0)
    #f = unwrap_phase(np.angle(np.sum(np.conj(Vpower[i][:,:,:,:-1])*Vpower[i][:,:,:,1:],axis=3)))
    f = np.angle(np.sum(np.conj(Vpower[i][:,:,:,:-1])*Vpower[i][:,:,:,1:],axis=3))
    hpp[i][snr[i] < snrThresh] = np.nan
    vpp[i][snr[i] < snrThresh] = np.nan
    f[snr[i] < snrThresh] = np.nan
    phi[i][snr[i] < snrThresh] = np.nan
    rhohv[i][snr[i] < snrThresh] = np.nan
    df.append(f/(2*np.pi*radarStruct['prt']))
    # br.append(np.reshape(getBistaticRanges(txPos2,rxPos2[:,i],np.array((xx[i].flatten(),yy[i].flatten(),zz[i].flatten()))),xx[i].shape))

    vel = np.zeros(df[i].shape)
    xx0 = np.zeros(df[i].shape)
    yy0 = np.zeros(df[i].shape)
    zz0 = np.zeros(df[i].shape)
    beta0 = np.zeros(df[i].shape)
    for l in range(nEl):
        pts = localize(90-txAz,radarStruct['txMechEl'][l]*np.ones(txAz.shape),rr[i]/c,rxPos2[:,i],txPos2)
        vel[:,l,:] = freq2vel(df[i][:,l,:],pts[:,:,0],pts[:,:,1],pts[:,:,2],rxPos2[:,i],txPos2,_lambda)
        xx0[:,l,:] = pts[:,:,0]
        yy0[:,l,:] = pts[:,:,1]
        zz0[:,l,:] = pts[:,:,2]
        beta0[:,l,:] = getBa(pts[:,:,0],pts[:,:,1],pts[:,:,2],rxPos2[:,i])
    xx0[np.isnan(xx0)] = 0.0
    yy0[np.isnan(yy0)] = 0.0
    zz0[np.isnan(zz0)] = 0.0
    beta.append(beta0)
    vels.append(vel)
    xx.append(xx0)
    yy.append(yy0)
    zz.append(zz0)
noiseApplied = True

if saveMode:
    radars = {}
    # radars['Hpower'] = [arr.astype('complex64') for arr in Hpower]
    # radars['Vpower'] = [arr.astype('complex64') for arr in Vpower]
    radars['hpp'] = [arr.astype('float32') for arr in hpp]
    radars['vpp'] = [arr.astype('float32') for arr in vpp]
    radars['rr'] = rr
    radars['snr'] = [arr.astype('float32') for arr in snr]
    radars['df'] = [arr.astype('float32') for arr in df]
    radars['phi'] = [arr.astype('float32') for arr in phi]
    # radars['vel'] = [arr.astype('float32') for arr in vels]
    # radars['xx'] = xx
    # radars['yy'] = yy
    # radars['zz'] = zz
    radars['br'] = br
    radars['beta'] = beta
    #os.chdir('/Users/semmerson/Documents/python/data/PRS')
    if dishMode:
        mode = 'dish'
    elif idealPat:
        mode = 'ideal'
    elif whitening:
        mode = 'whitened'
    else:
        mode = 'unwhitened'
    if _lambda < 0.045:
        band = 'X'
    elif _lambda < 0.075:
        band = 'C'
    else:
        band = 'S'
    att = ''
    if radarStruct['attenuationMode']:
        att = 'A'
    if len(radarStruct['txMechEl']) > 1:
        fName = f'{band}_{radarStruct["rxPos"].shape[0]}rx_{len(radarStruct["txAz"])}az_{radarStruct["txMechEl"][0]}-{radarStruct["txMechEl"][-1]}el_{radarStruct["M"]}M_{mode}_{2*xSize}km+{mode}+{att}.xz'
    else:
        fName = f'{band}_{radarStruct["rxPos"].shape[0]}rx_{len(radarStruct["txAz"])}az_{radarStruct["txMechEl"]}el_{radarStruct["M"]}M_{2*xSize}km+{mode}+{att}.xz'
    with lzma.open(fName,'wb') as f:
        pickle.dump(radarStruct,f)
        pickle.dump(wxStruct,f)
        pickle.dump(radars,f)
    del radars
